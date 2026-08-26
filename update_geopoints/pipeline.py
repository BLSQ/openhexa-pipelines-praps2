from openhexa.sdk import current_run, pipeline
import requests
import csv
import io
import config


@pipeline(name="update_geopoints")
def update_geopoints():
    """
    This pipeline updates the live database of geopoints underlying the simplified KoboToolbox form
    It produces the following steps:
      - Fetches the latest submissions from the simplified KoboToolbox form
      - Processes the data to generate a geopoints dataset and convert it to CSV format
      - Updates the corresponding CSV file in the form's media files
      - Redeploys the form to ensure that the changes are reflected in the Android app
    """
    submissions = download_kobo_form_data(config.KOBO_BASE_URL, config.FORM_ID)
    submissions = delete_duplicate_submissions(submissions)
    geopoints = generate_geopoints_dataset(submissions)
    geopoints_csv = convert_to_csv(geopoints)
    delete_current_geopoints_dataset(
        config.KOBO_BASE_URL, config.FORM_ID, config.HEADERS, "points_live"
    )
    upload_updated_geopoints_dataset(
        config.KOBO_BASE_URL,
        config.FORM_ID,
        config.HEADERS,
        "points_live",
        geopoints_csv,
    )
    redeploy_kobo_form(config.KOBO_BASE_URL, config.FORM_ID, config.HEADERS)


def download_kobo_form_data(base_url: str, form_id: str) -> list:
    """
    Downloads the latest submissions from the simplified KoboToolbox form.

    Args:
        base_url (str): The base URL of the KoboToolbox API.
        form_id (str): The ID of the KoboToolbox form.

    Returns:
        list: A list of all submissions retrieved from the form.
    """
    current_run.log_info("Downloading Kobo data...")
    data_url = f"{base_url}/api/v2/assets/{form_id}/data.json"
    submissions = []

    # Loop through all pages until "next" is null
    while data_url:
        response = requests.get(data_url, headers=config.HEADERS)
        response.raise_for_status()
        data = response.json()
        submissions.extend(data.get("results", []))
        data_url = data.get("next")

    current_run.log_info(
        f"Successfully downloaded {len(submissions)} total submissions."
    )

    return submissions


def delete_duplicate_submissions(submissions: list) -> list:
    """
    Deletes duplicate submissions based on the INFRASTRUCTURE ID, keeping only the latest submission for each ID.

    Args:
        submissions (list): A list of submissions retrieved from the form.

    Returns:
        list: A list of unique submissions with duplicates removed.
    """
    current_run.log_info("Deleting duplicate submissions based on INFRASTRUCTURE ID...")
    submissions.sort(key=lambda x: x.get("starttime", ""), reverse=True)
    unique_submissions = {}
    for sub in submissions:
        infra_id = sub.get("group2-1/INFRASTRUCTURE_ID")
        if infra_id and infra_id not in unique_submissions:
            unique_submissions[infra_id] = sub

    submissions = list(unique_submissions.values())
    current_run.log_info(
        f"Successfully deleted duplicates. {len(submissions)} unique submissions remain."
    )

    return submissions


def generate_geopoints_dataset(submissions: list) -> list:
    """
    Generates a geopoints dataset from the submissions.

    Args:
        submissions (list): A list of submissions retrieved from the form.

    Returns:
        list: A list of dictionaries representing the geopoints dataset.
    """
    current_run.log_info("Generating geopoints dataset...")

    geopoints = []
    for sub in submissions:
        # Check if the record has an ID, Label, and Geometry (Coordinates)
        infra_id = sub.get("group2-1/INFRASTRUCTURE_ID")
        luv3_label = sub.get("group2-1/LUV3")
        luv5_label = sub.get("group2-1/LUV5")
        luv6_geo = sub.get("group2-1/LUV6")
        type_acronym = sub.get("group2/TYPE_ACRONYM")

        # Only include valid points
        if infra_id:
            geopoints.append(
                {
                    "LUV3": luv3_label,
                    "LUV5": luv5_label,
                    "geometry": luv6_geo,
                    "INFRASTRUCTURE_ID": infra_id,
                    "TYPE_ACRONYM": type_acronym,
                }
            )

    if not geopoints:
        current_run.log_info("No valid coordinates found in the database. Exiting.")
        return

    # Sort alphabetically by INFRASTRUCTURE_ID
    geopoints.sort(
        key=lambda point: (
            point["INFRASTRUCTURE_ID"].casefold(),
            point["INFRASTRUCTURE_ID"],
        )
    )

    current_run.log_info(
        f"Successfully generated geopoints dataset with {len(geopoints)} points."
    )

    return geopoints


def convert_to_csv(geopoints: list) -> str:
    """
    Converts the geopoints dataset to a CSV format.

    Args:
        geopoints (list): A list of dictionaries representing the geopoints dataset.

    Returns:
        str: The CSV content as a string.
    """
    current_run.log_info("Converting geopoints dataset to CSV...")

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["LUV3", "LUV5", "geometry", "TYPE_ACRONYM", "INFRASTRUCTURE_ID"],
    )
    writer.writeheader()
    writer.writerows(geopoints)
    csv_content = output.getvalue()
    current_run.log_info("Successfully converted geopoints dataset to CSV.")

    return csv_content


def delete_current_geopoints_dataset(
    base_url: str, form_id: str, headers: dict, geopoints_file_name: str
):
    """
    Deletes the current geopoints dataset CSV file from the Kobo form.

    Args:
        base_url (str): The base URL of the Kobo API.
        form_id (str): The ID of the Kobo form.
        headers (dict): The headers for the API request.
        geopoints_file_name (str): The name of the geopoints CSV file to delete.
    """
    files_url = f"{base_url}/api/v2/assets/{form_id}/files/"
    files_response = requests.get(files_url, headers=headers)
    files_response.raise_for_status()

    for file_obj in files_response.json().get("results", []):
        if file_obj.get("metadata", {}).get("filename") == f"{geopoints_file_name}.csv":
            current_run.log_info(f"Deleting old CSV (UID: {file_obj['uid']})...")
            delete_url = f"{files_url}{file_obj['uid']}/"
            requests.delete(delete_url, headers=headers)


def upload_updated_geopoints_dataset(
    base_url: str,
    form_id: str,
    headers: dict,
    geopoints_file_name: str,
    new_geopoints_csv_content: str,
):
    """
    Uploads the updated geopoints dataset CSV file to the Kobo form.

    Args:
        base_url (str): The base URL of the Kobo API.
        form_id (str): The ID of the Kobo form.
        headers (dict): The headers for the API request.
        geopoints_file_name (str): The name of the geopoints CSV file to upload.
        new_geopoints_csv_content (str): The content of the new geopoints CSV file.
    """
    current_run.log_info(f"Uploading new {geopoints_file_name}.csv...")

    files_data = {
        "description": "Automated map points update",
        "file_type": "form_media",
        "metadata": f'{{"filename": "{geopoints_file_name}.csv"}}',
    }

    files_upload = {
        "content": (f"{geopoints_file_name}.csv", new_geopoints_csv_content, "text/csv")
    }
    files_url = f"{base_url}/api/v2/assets/{form_id}/files/"

    upload_response = requests.post(
        files_url, headers=headers, data=files_data, files=files_upload
    )

    upload_response.raise_for_status()

    current_run.log_info(
        f"Success! The updated geopoints dataset {geopoints_file_name}.csv has been updated."
    )


def redeploy_kobo_form(base_url: str, form_id: str, headers: dict):
    """Redeploys the Kobo form."""
    current_run.log_info("Redeploying Kobo form...")
    deploy_url = f"{base_url}/api/v2/assets/{form_id}/deployment/"
    deploy_response = requests.patch(deploy_url, headers=headers)

    deploy_response.raise_for_status()
    current_run.log_info("Success! Form redeployed and ready for Android.")


if __name__ == "__main__":
    update_geopoints()
