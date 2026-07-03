import requests
import csv
import io
import config


# ==========================================
# CONFIGURATION
# ==========================================
def sync_kobo_map_data():
    print("Fetching live data from KoboToolbox...")

    # 1. Download the live submissions
    data_url = f"{config.KOBO_BASE_URL}/api/v2/assets/{config.FORM_ID_STRING}/data.json"
    submissions = []

    # Loop through all pages until "next" is null
    while data_url:
        response = requests.get(data_url, headers=config.HEADERS)
        response.raise_for_status()
        data = response.json()
        submissions.extend(data.get("results", []))
        data_url = data.get("next")

    print(f"Successfully downloaded {len(submissions)} total submissions.")

    # delete duplicates of INFRASTRUCTURE_ID (keep the most recent by starttime)
    submissions.sort(key=lambda x: x.get("starttime", ""), reverse=True)
    unique_submissions = {}
    for sub in submissions:
        infra_id = sub.get("group3/INFRASTRUCTURE_ID")
        if infra_id and infra_id not in unique_submissions:
            unique_submissions[infra_id] = sub

    submissions = list(unique_submissions.values())

    # 2. Extract and format the data
    map_points = []
    for sub in submissions:
        # Check if the record has an ID, Label, and Geometry (Coordinates)
        infra_id = sub.get("group3/INFRASTRUCTURE_ID")
        luv3_label = sub.get("group3/LUV3")
        luv5_label = sub.get("group3/LUV5")
        luv6_geo = sub.get("group3/LUV6")
        type_acronym = sub.get("group2/TYPE_ACRONYM")

        # Only include valid points
        if infra_id and luv5_label and luv6_geo:
            map_points.append(
                {
                    "LUV3": luv3_label,
                    "LUV5": luv5_label,
                    "geometry": luv6_geo,
                    "INFRASTRUCTURE_ID": infra_id,
                    "TYPE_ACRONYM": type_acronym,
                }
            )

    if not map_points:
        print("No valid coordinates found in the database. Exiting.")
        return

    # 3. Create the CSV file in memory
    print(f"Formatting {len(map_points)} records into CSV...")
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["LUV3", "LUV5", "geometry", "TYPE_ACRONYM", "INFRASTRUCTURE_ID"],
    )
    writer.writeheader()
    writer.writerows(map_points)
    csv_content = output.getvalue()

    # 4. Find and delete the existing CSV in Kobo's Media files
    files_url = f"{config.KOBO_BASE_URL}/api/v2/assets/{config.FORM_ID_STRING}/files/"
    files_response = requests.get(files_url, headers=config.HEADERS)
    files_response.raise_for_status()

    for file_obj in files_response.json().get("results", []):
        if file_obj.get("metadata", {}).get("filename") == "points_live.csv":
            print(f"Deleting old CSV (UID: {file_obj['uid']})...")
            delete_url = f"{files_url}{file_obj['uid']}/"
            requests.delete(delete_url, headers=config.HEADERS)

    # 5. Upload the new CSV
    print("Uploading new points_live.csv...")

    # We must properly format the payload for Kobo's multipart/form-data requirements
    files_data = {
        "description": "Automated map points update",
        "file_type": "form_media",
        "metadata": '{"filename": "points_live.csv"}',
    }

    files_upload = {"content": ("points_live.csv", csv_content, "text/csv")}

    # Kobo requires this specific request format for asset file uploads
    upload_response = requests.post(
        files_url, headers=config.HEADERS, data=files_data, files=files_upload
    )

    upload_response.raise_for_status()

    print("Success! The map data has been updated.")

    # 6. Redeploy the form to push the new CSV to the mobile app
    print("Redeploying the form...")
    deploy_url = (
        f"{config.KOBO_BASE_URL}/api/v2/assets/{config.FORM_ID_STRING}/deployment/"
    )
    deploy_response = requests.patch(deploy_url, headers=config.HEADERS)

    deploy_response.raise_for_status()
    print("Success! Form redeployed and ready for Android.")


if __name__ == "__main__":
    sync_kobo_map_data()
