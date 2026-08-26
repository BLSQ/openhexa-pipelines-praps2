"""Push the consolidated dataset (produced by the consolidate_legacy_surveys pipeline)
into the live fiche_simplifiee_infrastructures Kobo form as new submissions
"""

import io
import re
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.kobo import Api

import config


@pipeline(name="pousser-la-db-consolidee-vers-kobo")
@parameter(
    "input_file",
    name="Fichier consolidé",
    help="Chemin (relatif au workspace) du fichier consolidé à pousser vers Kobo",
    type=str,
    default="data/kobo/surveys/CONSOLIDATED_DB_with_infrastructure_id.xlsx",
)
@parameter(
    "pacing_seconds",
    name="Délai entre soumissions (secondes)",
    help="Pause entre chaque soumission, pour ne pas saturer la connexion",
    type=float,
    default=0.4,
)
def push_consolidated_db_to_kobo(input_file: str, pacing_seconds: float):
    """Fetch the live form's field->xpath map, load the consolidated dataset, and push
    every row as a new Kobo submission.
    """
    api = Api(url=config.kobo_connector["url"])
    api.authenticate(token=config.kobo_connector["token"])
    form_uid = get_survey_uid(api, config.FORM_NAME)

    field_xpath_map, form_version = fetch_field_xpath_map(api, form_uid)
    records = load_records(Path(workspace.files_path, input_file))
    push_records(api, form_uid, form_version, records, field_xpath_map, pacing_seconds)


def get_survey_uid(api: Api, kobo_name: str) -> str:
    """Look up a survey's asset uid by its exact live Kobo project name.

    Avoids hard-coding an opaque uid in config.py -- `api.surveys` is a cached property,
    so this only costs one `assets.json` call, not one per lookup.
    """
    for survey in api.surveys:
        if survey["name"] == kobo_name:
            return survey["uid"]
    raise ValueError(f"No Kobo survey found with name {kobo_name!r}")


def fetch_field_xpath_map(api: Api, form_uid: str) -> tuple:
    """Read the live form's own schema once, returning (field_xpath_map, form_version).

    field_xpath_map maps each flat field name to its full group-nested xpath (e.g.
    "DATE" -> "group1/DATE"), so submissions land under the correct XForm group
    structure regardless of which group a field lives in.

    form_version (content.settings.version) is the deployed form's own version string,
    used as the ODK submission's `version` attribute -- fetched live rather than
    hard-coded, since it's a property of the current deployment, not of this connection,
    and confirmed to drift (a previously hard-coded value here was stale).
    """
    survey = api.get_survey(form_uid)
    content = survey.meta.get("content", {})

    field_xpath_map = {
        element["name"]: element["$xpath"]
        for element in content.get("survey", [])
        if "name" in element and "$xpath" in element
    }
    form_version = content.get("settings", {}).get("version")
    current_run.log_info(
        f"Mapped {len(field_xpath_map)} form fields to their group paths "
        f"(form version {form_version})"
    )
    return field_xpath_map, form_version


def load_records(path: Path) -> list:
    """Read the consolidated dataset and return one dict per row."""
    df = pl.read_excel(path)
    current_run.log_info(f"Loaded {len(df)} records from {path}")
    return df.to_dicts()


def clean_record(row: dict) -> dict:
    """Drop null/empty/system (underscore-prefixed) fields, and assign a fresh
    instanceID so Kobo doesn't treat this push as a duplicate of a prior attempt.
    """
    clean = {
        k: v
        for k, v in row.items()
        if v is not None
        and str(v).strip() != ""
        and str(v) != "None"
        and not k.startswith("_")
    }
    clean["meta/instanceID"] = f"uuid:{uuid.uuid4()}"
    return clean


def sanitize_tag(name: str) -> str:
    """Make `name` usable as a valid XML element tag."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(name))
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def row_to_odk_xml(
    id_string: str, version_string: str, record: dict, xpath_map: dict
) -> bytes:
    """Convert a flat {field: value} record into nested ODK submission XML, injecting
    each field under its real Kobo group path via `xpath_map`.
    """
    root = ET.Element(sanitize_tag(id_string), id=id_string, version=version_string)

    for key, value in record.items():
        target_path = key
        if "/" not in key and "." not in key:
            target_path = xpath_map.get(key, key)

        normalized_key = target_path.strip("/").strip(".").replace(".", "/")
        if not normalized_key:
            continue

        parts = normalized_key.split("/")
        current_node = root
        for part in parts[:-1]:
            sanitized_part = sanitize_tag(part)
            found = current_node.find(sanitized_part)
            current_node = (
                found
                if found is not None
                else ET.SubElement(current_node, sanitized_part)
            )

        leaf = ET.SubElement(current_node, sanitize_tag(parts[-1]))
        # Strip trailing Excel zero-timestamps out of raw date strings
        val_str = str(value)
        if " 00:00:00" in val_str:
            val_str = val_str.split(" ")[0]
        leaf.text = val_str

    xml_str = ET.tostring(root, encoding="utf-8")
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + xml_str


def push_records(
    api: Api,
    form_uid: str,
    form_version: str,
    records: list,
    field_xpath_map: dict,
    pacing_seconds: float,
):
    """Push every record as a new Kobo submission, one at a time, pacing between
    requests to avoid saturating the connection.
    """
    success_count = 0
    error_count = 0
    for idx, row in enumerate(records):
        clean_data = clean_record(row)
        xml_bytes = row_to_odk_xml(form_uid, form_version, clean_data, field_xpath_map)
        files = {
            "xml_submission_file": ("submission.xml", io.BytesIO(xml_bytes), "text/xml")
        }

        response = api.session.post(config.SUBMISSION_URL, files=files)
        if response.status_code in (200, 201, 202):
            success_count += 1
            current_run.log_info(
                f"[{idx + 1}/{len(records)}] Success: {clean_data['meta/instanceID']}"
            )
        else:
            error_count += 1
            current_run.log_warning(
                f"[{idx + 1}/{len(records)}] Failed (status {response.status_code}): "
                f"{response.text}"
            )

        time.sleep(pacing_seconds)

    current_run.log_info(
        f"Done: {success_count} succeeded, {error_count} failed out of {len(records)}"
    )


if __name__ == "__main__":
    push_consolidated_db_to_kobo()
