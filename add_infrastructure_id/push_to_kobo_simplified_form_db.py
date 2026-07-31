import polars as pl
import requests
import uuid
import time
import io
import re
import xml.etree.ElementTree as ET
import config


# --- Setup Headers ---
headers = {"Authorization": f"Token {config.API_TOKEN}"}

# -------------------------------------------------------------------------
# 1. Fetch the Form Schema to Automatically Map Field XPaths
# -------------------------------------------------------------------------
print("Fetching live form asset schema to build group path mappings...")
schema_response = requests.get(config.ASSET_SCHEMA_URL, headers=headers)
schema_response.raise_for_status()
survey_elements = schema_response.json().get("content", {}).get("survey", [])

# Build a lookup map: flat_name -> nested/group/path
# Example: "DATE" -> "group1/DATE"
field_xpath_map = {}
for element in survey_elements:
    if "name" in element and "$xpath" in element:
        field_xpath_map[element["name"]] = element["$xpath"]

print(
    f"Successfully mapped {len(field_xpath_map)} form fields to their structural groups."
)


# --- Helper Function: Clean and force strings to be valid XML elements ---
def sanitize_tag(name):
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(name))
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


# --- Helper Function: Convert flat headers into nested ODK XML ---
def row_to_odk_xml(id_string, version_string, record, xpath_map):
    root = ET.Element(sanitize_tag(id_string), id=id_string, version=version_string)

    for key, value in record.items():
        # Dynamic Group Injection: Look up the flat field name's true group path
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
            if found is None:
                current_node = ET.SubElement(current_node, sanitized_part)
            else:
                current_node = found

        # Append clean value leaf node
        sanitized_leaf = sanitize_tag(parts[-1])
        leaf = ET.SubElement(current_node, sanitized_leaf)

        # Strip trailing Excel zero-timestamps out of raw dates strings
        val_str = str(value)
        if " 00:00:00" in val_str:
            val_str = val_str.split(" ")[0]

        leaf.text = val_str

    xml_str = ET.tostring(root, encoding="utf-8")
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + xml_str


# -------------------------------------------------------------------------
# 2. Load and Stream Excel Data Bundle
# -------------------------------------------------------------------------
print(
    f"Reading historical data from Excel workbook: {config.OUTPUT_PATH}/CONSOLIDATED_DB_with_infrastructure_id.xlsx..."
)
df = pl.read_excel(f"{config.OUTPUT_PATH}/CONSOLIDATED_DB_with_infrastructure_id.xlsx")

records = df.to_dicts()

success_count = 0
error_count = 0

print(f"Starting grouped data transmission of {len(records)} records...")
print("-" * 50)

# --- Upload Loop ---
for idx, row in enumerate(records):
    # Clean the row payload: Drop missing values, empty fields, and system metrics
    clean_data = {}
    for k, v in row.items():
        if v is None or str(v).strip() == "" or str(v) == "None":
            continue
        if k.startswith("_"):
            continue
        clean_data[k] = v

    # Enforce a brand new unique instanceID to clear duplicate checking locks
    instance_id = f"uuid:{uuid.uuid4()}"
    clean_data["meta/instanceID"] = instance_id

    # Construct the clean XML tree with active xpath mapping dictionary injection
    xml_bytes = row_to_odk_xml(
        config.FORM_ID_STRING, config.FORM_VERSION, clean_data, field_xpath_map
    )

    # Pack the binary bytes array into a compliant form data mapping dictionary
    files = {
        "xml_submission_file": ("submission.xml", io.BytesIO(xml_bytes), "text/xml")
    }

    # Post to the submission gateway
    response = requests.post(config.SUBMISSION_URL, headers=headers, files=files)

    # Response tracking logs
    if response.status_code in [200, 201, 202]:
        success_count += 1
        print(f"[{idx + 1}/{len(records)}] Success: Synchronized entry {instance_id}")
    else:
        error_count += 1
        print(
            f"[{idx + 1}/{len(records)}] Failed (Status {response.status_code}): {response.text}"
        )

    # Pacing safety delay to protect the socket buffer connection
    time.sleep(0.4)

print("-" * 50)
print(f"Upload Pipeline Finished!")
print(f"Successful Syncs: {success_count} | Failures Encountered: {error_count}")
