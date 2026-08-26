import os
from pathlib import Path

from dotenv import load_dotenv
from openhexa.sdk import workspace

# set up env (for local testing only)
dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
os.environ["HEXA_WORKSPACE"] = os.getenv("HEXA_WORKSPACE")
os.environ["HEXA_SERVER_URL"] = os.getenv("HEXA_SERVER_URL")
os.environ["HEXA_TOKEN"] = os.getenv("HEXA_TOKEN")

# Kobo Connector Instances
connection = workspace.custom_connection("kobo_api")
kobo_connector = {
    "url": connection.url,
    "token": connection.token,
}

# Consolidated form specs
FORM_NAME = "FICHE_SIMPLIFIEE_INFRASTRUCTURES"
_base_url = connection.url.rstrip("/")
if _base_url.endswith("/api/v2"):
    _base_url = _base_url[: -len("/api/v2")]
SUBMISSION_URL = f"{_base_url}/submission"
