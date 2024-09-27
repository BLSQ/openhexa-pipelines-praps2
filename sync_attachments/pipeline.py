import os
from pathlib import Path

import polars as pl
from google.cloud import storage
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.kobo import Api

SURVEYS = [
    "marches_a_betail",
    "parcs_de_vaccination",
    "points_d_eau",
    "unites_veterinaires",
    "fourrages_cultives",
    "gestion_durable_des_paysages",
    "activites_generatrices_de_revenus",
]


@pipeline("sync-attachments", name="sync-attachments")
@parameter(
    "input_dir",
    name="Dossier d'entrée",
    help="Répertoire où se trouvent les fiches extraites",
    type=str,
    default="data/kobo/surveys",
)
@parameter(
    "output_dir",
    name="Dossier de sortie",
    help="Répertoire où enregistrer les images",
    type=str,
    default="data/kobo/pictures",
)
@parameter(
    "bucket", name="GCS bucket", help="Nom du bucket GCS où stocker les images", type=str, default="hexa-public-praps"
)
def sync_attachments(input_dir: str, output_dir: str, bucket: str):
    input_dir = Path(workspace.files_path, input_dir)
    output_dir = Path(workspace.files_path, output_dir)
    task1 = download_attachments(input_dir, output_dir)
    upload_attachments(output_dir, bucket, wait=task1)


def authenticate() -> storage.Client:
    """Authenticate to GCS bucket."""
    with open("/tmp/gcs.json", "w") as f:
        f.write(workspace.gcs_connection("hexa-public-praps").service_account_key)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcs.json"
    current_run.log_info("Authenticated to GCS bucket `hexa-public-praps`")
    return storage.Client()


def _download(url: str, dst_dir: Path, api: Api):
    with api.session.get(url, stream=True) as r:
        if "content-disposition" not in r.headers:
            current_run.log_warning(f"Could not download attachment `{url}`")
            return
        fname = r.headers["Content-Disposition"].split("filename=")[-1]
        fpath = dst_dir / fname
        if not fpath.exists():
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024**2):
                    if chunk:
                        f.write(chunk)


def download_attachments(input_dir: Path, output_dir: Path):
    """Download attachments for a given survey."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    con = workspace.custom_connection("kobo_api")
    api = Api(url=con.url)
    api.authenticate(token=con.token)
    current_run.log_info("Connected to KoboToolbox API at url: {}".format(con.url))

    for survey in SURVEYS:
        fname = input_dir / f"{survey}.parquet"
        if not fname.exists():
            current_run.log_warning(f"File not found for survey `{survey}`")

        df = pl.read_parquet(fname)
        for row in df.iter_rows(named=True):
            attachments = row.get("_attachments")
            for attachment in attachments:
                url = attachment.get("download_url")
                if url:
                    if "placeholder.png" not in url:
                        _download(url, output_dir, api)
                        current_run.log_info(f"Downloaded `{url}`")

    return True


def upload_attachments(input_dir: Path, dst_bucket: str, wait: bool = False):
    """Upload attachments to GCS bucket."""
    # authenticate to gcs bucket
    with open("/tmp/gcs.json", "w") as f:
        f.write(workspace.gcs_connection("hexa-public-praps").service_account_key)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcs.json"
    client = storage.Client()
    bucket = client.Bucket(dst_bucket)

    dst_files = [b.name for b in bucket.list_blobs()]

    EXTENSIONS = ["*.jpg", "*.jpeg", ".JPG", ".JPEG", "*.png", "*.PNG"]
    for pattern in EXTENSIONS:
        for fp in input_dir.glob(pattern):
            if fp.name not in dst_files:
                blob = bucket.blob(fp.name)
                blob.upload_from_filename(fp.absolute().as_posix())
                current_run.log_info(f"Uploaded `{fp.name}` to GCS bucket `{dst_bucket}`")
