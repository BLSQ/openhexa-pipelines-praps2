import json
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
    "fiche_simplifiee_infrastructures",
    "fourrage_cultive",
    "gestion_durable_des_paysages",
    "activites_generatrices_de_revenus",
    "sous_projets_innovants",
]

# The ~50 Mali/HCDR rows in fiche_simplifiee_infrastructures were migrated from this
# form, which lives under a *different* Kobo account than everything else here -- the
# main `kobo_api` connection's token has no access to it.
MALI_FORM_UID = "a4LDskBmypjPahmmZqrpmQ"


@pipeline(name="sync-attachments")
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
    "bucket",
    name="GCS bucket",
    help="Nom du bucket GCS où stocker les images",
    type=str,
    default="hexa-public-praps",
)
@parameter(
    "delete_undersized_files",
    name="Supprimer les fichiers corrompus",
    help=(
        "Avant de (re)télécharger quoi que ce soit, supprime -- localement et dans le bucket GCS "
        "-- tout fichier déjà présent dont la taille correspond exactement à la taille ci-dessous "
        "(23 octets = signature connue des images de remplacement corrompues téléchargées par une "
        "ancienne version de ce pipeline, via le fallback `<colonne>_URL`, désormais supprimé). "
        "À activer une fois pour nettoyer, puis remettre à faux pour les exécutions suivantes."
    ),
    type=bool,
    default=False,
)
@parameter(
    "max_corrupted_size_bytes",
    name="Taille max. d'un fichier corrompu (octets)",
    help=(
        "Tout fichier dont la taille est strictement inférieure à cette valeur est supprimé. "
        "Les images corrompues connues font 23 octets ; ce seuil (25 par défaut) garde une petite "
        "marge tout en restant bien en dessous de la taille d'une vraie photo."
    ),
    type=int,
    default=25,
)
def sync_attachments(
    input_dir: str,
    output_dir: str,
    bucket: str,
    delete_undersized_files: bool = False,
    max_corrupted_size_bytes: int = 25,
):
    input_dir = Path(workspace.files_path, input_dir)
    output_dir = Path(workspace.files_path, output_dir)
    if delete_undersized_files:
        delete_undersized_attachments(output_dir, bucket, max_corrupted_size_bytes)
    task1 = download_attachments(input_dir, output_dir)
    upload_attachments(output_dir, bucket, wait=task1)


def _download(url: str, dst_dir: Path, api: Api):
    """Download one attachment, named after the response's Content-Disposition header."""
    with api.session.get(url, stream=True) as r:
        content_type = r.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            # a non-image response (HTML error/permission page, JSON, an unfollowed
            # redirect body, ...) saved under a .jpg name is indistinguishable from a
            # real photo until something tries to render it -- refuse it here instead,
            # and log enough to tell exactly what came back.
            current_run.log_warning(
                f"Attachment `{url}` did not return image content (status "
                f"{r.status_code}, content-type '{content_type}') -- skipping"
            )
            return
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


def _gcs_client() -> storage.Client:
    """Authenticate to GCS and return a client, reused by cleanup and upload."""
    with open("/tmp/gcs.json", "w") as f:
        f.write(workspace.gcs_connection("hexa-public-praps").service_account_key)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcs.json"
    return storage.Client()


def delete_undersized_attachments(
    output_dir: Path, bucket_name: str, max_bad_size_bytes: int
):
    """Delete already-synced files (locally and in GCS) smaller than `max_bad_size_bytes`.

    Confirmed empirically: every corrupted placeholder image downloaded by the now-removed
    `<column>_URL` fallback is exactly 23 bytes -- no real infrastructure photo is ever
    anywhere close to that size, so a small threshold (default 25) is safe from catching a
    legitimately small real photo while allowing a little margin over the exact known
    value. Deleting them here -- before download_attachments runs -- clears the way for
    the real photo to be fetched and uploaded in this same run, since both `_download`'s
    `if not fpath.exists()` guard and the `dst_files`/GCS listing checks would otherwise
    treat the corrupted file as already synced.
    """
    deleted = 0

    if output_dir.exists():
        for fp in output_dir.iterdir():
            if fp.is_file() and fp.stat().st_size < max_bad_size_bytes:
                size = fp.stat().st_size
                fp.unlink()
                deleted += 1
                current_run.log_warning(
                    f"Deleted local file `{fp.name}` ({size} bytes)"
                )

    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    for blob in bucket.list_blobs():
        if blob.size is not None and blob.size < max_bad_size_bytes:
            current_run.log_warning(
                f"Deleted GCS blob `{blob.name}` ({blob.size} bytes)"
            )
            blob.delete()
            deleted += 1

    current_run.log_info(
        f"Cleanup done: {deleted} file(s)/blob(s) under {max_bad_size_bytes} bytes deleted"
    )
    return True


def _download_mali_attachments(output_dir: Path, dst_files: set):
    """Download real attachments natively from the Mali/HCDR form's own account.

    Requires a second custom connection, `kobo-api-mali`, scoped to the Kobo account
    that hosts MALI_FORM_UID -- the main `kobo_api` connection cannot see it. Missing
    connection is not fatal: logs a warning and lets the rest of the pipeline run, since
    that only leaves the ~50 Mali rows unsynced rather than breaking everything else.
    """
    try:
        con = workspace.custom_connection("kobo-api-mali")
    except Exception:
        current_run.log_warning(
            "No `kobo-api-mali` connection configured -- skipping attachment download "
            "for the Mali/HCDR form. Photos for those fiche_simplifiee_infrastructures "
            "rows will not be synced until this connection is added."
        )
        return

    api = Api(url=con.url)
    api.authenticate(token=con.token)
    survey = api.get_survey(MALI_FORM_UID)
    submissions = survey.get_data()
    current_run.log_info(
        f"Connected to Mali KoboToolbox account at url: {con.url} "
        f"({len(submissions)} submissions)"
    )

    for submission in submissions:
        for attachment in submission.get("_attachments") or []:
            url = attachment.get("download_url")
            if not url or "placeholder.png" in url:
                continue
            fname = (attachment.get("filename") or "").split("/")[-1]
            if not fname or fname in dst_files:
                continue
            _download(url, output_dir, api)
            current_run.log_info(f"Downloaded `{url}` (Mali form)")
            dst_files.add(fname)


def download_attachments(input_dir: Path, output_dir: Path):
    """Download attachments for a given survey."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    dst_files = {
        f.name for f in output_dir.iterdir()
    }  # set: O(1) membership checks below

    _download_mali_attachments(output_dir, dst_files)

    con = workspace.custom_connection("kobo_api")
    api = Api(url=con.url)
    api.authenticate(token=con.token)
    current_run.log_info("Connected to KoboToolbox API at url: {}".format(con.url))

    for survey in SURVEYS:
        # load survey data as dataframe
        # links to attachments are stored in the `_attachments` column
        df_fname = input_dir / f"{survey}.parquet"
        if not df_fname.exists():
            current_run.log_warning(f"File not found for survey `{survey}`")
        df = pl.read_parquet(df_fname)

        # multiple attachments can be stored in the same cell
        for attachments in df["_attachments"]:
            if not attachments:
                continue
            attachments = json.loads(attachments)
            for attachment in attachments:
                url = attachment.get("download_url")
                if url:
                    # urls with placeholder images do not need to be downloaded
                    if "placeholder.png" in url:
                        continue
                    # get filename and check if it has already been downloaded
                    fname = attachment.get("filename").split("/")[-1]
                    if fname in dst_files:
                        continue
                    # download attachment
                    _download(url, output_dir, api)
                    current_run.log_info(f"Downloaded `{url}`")
                    dst_files.add(fname)

    return True


def upload_attachments(input_dir: Path, dst_bucket: str, wait: bool = False):
    """Upload attachments to GCS bucket."""
    client = _gcs_client()
    bucket = client.bucket(dst_bucket)

    dst_files = {
        b.name for b in bucket.list_blobs()
    }  # set: O(1) membership checks below

    EXTENSIONS = ["*.jpg", "*.jpeg", ".JPG", ".JPEG", "*.png", "*.PNG"]
    for pattern in EXTENSIONS:
        for fp in input_dir.glob(pattern):
            if fp.name not in dst_files:
                blob = bucket.blob(fp.name)
                blob.upload_from_filename(fp.absolute().as_posix())
                current_run.log_info(
                    f"Uploaded `{fp.name}` to GCS bucket `{dst_bucket}`"
                )


if __name__ == "__main__":
    sync_attachments()
