import datetime
import hashlib
import os
from pathlib import Path
from typing import List

import geopandas as gpd
import polars as pl
import requests
import surveys
from openhexa.sdk import CustomConnection, current_run, parameter, pipeline, workspace
from openhexa.sdk.utils import Environment, get_environment
from openhexa.toolbox.kobo import Api
from openhexa.toolbox.kobo.utils import to_geodataframe
from sqlalchemy import create_engine

SURVEYS = [
    ("aHjhdBWbXzGf6d9bYm88Hq", "indicateurs_regionaux"),
    ("aCHSjHDuphcc2KjCgqAyxP", "indicateurs_pays"),
    ("a67gG6NTYHrxeEJkRDzk3q", "marches_a_betail"),
    ("a8VM8vXBA5vC2RvZ5g3MK4", "parcs_de_vaccination"),
    ("aQiqocfeJbxAd4zRehcXEP", "points_d_eau"),
    ("aRSr2SdzvpPEmnn6cJZXhK", "unites_veterinaires"),
    ("a59pdEM78L8WRtTTfpHGom", "fourrage_cultive"),
    ("a6gNzGeR2ScNZtYapFhTxJ", "sous_projets_innovants"),
    ("aEBHXt3rCHYxifEZLBMPU4", "gestion_durable_des_paysages"),
    ("aKjQ9Ak8rsPhtgxMup8mm6", "activites_generatrices_de_revenus"),
]


@pipeline("extract-surveys", name="extract-surveys")
@parameter(
    "output_dir",
    name="Dossier de sortie",
    help="Répertoire où enregistrer les fiches",
    type=str,
    default="data/kobo",
)
@parameter(
    "overwrite",
    name="Ecraser les données existantes",
    help="Re-télécharger l'ensemble des fiches et remplacer les fichiers existants",
    type=bool,
    default=True,
)
def extract_surveys(output_dir: str, overwrite: bool):
    con = workspace.custom_connection("KOBO_API")
    output_dir = Path(workspace.files_path, output_dir)

    for subdir in ("raw", "surveys", "geo", "metadata", "snapshots"):
        os.makedirs(Path(output_dir, subdir), exist_ok=True)

    if overwrite:
        task1 = download(con, output_dir)
    else:
        task1 = True

    task2 = transform(src_dir=Path(output_dir, "raw"), output_dir=output_dir, wait=task1)
    push(output_dir, wait=task2)
    if get_environment() == Environment.CLOUD_PIPELINE:
        update_datasets(output_dir, wait=task2)


@extract_surveys.task
def download(con: CustomConnection, output_dir: Path) -> List[Path]:
    api = Api(con.url)
    api.authenticate(con.token)

    for uid, name in SURVEYS:
        surveys.download_survey_data(
            api, uid, name, dst_file=Path(output_dir, "raw", f"{name}.parquet")
        )
        surveys.download_survey_fields(
            api, uid, name, dst_file=Path(output_dir, "metadata", f"{name}_fields.parquet")
        )

    return True


@extract_surveys.task
def transform(src_dir: str, output_dir: str, wait: bool) -> bool:
    for _, name in SURVEYS:
        src_file = Path(src_dir, f"{name}.parquet")
        if not src_file.exists():
            current_run.log_warning(f"Cannot find raw data for survey {name}")
            continue

        survey = pl.read_parquet(src_file)
        df = surveys.transform_survey(survey, name)
        df.write_parquet(Path(output_dir, "surveys", f"{name}.parquet"))
        df.write_excel(Path(output_dir, "surveys", f"{name}.xlsx"))

        if name in ("indicateurs_regionaux", "indicateurs_pays"):
            continue

        geo = to_geodataframe(df.with_columns(pl.col("_geolocation").str.json_decode()))
        geo.to_file(Path(output_dir, "geo", f"{name}.gpkg"), driver="GPKG")

        snapshots = surveys.concatenate_snapshots(df, column_unique_id="infrastructure_id")
        snapshots.write_parquet(Path(output_dir, "snapshots", f"{name}_snapshots.parquet"))

        if get_environment() == Environment.CLOUD_PIPELINE:
            current_run.add_file_output(Path(output_dir, "surveys", f"{name}.parquet").as_posix())

        current_run.log_info(f"Processed survey {name} ({len(df)} values)")

    return True


@extract_surveys.task
def push(src_dir: str, wait: bool) -> bool:
    """Push survey data to the data warehouse."""
    con = create_engine(workspace.database_url)

    for _, name in SURVEYS:
        if name in ("indicateurs_regionaux", "indicateurs_pays"):
            df = pl.read_parquet(Path(src_dir, "surveys", f"{name}.parquet"))
            df.write_database(
                name,
                workspace.database_url,
                if_table_exists="replace",
            )
            current_run.add_database_output(name)
            current_run.log_info(f"Writing database table {name}")

        else:
            gpkg = gpd.read_file(Path(src_dir, "geo", f"{name}.gpkg"))
            gpkg.to_postgis(name, con, "public", if_exists="replace")
            current_run.add_database_output(name)
            current_run.log_info(f"Writing database table {name}")

            df = pl.read_parquet(Path(src_dir, "snapshots", f"{name}_snapshots.parquet"))
            df.write_database(
                f"{name}_snapshots", workspace.database_url, if_table_exists="replace"
            )
            current_run.add_database_output(f"{name}_snapshots")
            current_run.log_info(f"Writing database table {name}_snapshots")

    return True


@extract_surveys.task
def update_datasets(src_dir: str, wait: bool) -> bool:
    def get_md5(url: str):
        """Get MD5 hash of a dataset file."""
        r = requests.head(url)
        r.raise_for_status()
        return r.headers["ETag"].replace('"', "")

    DATASETS = [
        ("indicateurs_regionaux", "Indicateurs Régionaux", "indicateurs-regionaux-d8d9d9"),
        ("indicateurs_pays", "Indicateurs Nationaux", "indicateurs-nationaux-c99b47"),
        ("fourrage_cultive", "Fourrages Cultivés", "fourrages-cultives-b6f422"),
        (
            "gestion_durable_des_paysages",
            "Superficies Sous Pratique de Gestion Durable des Paysages",
            "superficies-sous-pratiq-460c3f",
        ),
        (
            "activites_generatrices_de_revenus",
            "Activités Génératrices de Revenus",
            "activites-generatrices-cadd93",
        ),
        ("sous_projets_innovants", "Sous-Projets Innovants", "sous-projets-innovants-7c3877"),
        ("parcs_de_vaccination", "Parcs de vaccination", "parcs-de-vaccination-a6fbd3"),
        ("unites_veterinaires", "Unités vétérinaires", "unites-veterinaires-05ee52"),
        ("marches_a_betail", "Marchés à Bétail", "marches-a-betail-286942"),
        ("points_d_eau", "Points d'Eau", "points-d-eau-0935a6"),
    ]

    for survey_name, dataset_name, dataset_uid in DATASETS:
        src_files = [
            Path(src_dir, "surveys", f"{survey_name}.parquet"),
            Path(src_dir, "surveys", f"{survey_name}.xlsx"),
            Path(src_dir, "geo", f"{survey_name}.gpkg"),
            Path(src_dir, "metadata", f"{survey_name}_fields.xlsx"),
        ]

        src_files = [p for p in src_files if p.exists()]

        dataset = workspace.get_dataset(dataset_uid)
        latest = dataset.latest_version

        if latest:
            src_hashes = [
                hashlib.md5(open(src_file, "rb").read()).hexdigest() for src_file in src_files
            ]
            dst_hashes = [get_md5(f.download_url) for f in latest.files]

            if set(src_hashes) == set(dst_hashes):
                continue

        new_version = dataset.create_version(name=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        for src_file in src_files:
            new_version.add_file(src_file.as_posix(), src_file.name)

    return True


if __name__ == "__main__":
    extract_surveys()
