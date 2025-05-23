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
    ("aNbJpiDsPCyC2uUmCxW5ph", "indicateurs_regionaux"),
    ("aCHSjHDuphcc2KjCgqAyxP", "indicateurs_pays"),
    ("aCgpBnsy39TLqiSsujMiYA", "marches_a_betail"),
    ("aLWDMaeZnBHoZvoUHgy9WY", "parcs_de_vaccination"),
    ("aEsmYrLyFiN9o9PK3TtbP3", "points_d_eau"),
    ("aNQNnPXKrf8CKJVddw3x3M", "unites_veterinaires"),
    ("aG7DZmn6TK6SR6eyD2f3bj", "fourrage_cultive"),
    ("aPMh3Q2LRKZQ3uxT4cBK2s", "sous_projets_innovants"),
    ("aCtqjcw7etkEQXoBf6VMX7", "gestion_durable_des_paysages"),
    ("aAv4xBADqLzQw5h9fNDRts", "activites_generatrices_de_revenus"),
    # ("aEcSkGSLEadq6i5TSTNCjH", "cultures_vivrieres"),
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
    "push_to_db",
    name="Mettre à jour la base de données",
    help="Mettre à jour la base de données avec les fiches extraites",
    type=bool,
    default=True,
)
@parameter(
    "overwrite",
    name="Ecraser les données existantes",
    help="Re-télécharger l'ensemble des fiches et remplacer les fichiers existants",
    type=bool,
    default=True,
)
def extract_surveys(output_dir: str, push_to_db: bool, overwrite: bool):
    con = workspace.custom_connection("kobo_api")
    output_dir = Path(workspace.files_path, output_dir)

    for subdir in ("raw", "surveys", "geo", "metadata", "snapshots"):
        os.makedirs(Path(output_dir, subdir), exist_ok=True)

    if overwrite:
        task1 = download(con, output_dir)
    else:
        task1 = True

    task2 = transform(
        src_dir=Path(output_dir, "raw"), output_dir=output_dir, wait=task1
    )

    if push_to_db:
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
            api,
            uid,
            name,
            dst_file=Path(output_dir, "metadata", f"{name}_fields.parquet"),
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

        dst_file = Path(output_dir, "geo", f"{name}.gpkg")
        if dst_file.exists():
            dst_file.unlink()

        geo.to_file(Path(output_dir, "geo", f"{name}.gpkg"))

        snapshots = surveys.concatenate_snapshots(
            df, column_unique_id="infrastructure_id"
        )
        snapshots.write_parquet(
            Path(output_dir, "snapshots", f"{name}_snapshots.parquet")
        )

        if get_environment() == Environment.CLOUD_PIPELINE:
            current_run.add_file_output(
                Path(output_dir, "surveys", f"{name}.parquet").as_posix()
            )

        current_run.log_info(f"Processed survey {name} ({len(df)} values)")

    return True


@extract_surveys.task
def push(src_dir: str, wait: bool) -> bool:
    """Push survey data to the data warehouse."""
    con = create_engine(workspace.database_url)

    for _, name in SURVEYS:
        if name in ("indicateurs_regionaux", "indicateurs_pays", "cultures_vivrieres"):
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

            df = pl.read_parquet(
                Path(src_dir, "snapshots", f"{name}_snapshots.parquet")
            )
            df.write_database(
                f"{name}_snapshots", workspace.database_url, if_table_exists="replace"
            )
            current_run.add_database_output(f"{name}_snapshots")
            current_run.log_info(f"Writing database table {name}_snapshots")

    # mirror new data into old tables (they are still used by Geonode)
    mapping = {
        "PRAPS2_Activites_Generatrices_de_Revenus": "activites_generatrices_de_revenus",
        "PRAPS2_Fourrages_Cultives": "fourrage_cultive",
        "PRAPS2_Gestion_Durable_des_Paysages": "gestion_durable_des_paysages",
        "PRAPS2_Marches_a_Betail": "marches_a_betail",
        "PRAPS2_Points_d_Eau": "points_d_eau",
        "PRAPS2_Sous_Projets_Innovants": "sous_projets_innovants",
        "PRAPS2_Unites_Veterinaires": "unites_veterinaires",
        "PRAPS2_Parcs_de_Vaccination": "parcs_de_vaccination",
    }

    engine = create_engine(workspace.database_url)
    for old_table, new_table in mapping.items():
        new = gpd.read_postgis(
            f'select * from "{new_table}"', con=engine, geom_col="geometry"
        )
        new.to_postgis(old_table, con=engine, if_exists="replace")

    return True


@extract_surveys.task
def update_datasets(src_dir: str, wait: bool) -> bool:
    def get_md5(url: str):
        """Get MD5 hash of a dataset file."""
        r = requests.head(url)
        r.raise_for_status()
        return r.headers["ETag"].replace('"', "")

    DATASETS = [
        (
            "indicateurs_regionaux",
            "Indicateurs Régionaux",
            "indicateurs-regionaux-d8d9d9",
        ),
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
        (
            "sous_projets_innovants",
            "Sous-Projets Innovants",
            "sous-projets-innovants-7c3877",
        ),
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
                hashlib.md5(open(src_file, "rb").read()).hexdigest()
                for src_file in src_files
            ]
            dst_hashes = [get_md5(f.download_url) for f in latest.files]

            if set(src_hashes) == set(dst_hashes):
                continue

        new_version = dataset.create_version(
            name=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        for src_file in src_files:
            new_version.add_file(src_file.as_posix(), src_file.name)

    return True


if __name__ == "__main__":
    extract_surveys()
