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

# Surveys that never merged into the consolidated form and still stand on their own --
# downloaded, transformed, and pushed exactly as before the consolidation work.
LEGACY_SURVEYS = [
    # ("aNbJpiDsPCyC2uUmCxW5ph", "indicateurs_regionaux"), # replaced by CDR
    # ("aCHSjHDuphcc2KjCgqAyxP", "indicateurs_pays"), # replaced by CDR
    ("aG7DZmn6TK6SR6eyD2f3bj", "fourrage_cultive"),
    ("aPMh3Q2LRKZQ3uxT4cBK2s", "sous_projets_innovants"),
    ("aCtqjcw7etkEQXoBf6VMX7", "gestion_durable_des_paysages"),
    ("aAv4xBADqLzQw5h9fNDRts", "activites_generatrices_de_revenus"),
    # ("aEcSkGSLEadq6i5TSTNCjH", "cultures_vivrieres"), # out of scope
]

CONSOLIDATED_SURVEY = (
    "aRZ43wRerX8pCdo4XrKw4n",
    "fiche_simplifiee_infrastructures",
)  # contains 4 legacy surveys (marches_a_betail, parcs_de_vaccination, points_d_eau, unites_veterinaires) + 1 Mali-specific survey (infrastructures_hors_cdr)

SURVEYS = LEGACY_SURVEYS + [CONSOLIDATED_SURVEY]


@pipeline(name="extract-surveys")
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


def _write_survey_outputs(
    name: str, df: pl.DataFrame, df_no_duplicates: pl.DataFrame, output_dir: str
) -> None:
    """Write the standard surveys/geo/snapshots artefacts for one output table.

    Shared by the consolidated-survey splits and the standalone legacy surveys --
    the write logic is identical, only how `df`/`df_no_duplicates` were produced differs.
    """
    GEONODE_EXTRA_COLUMNS = ["_tags", "_notes"]
    TABLES_WITHOUT_GEONODE_EXTRA_COLUMNS = {"parcs_de_vaccination"}
    if name not in TABLES_WITHOUT_GEONODE_EXTRA_COLUMNS:
        missing = [c for c in GEONODE_EXTRA_COLUMNS if c not in df.columns]
        if missing:
            df = df.with_columns(
                [pl.lit(None).cast(pl.String).alias(c) for c in missing]
            )

        missing = [
            c for c in GEONODE_EXTRA_COLUMNS if c not in df_no_duplicates.columns
        ]
        if missing:
            df_no_duplicates = df_no_duplicates.with_columns(
                [pl.lit(None).cast(pl.String).alias(c) for c in missing]
            )

    df.write_parquet(Path(output_dir, "surveys", f"{name}_with_duplicates.parquet"))
    df_no_duplicates.write_parquet(Path(output_dir, "surveys", f"{name}.parquet"))
    df_no_duplicates.write_excel(Path(output_dir, "surveys", f"{name}.xlsx"))

    if not len(df_no_duplicates):
        current_run.log_warning(
            f"{name}: no submissions in this run, skipping geo/snapshots output"
        )
        return

    geo = to_geodataframe(
        df_no_duplicates.with_columns(
            pl.col("_geolocation").str.json_decode(dtype=pl.List(pl.Float64))
        )
    )
    geo = geo[geo.geometry.notna()]

    if not len(geo):
        current_run.log_warning(
            f"{name}: no submissions with a valid geometry, skipping gpkg output"
        )
    else:
        dst_file = Path(output_dir, "geo", f"{name}.gpkg")
        if dst_file.exists():
            dst_file.unlink()
        geo.to_file(dst_file)

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

    current_run.log_info(f"Processed {name} ({len(df_no_duplicates)} entries)")


@extract_surveys.task
def transform(src_dir: str, output_dir: str, wait: bool) -> bool:
    # standalone legacy surveys: never merged into the consolidated form, so they skip
    # split_consolidated/conform_to_legacy_schema/drop_blocked_submissions entirely --
    # those are consolidated-survey-specific concerns.
    for _, name in LEGACY_SURVEYS:
        src_file = Path(src_dir, f"{name}.parquet")
        if not src_file.exists():
            current_run.log_warning(f"Cannot find raw data for survey {name}")
            continue

        survey = pl.read_parquet(src_file)
        df, df_no_duplicates = surveys.transform_survey(survey, name)
        _write_survey_outputs(name, df, df_no_duplicates, output_dir)

    _, consolidated_name = CONSOLIDATED_SURVEY
    src_file = Path(src_dir, f"{consolidated_name}.parquet")
    if not src_file.exists():
        current_run.log_warning(f"Cannot find raw data for survey {consolidated_name}")
        return True

    survey = pl.read_parquet(src_file)

    df, df_no_duplicates = surveys.transform_survey(survey, consolidated_name)
    df = surveys.drop_blocked_submissions(df)
    df_no_duplicates = surveys.drop_blocked_submissions(df_no_duplicates)

    _write_survey_outputs(consolidated_name, df, df_no_duplicates, output_dir)

    splits = surveys.split_consolidated(df)
    splits_no_duplicates = surveys.split_consolidated(df_no_duplicates)

    for name in surveys.OUTPUT_TABLES:
        split_df = surveys.conform_to_legacy_schema(splits[name], name)
        split_no_dup = surveys.conform_to_legacy_schema(
            splits_no_duplicates[name], name
        )
        _write_survey_outputs(name, split_df, split_no_dup, output_dir)

    return True


@extract_surveys.task
def push(src_dir: str, wait: bool) -> bool:
    """Push survey data to the data warehouse."""
    engine = create_engine(workspace.database_url)

    legacy_survey_names = [name for _, name in LEGACY_SURVEYS]
    _, consolidated_name = CONSOLIDATED_SURVEY

    for name in legacy_survey_names + surveys.OUTPUT_TABLES + [consolidated_name]:
        gpkg_file = Path(src_dir, "geo", f"{name}.gpkg")
        if not gpkg_file.exists():
            current_run.log_warning(f"{name}: no gpkg file, skipping database push")
            continue

        gpkg = gpd.read_file(gpkg_file)
        gpkg = gpkg[gpkg.geometry.notna()]
        gpkg.to_postgis(name, engine, "public", if_exists="replace")
        current_run.add_database_output(name)
        current_run.log_info(f"Writing database table {name}")

        snapshots_file = Path(src_dir, "snapshots", f"{name}_snapshots.parquet")
        if snapshots_file.exists():
            df = pl.read_parquet(snapshots_file)
            df.write_database(
                f"{name}_snapshots", workspace.database_url, if_table_exists="replace"
            )
            current_run.add_database_output(f"{name}_snapshots")
            current_run.log_info(f"Writing database table {name}_snapshots")

    # mirror new data into old tables (they are still used by Geonode)
    mapping = {
        "PRAPS2_Marches_a_Betail": "marches_a_betail",
        "PRAPS2_Points_d_Eau": "points_d_eau",
        "PRAPS2_Unites_Veterinaires": "unites_veterinaires",
        "PRAPS2_Parcs_de_Vaccination": "parcs_de_vaccination",
        "PRAPS2_Infrastructures_Hors_CDR": "infrastructures_hors_cdr",
        "PRAPS2_Fourrages_Cultives": "fourrage_cultive",
        "PRAPS2_Gestion_Durable_des_Paysages": "gestion_durable_des_paysages",
        "PRAPS2_Activites_Generatrices_de_Revenus": "activites_generatrices_de_revenus",
        "PRAPS2_Sous_Projets_Innovants": "sous_projets_innovants",
    }

    for old_table, new_table in mapping.items():
        try:
            new = gpd.read_postgis(
                f'select * from "{new_table}"', con=engine, geom_col="geometry"
            )
            new.to_postgis(old_table, con=engine, if_exists="replace")
        except Exception as e:
            current_run.log_warning(
                f"Could not mirror {new_table} into {old_table}: {e}"
            )

    return True


@extract_surveys.task
def update_datasets(src_dir: str, wait: bool) -> bool:
    def get_md5(url: str):
        """Get MD5 hash of a dataset file."""
        r = requests.head(url)
        r.raise_for_status()
        return r.headers["ETag"].replace('"', "")

    DATASETS = [
        ("parcs_de_vaccination", "Parcs de vaccination", "parcs-de-vaccination-a6fbd3"),
        ("unites_veterinaires", "Unités vétérinaires", "unites-veterinaires-05ee52"),
        ("marches_a_betail", "Marchés à Bétail", "marches-a-betail-286942"),
        ("points_d_eau", "Points d'Eau", "points-d-eau-0935a6"),
        (
            "infrastructures_hors_cdr",
            "Infrastructures Hors CDR",
            "infrastructures-hors-cdr",
        ),
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
    ]

    # split_consolidated outputs share one consolidated fields file (there is no
    # per-survey one for them anymore); standalone legacy surveys keep their own.
    consolidated_fields_file = Path(
        src_dir, "metadata", "fiche_simplifiee_infrastructures_fields.xlsx"
    )
    split_from_consolidated = set(surveys.OUTPUT_TABLES)

    for survey_name, dataset_name, dataset_uid in DATASETS:
        if dataset_uid.startswith("TODO"):
            current_run.log_info(f"{survey_name}: dataset UID not set yet, skipping")
            continue

        src_files = [
            Path(src_dir, "surveys", f"{survey_name}.parquet"),
            Path(src_dir, "surveys", f"{survey_name}.xlsx"),
            Path(src_dir, "geo", f"{survey_name}.gpkg"),
        ]
        src_files = [p for p in src_files if p.exists()]

        if survey_name in split_from_consolidated:
            fields_file = consolidated_fields_file
        else:
            fields_file = Path(src_dir, "metadata", f"{survey_name}_fields.xlsx")
        if fields_file.exists():
            src_files.append(fields_file)

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
