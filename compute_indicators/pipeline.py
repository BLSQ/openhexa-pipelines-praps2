import datetime
import hashlib
from pathlib import Path

import indicators
import polars as pl
import requests
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.sdk.utils import Environment, get_environment


@pipeline("compute-indicators", name="compute-indicators")
@parameter(
    "survey_dir",
    name="Dossier d'entrée (fiches)",
    help="Répertoire où les fiches sont stockées",
    type=str,
    default="data/kobo/raw",
)
@parameter(
    "cdr_dir",
    name="Dossier de sortie (indicateurs CDR)",
    help="Répertoire où enregistrer les données",
    type=str,
    default="data/cdr",
)
def compute_indicators(survey_dir: str, cdr_dir: str):
    df = compute(
        survey_dir=Path(workspace.files_path, survey_dir),
        cdr_dir=Path(workspace.files_path, cdr_dir),
    )
    push(df)

    if get_environment() == Environment.CLOUD_PIPELINE:
        update_dataset(Path(workspace.files_path, cdr_dir), wait=df)


@compute_indicators.task
def compute(survey_dir: Path, cdr_dir: Path):
    df = indicators.combine_indicators(
        indicateurs_regionaux=pl.read_parquet(Path(survey_dir, "indicateurs_regionaux.parquet")),
        indicateurs_pays=pl.read_parquet(Path(survey_dir, "indicateurs_pays.parquet")),
        gestion_durable=pl.read_parquet(Path(survey_dir, "gestion_durable_des_paysages.parquet")),
        unites_veterinaires=pl.read_parquet(Path(survey_dir, "unites_veterinaires.parquet")),
        parcs_de_vaccination=pl.read_parquet(Path(survey_dir, "parcs_de_vaccination.parquet")),
        points_d_eau=pl.read_parquet(Path(survey_dir, "points_d_eau.parquet")),
        marches_a_betail=pl.read_parquet(Path(survey_dir, "marches_a_betail.parquet")),
        sous_projets=pl.read_parquet(Path(survey_dir, "sous_projets_innovants.parquet")),
        activites=pl.read_parquet(Path(survey_dir, "activites_generatrices_de_revenus.parquet")),
        praps1=pl.read_csv(Path(cdr_dir, "cdr_praps1_initial_values.csv")),
    )
    current_run.log_info(
        f"Computed {len(df['indicator_code'].unique()) - 1} indicators ({len(df)} values)"
    )

    df = indicators.join_metadata(
        df, indicators_metadata=pl.read_csv(Path(cdr_dir, "indicators_metadata.csv"))
    )
    current_run.log_info(f"Joined metadata ({len(df)} values)")

    df = indicators.spatial_aggregation(df)
    current_run.log_info(f"Applied spatial aggregation ({len(df)} values)")

    df = indicators.fill_missing_values(df)
    current_run.log_info(f"Filled missing values ({len(df)} values)")

    df = indicators.cumulate_indicators(df)
    current_run.log_info(f"Cumulated indicators ({len(df)} values)")

    df = indicators.retro_compatibility(df)
    current_run.log_info(f"Modified columns for retro-compatibility ({len(df)} values)")

    fp_parquet = Path(cdr_dir, "indicateurs.parquet")
    fp_xlsx = Path(cdr_dir, "indicateurs.xlsx")
    df.write_parquet(fp_parquet)
    current_run.log_info(f"Saved {fp_parquet.name}")
    df.write_excel(fp_xlsx)
    current_run.log_info(f"Saved {fp_xlsx.name}")

    if get_environment() == Environment.CLOUD_PIPELINE:
        current_run.add_file_output(fp_parquet.as_posix())
        current_run.add_file_output(fp_xlsx.as_posix())

    return df


@compute_indicators.task
def push(df: pl.DataFrame) -> bool:
    df.write_database("indicators", connection=workspace.database_url, if_table_exists="replace")
    current_run.log_info(f"Writing to database table `indicators` ({len(df)} rows)")

    df.write_database(
        "PRAPS2_Indicators_Aggregated", connection=workspace.database_url, if_table_exists="replace"
    )
    current_run.log_info(
        f"Writing to database table `PRAPS2_Indicators_Aggregated` ({len(df)} rows)"
    )

    if get_environment() == Environment.CLOUD_PIPELINE:
        current_run.add_database_output("indicators")
        current_run.add_database_output("PRAPS2_Indicators_Aggregated")

    return True


@compute_indicators.task
def update_dataset(cdr_dir: str, wait: bool) -> bool:
    def get_md5(url: str):
        """Get MD5 hash of a dataset file."""
        r = requests.head(url)
        r.raise_for_status()
        return r.headers["ETag"].replace('"', "")

    src_files = [Path(cdr_dir, "indicateurs.parquet"), Path(cdr_dir, "indicateurs.xlsx")]

    dataset = workspace.get_dataset("indicateurs-cdr-353681")
    latest = dataset.latest_version

    if latest:
        src_hashes = [
            hashlib.md5(open(src_file, "rb").read()).hexdigest() for src_file in src_files
        ]
        dst_hashes = [get_md5(f.download_url) for f in latest.files]

        if set(src_hashes) == set(dst_hashes):
            current_run.log_info("No changes detected. Skipping dataset update")
            return True

    new_version = dataset.create_version(name=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    for src_file in src_files:
        new_version.add_file(src_file.as_posix(), src_file.name)

    current_run.log_info("Updated dataset")

    return True


if __name__ == "__main__":
    compute_indicators()
