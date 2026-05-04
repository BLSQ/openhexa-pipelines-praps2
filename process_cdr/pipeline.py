import polars as pl
from openhexa.sdk import current_run, parameter, pipeline, workspace

import re
from pathlib import Path
import utils
import config


@pipeline("process-cdr", name="process-cdr")
@parameter(
    "cdr_raw_dir",
    name="Dossier des CDR",
    help="Dossier où sont sauvegardés les CDR originaux",
    type=str,
    default="data/cdr/raw",
)
@parameter(
    "cdr_processed_dir",
    name="Dossier de sortie",
    help="Dossier de sortie des CDR transformés",
    type=str,
    default="data/cdr/processed",
)
@parameter(
    "cdr_indicators_ref_file",
    name="Fichier de nomenclature des indicateurs du CDR",
    help="Fichier de nomenclature des indicateurs du CDR",
    type=str,
    default="data/cdr/indicators_metadata_20260504.csv",
)
def process_cdr(
    cdr_raw_dir: str,
    cdr_processed_dir: str,
    cdr_indicators_ref_file: str,
):
    """
    Pipeline to process CDR Excel file and transform it into a long format Polars dataframe.
    """
    cdr_raw = import_file(cdr_raw_dir)
    cdr_indicators_map = import_indicators_map(cdr_indicators_ref_file)

    cdr_df = process(cdr_raw)
    cdr_df = assign_indicator_codes(cdr_df, cdr_indicators_map)
    cdr_df = final_cleaning(cdr_df)
    save_output(cdr_df, cdr_processed_dir, "cdr_transformed_2025")


def import_file(cdr_dir: str) -> pl.DataFrame:
    """Import most recent CDR file from the specified directory"""
    cdr_path = Path(workspace.files_path, cdr_dir)
    if not cdr_path.exists() or not cdr_path.is_dir():
        raise FileNotFoundError(
            f"Directory {cdr_dir} does not exist or is not a directory."
        )

    excel_files = list(cdr_path.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in directory {cdr_dir}.")

    # Get the most recent file based on modification time
    most_recent_file = max(excel_files, key=lambda f: f.stat().st_mtime)
    current_run.log_info(f"Most recent CDR file found: {most_recent_file}")

    df_raw = pl.read_excel(most_recent_file, has_header=False)

    return df_raw


def import_indicators_map(cdr_indicators_ref_file: str) -> pl.DataFrame:
    """Import the indicators reference file which contains mapping of indicator names to codes and other metadata."""
    ref_path = Path(workspace.files_path, cdr_indicators_ref_file)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Indicators reference file {cdr_indicators_ref_file} not found."
        )

    indicators_map = pl.read_csv(ref_path)
    current_run.log_info(
        f"Indicators reference file loaded with {indicators_map.height} entries."
    )

    return indicators_map


def process(cdr_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Process the raw CDR dataframe to extract year, forward fill merged cells, and transform to
    long format.
    """
    current_run.log_info("Processing CDR raw data...")

    # 1. Identify Year from Row 4 (index 3) or surrounding rows
    year = None
    for r_idx in [2, 3, 4]:
        if r_idx >= cdr_raw.height:
            continue
        row = cdr_raw.row(r_idx)
        for cell in row:
            if cell and "CIBLE FIN" in str(cell):
                match = re.search(r"CIBLE FIN (\d{4})", str(cell))
                if match:
                    year = int(match.group(1))
                    break
        if year:
            break

    if not year:
        current_run.log_error("Could not identify year from CDR file.")
        raise ValueError("Year not found in CDR file.")

    # 2. Map and Forward Fill the whole dataframe first
    # This ensures that merged indicator names are captured before we slice/filter.
    df_full = cdr_raw.select(
        [
            pl.col("column_1").alias("Code"),
            pl.col("column_2").alias("Indicateur_Name"),
            pl.col("column_4").alias("Unité"),
            pl.col("column_5").alias("Pays"),
            pl.col("column_7").alias("Valeur_cibles"),
            pl.col("column_8").alias("Valeur_résultats"),
        ]
    ).with_columns(
        [
            pl.col("Code").forward_fill(),
            pl.col("Indicateur_Name").forward_fill(),
            pl.col("Unité").forward_fill(),
        ]
    )

    # 3. Slice from row 5 (index 4) onward where data starts
    df_data = df_full.slice(4)

    # 4. Add Year and Clean
    df_transformed = df_data.with_columns(pl.lit(year).alias("Année"))

    # Filter out rows where Valeur résultats is missing or contains the term "Résultats"
    df_transformed = df_transformed.filter(
        pl.col("Valeur_résultats").is_not_null()
        & ~pl.col("Valeur_résultats").str.contains("Résultats")
    )

    # Replace "Valeur_xxx" values of "oui" and "non" with 1 and 0 respectively
    for col in ["Valeur_cibles", "Valeur_résultats"]:
        df_transformed = df_transformed.with_columns(
            pl.when(pl.col(col).str.to_lowercase() == "oui")
            .then(1)
            .when(pl.col(col).str.to_lowercase() == "non")
            .then(0)
            .otherwise(pl.col(col))
            .alias(col)
        )

    # Convert Valeur to numeric
    df_transformed = df_transformed.with_columns(
        pl.col("Valeur_cibles").cast(pl.Float64, strict=False),
        pl.col("Valeur_résultats").cast(pl.Float64, strict=False),
    )

    current_run.log_info(f"Transformed {df_transformed.height} rows.")

    return df_transformed


def assign_indicator_codes(
    df_transformed: pl.DataFrame, cdr_indicators_map: pl.DataFrame
) -> pl.DataFrame:
    """ "
    Assign indicator codes to the transformed dataframe by matching indicator names with the
    reference map, and applying manual mapping for unmatched indicators.
    """
    # first normalize the indicator names in both dataframes to improve matching
    df_transformed = df_transformed.with_columns(
        utils.normalize_indicator_column("Indicateur_Name").alias(
            "Indicateur_Name_clean"
        )
    )
    cdr_indicators_map = cdr_indicators_map.with_columns(
        utils.normalize_indicator_column("designation").alias("Indicateur_Name_clean")
    )

    # drop sub-indicators (those starting with the term "Dont ") (these will be calculated later on)
    cdr_indicators_map = cdr_indicators_map.filter(
        ~pl.col("Indicateur_Name_clean").str.starts_with("dont ")
    )

    # join with reference map
    df_transformed = df_transformed.join(
        cdr_indicators_map.select(
            [
                pl.col("Indicateur_Name_clean"),
                pl.col("code").alias("Code_clean"),
                pl.col("designation").alias("indicator_name"),
                pl.col("unite").alias(
                    "unit"
                ),  # to keep for future use, even if not used in current processing
            ]
        ),
        on=["Indicateur_Name_clean"],
        how="left",
    )

    # perform manual matching on remaining unmatched indicators using the mapping defined in config.py
    for indicator_name, code in config.missing_indicator_code_mapping.items():
        df_transformed = df_transformed.with_columns(
            pl.when(pl.col("Indicateur_Name_clean") == indicator_name)
            .then(pl.lit(code))
            .otherwise(pl.col("Code_clean"))
            .alias("Code_clean")
        )

    # now if an indicator has its clean name starting with "dont " it is a sub-indicator.
    # For those, use the dataset as sorted, and assign the the last code seen + the suffix
    # "1", "2", etc.
    df_transformed = df_transformed.with_columns(
        pl.col("Indicateur_Name_clean")
        .str.starts_with("dont ")
        .alias("is_sub_indicator")
    )
    df_transformed = df_transformed.with_columns(
        pl.when(pl.col("is_sub_indicator").not_())
        .then(pl.col("Code_clean"))
        .otherwise(None)
        .fill_null(strategy="forward")
        .alias("Parent_Code")
    )
    is_new_name = (
        (
            (pl.col("is_sub_indicator"))
            & (
                pl.col("Indicateur_Name_clean")
                != pl.col("Indicateur_Name_clean").shift()
            )
        )
        .fill_null(False)
        .cast(pl.Int32)
    )
    df_transformed = df_transformed.with_columns(
        is_new_name.cum_sum().over("Parent_Code").alias("sub_suffix")
    )
    df_transformed = df_transformed.with_columns(
        pl.when(pl.col("is_sub_indicator").not_())
        .then(pl.col("Code_clean"))
        .otherwise(pl.col("Parent_Code") + pl.col("sub_suffix").cast(pl.Utf8))
        .alias("Code_final")
    )

    # drop intermediate columns and rename final code column
    df_transformed = df_transformed.drop(
        [
            "Indicateur_Name",
            "Code",
            "Indicateur_Name_clean",
            "Code_clean",
            "is_sub_indicator",
            "Parent_Code",
            "sub_suffix",
        ]
    ).rename({"Code_final": "indicator_code"})

    return df_transformed


def final_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    """
    Final cleaning steps:
        - fill missing country values
        - drop total rows
        - clean country names based on config file
        - convert to standard values based on unit col
        - add suffix -01-01 to year to fit kobo data format
        - add level column
        - add project column
        - add year column
        - final column names clean up
    """
    # fill in missing country values
    df_cleaned = df.with_columns(
        pl.when(pl.col("Pays").is_not_null())
        .then(pl.col("Pays"))
        .when((pl.col("indicator_code") == "IRI-6") & (pl.col("Valeur_cibles") == 185))
        .then(pl.lit("MR"))
        .when((pl.col("indicator_code") == "IRI-7") & (pl.col("Valeur_cibles") == 5576))
        .then(pl.lit("MR"))
        .when(pl.col("indicator_code").is_in(config.regional_indicators))
        .then(pl.lit("REGIONAL"))
        .otherwise(pl.lit("NE"))
        .alias("Pays")
    )

    # drop total rows
    df_cleaned = df_cleaned.filter(~pl.col("Pays").str.contains("Total"))

    # clean country names based on config mapping
    df_cleaned = df_cleaned.with_columns(
        pl.col("Pays")
        .map_elements(
            lambda x: config.country_name_mapping.get(x, x), return_dtype=pl.String
        )
        .alias("Pays")
    )

    # convert to standard values based on unit column (e.g. if unit is en millions, multiply value by 1 million)
    cols_to_fix = ["Valeur_résultats", "Valeur_cibles"]
    df_cleaned = df_cleaned.with_columns(
        pl.when(pl.col("Unité").str.contains("(?i)million"))
        .then(pl.col(cols_to_fix) * 1_000_000)
        .when(pl.col("Unité").str.contains("(?i)millier"))
        .then(pl.col(cols_to_fix) * 1_000)
        .when(pl.col("Unité").str.contains("(?i)pourcentage"))
        .then(pl.col(cols_to_fix) / 100)
        .otherwise(pl.col(cols_to_fix))
    )

    # add suffix -01-01 to year and convert to date format to fit kobo data format
    df_cleaned = df_cleaned.with_columns(
        pl.col("Année")
        .cast(pl.String)
        .map_elements(lambda x: f"{x}-01-01", return_dtype=pl.String)
        .alias("Année")
    )

    # add level column (level 2 (Pays) except when country contains 'Régional', in which case level is 1 (Régional))
    df_cleaned = df_cleaned.with_columns(
        level=pl.when(pl.col("Pays").str.contains("Régional")).then(1).otherwise(2)
    )

    # add project col
    df_cleaned = df_cleaned.with_columns(pl.lit("PRAPS2").alias("project"))

    # add year column
    df_cleaned = df_cleaned.with_columns(
        pl.col("Année").str.slice(0, 4).cast(pl.Int32).alias("year")
    )

    # clean column names
    df_cleaned = df_cleaned.select(
        [
            pl.col("indicator_code"),
            pl.col("indicator_name"),
            pl.col("unit"),
            pl.col("year"),
            pl.col("Année").alias("date"),
            pl.col("project"),
            pl.col("level"),
            pl.col("Pays").alias("country"),
            pl.col("Valeur_résultats").alias("value"),
        ]
    )

    return df_cleaned


def save_output(df: pl.DataFrame, dir_name: str, file_name: str):
    """
    Save the processed dataframe to a parquet file in the specified directory and log the output file.
    """
    output_dir = Path(workspace.files_path, dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_name}.parquet"

    df.write_parquet(output_file)
    current_run.add_file_output(output_file.as_posix())


if __name__ == "__main__":
    process_cdr()
