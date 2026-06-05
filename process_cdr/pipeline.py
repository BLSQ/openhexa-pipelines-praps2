import polars as pl
from openhexa.sdk import current_run, parameter, pipeline, workspace

from pathlib import Path
import utils
import config
from openhexa.sdk.utils import Environment, get_environment


@pipeline(name="process-cdr")
@parameter(
    "cdr_dir",
    name="Dossier des CDR",
    help="Dossier où sont sauvegardés les CDR originaux",
    type=str,
    default="data",
)
@parameter(
    "indicators_metadata_file",
    name="Fichier de metadata des indicateurs du CDR",
    help="Fichier de metadata des indicateurs du CDR",
    type=str,
    default="data/cdr/indicators_metadata_v2.csv",
)
def process_cdr(
    cdr_dir: str,
    indicators_metadata_file: str,
):
    """
    Pipeline to process CDR Excel files and transform them into a long format Polars dataframe.
    """
    # import data
    indicators_metadata = import_indicators_metadata(indicators_metadata_file)
    cdr_2025_raw = import_file(
        f"{cdr_dir}/cdr/raw", "CDR_PRAPS_2_CONSOLIDE_PAYS_CILSS_31_12_25_VF.xlsx"
    )
    cdr_targets_old = import_file(f"{cdr_dir}/targets", "CDR_Targets.csv")

    # process 2025 CDR
    cdr_2025_df = process_2025_CDR(cdr_2025_raw)
    cdr_2025_df = assign_indicator_codes(cdr_2025_df, indicators_metadata)

    # process 2026-2027 CDR
    cdr_2026_2027_raw = import_file(
        f"{cdr_dir}/cdr/raw",
        "PRAPS-2 Projet de CDR révisé - décembre 2025 VF 191225.xlsx",
    )
    cdr_2026_2027_df = process_2026_2027_CDR(cdr_2026_2027_raw)
    cdr_2026_2027_df = assign_indicator_codes(cdr_2026_2027_df, indicators_metadata)

    # create cdr results df
    cdr_2025_results_df = clean_results_values(cdr_2025_df)

    # create combined target df
    cdr_2026_2027_targets_df = clean_target_values(cdr_2026_2027_df)
    combined_targets_df = combine_targets(cdr_targets_old, cdr_2026_2027_targets_df)

    # save outputs
    save_output(cdr_2025_results_df, f"{cdr_dir}/cdr/processed", "cdr_results_2025")
    save_output(combined_targets_df, f"{cdr_dir}/cdr/processed", "CDR_Targets_v2")
    push_to_db(combined_targets_df, "CDR_Targets_v2")


def import_file(cdr_dir: str, file_name: str) -> pl.DataFrame:
    """Import most recent CDR file from the specified directory"""
    cdr_path = Path(workspace.files_path, cdr_dir)
    if not cdr_path.exists() or not cdr_path.is_dir():
        raise FileNotFoundError(
            f"Directory {cdr_dir} does not exist or is not a directory."
        )
    if cdr_dir == "data/cdr/raw":
        df_raw = pl.read_excel(Path(cdr_path, file_name), has_header=False)
    elif cdr_dir == "data/targets":
        df_raw = pl.read_csv(Path(cdr_path, file_name), has_header=True)

    return df_raw


def import_indicators_metadata(indicators_metadata_file: str) -> pl.DataFrame:
    """Import the indicators metadata file which contains mapping of indicator names to codes and other metadata."""
    ref_path = Path(workspace.files_path, indicators_metadata_file)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Indicators metadata file {indicators_metadata_file} not found."
        )

    indicators_metadata = pl.read_csv(ref_path)
    current_run.log_info(
        f"Indicators metadata file loaded with {indicators_metadata.height} entries."
    )

    return indicators_metadata


def process_2025_CDR(cdr_2025_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Process the raw CDR dataframe to extract year, forward fill merged cells, and transform to
    long format.
    """
    current_run.log_info("Processing CDR raw data...")

    cdr_2025_raw = cdr_2025_raw.slice(4)
    cdr_2025_raw = cdr_2025_raw.with_row_index("original_order")
    df_full = cdr_2025_raw.select(
        [
            pl.col("column_1").alias("indicator_code_original"),
            pl.col("column_2").alias("indicator_name_original"),
            pl.col("column_4").alias("unit_original"),
            pl.col("column_5").alias("country"),
            pl.col("column_8").alias("result_value"),
            pl.col("original_order"),
        ]
    ).with_columns(
        [
            pl.col("indicator_code_original").forward_fill(),
            pl.col("indicator_name_original").forward_fill(),
            pl.col("unit_original").forward_fill(),
        ]
    )

    # recalculate original_order to allow correct assignmet of sub-indicators later on
    df_full = df_full.with_columns(
        (pl.col("original_order") // 100 * 100).alias("original_order_hundred")
    )
    df_full = df_full.with_columns(
        pl.col("original_order")
        .min()
        .over(
            [
                "indicator_name_original",
                "indicator_code_original",
                "original_order_hundred",
            ]
        )
        .alias("original_order")
    ).drop("original_order_hundred")

    # assign year 2025
    df_transformed = df_full.with_columns(pl.lit(2025).alias("year"))

    # Filter out rows where Valeur résultats is missing or contains the term "Résultats" (exception: rows with restructured funding")
    df_transformed = df_transformed.filter(
        pl.col("result_value").is_not_null()
        & ~pl.col("result_value").str.contains("Résultats").fill_null(False)
    )

    # Replace "Valeur_xxx" values of "oui" and "non" with 1 and 0 respectively
    for col in ["result_value"]:
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
        pl.col("result_value").cast(pl.Float64, strict=False),
    )

    # fill in missing country values
    df_transformed = df_transformed.with_columns(
        pl.when(pl.col("country").is_not_null())
        .then(pl.col("country"))
        # MR
        .when(
            (pl.col("indicator_code_original") == "6") & (pl.col("result_value") == 107)
        )
        .then(pl.lit("MR"))
        .when(
            (pl.col("indicator_code_original") == "7")
            & (pl.col("result_value") == 3_509)
        )
        .then(pl.lit("MR"))
        # NE
        .when(
            (pl.col("indicator_code_original") == "6") & (pl.col("result_value") == 89)
        )
        .then(pl.lit("NE"))
        .when(
            (pl.col("indicator_code_original") == "7")
            & (pl.col("result_value") == 7_699.4)
        )
        .then(pl.lit("NE"))
        .when(
            (pl.col("indicator_code_original") == "13") & (pl.col("country").is_null())
        )
        .then(pl.lit("NE"))
        # BF
        .when((pl.col("indicator_code_original").is_in(["FA 2"])))
        .then(pl.lit("BF"))
        # Non-missing
        .otherwise(pl.col("country"))
        .alias("country")
    )

    # convert from wide to long format
    df_transformed = df_transformed.unpivot(
        on=["result_value"],
        index=[
            "indicator_name_original",
            "unit_original",
            "country",
            "indicator_code_original",
            "original_order",
            "year",
        ],
        variable_name="value_type",
        value_name="value",
    )

    # replace value_type with "target" and "result"
    df_transformed = df_transformed.with_columns(
        pl.col("value_type")
        .map_elements(
            lambda x: "target" if x == "target_value" else "result",
            return_dtype=pl.String,
        )
        .alias("value_type")
    )

    # drop total rows
    df_transformed = df_transformed.filter(~pl.col("country").str.contains("Total"))

    current_run.log_info(f"Transformed {df_transformed.height} rows.")

    return df_transformed


def process_2026_2027_CDR(cdr_2026_2027_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Process the raw CDR dataframe to extract year, forward fill merged cells, and transform to
    long format.
    """
    current_run.log_info("Processing CDR raw data...")

    # drop first 3 rows
    cdr_2026_2027_raw = cdr_2026_2027_raw.slice(4)

    # create order column to keep track of original order for sub-indicators assignment later on
    cdr_2026_2027_raw = cdr_2026_2027_raw.with_row_index("original_order")

    # 2. Map and Forward Fill the whole dataframe first
    # This ensures that merged indicator names are captured before we slice/filter.
    df_full = cdr_2026_2027_raw.select(
        [
            pl.col("column_1").alias("indicator_code_original"),
            pl.col("column_2").alias("indicator_name_original"),
            pl.col("column_4").alias("unit_original"),
            pl.col("column_5").alias("country"),
            pl.col("column_6").alias("target_value_2021"),
            pl.col("column_10").alias("target_value_2026"),
            pl.col("column_11").alias("target_value_2027"),
            pl.col("original_order"),
        ]
    ).with_columns(
        [
            pl.col("indicator_code_original").forward_fill(),
            pl.col("indicator_name_original").forward_fill(),
            pl.col("unit_original").forward_fill(),
        ]
    )

    # recalculate original_order as the min original order for each combination of Indicateur_Name and Indicateur_Code (this is to ensure that sub-indicators that are separated by a new parent indicator are correctly assigned to their parent indicator in the next steps)
    # calculate field original_order_hundred equal to the hundred category of the order (e.g. 0-50 --> 0, 51-150 --> 100, etc.) and then calculate the min original order for each combination of Indicateur_Name, Indicateur_Code and original_order_hundred
    df_full = df_full.with_columns(
        (pl.col("original_order") // 100 * 100).alias("original_order_hundred")
    )
    df_full = df_full.with_columns(
        pl.col("original_order")
        .min()
        .over(
            [
                "indicator_name_original",
                "indicator_code_original",
                "original_order_hundred",
            ]
        )
        .alias("original_order")
    ).drop("original_order_hundred")

    # drop rows where country is missing
    df_full = df_full.filter(pl.col("country").is_not_null())

    # Replace "Valeur_xxx" values of "oui" and "non" with 1 and 0 respectively
    target_cols = [
        "target_value_2021",
        "target_value_2026",
        "target_value_2027",
    ]

    df_transformed = df_full.with_columns(
        [
            pl.col(col)
            .cast(pl.String)
            .str.to_lowercase()
            .replace({"oui": "1", "non": "0"})
            .str.replace(",", ".")
            .str.extract(r"(\d+\.?\d*)")
            .cast(pl.Float64, strict=False)
            .alias(col)
            for col in target_cols
        ]
    )

    # convert from wide to long format
    df_transformed = df_transformed.unpivot(
        on=target_cols,
        index=[
            "indicator_name_original",
            "unit_original",
            "country",
            "indicator_code_original",
            "original_order",
        ],
        variable_name="year",
        value_name="value",
    )
    df_transformed = df_transformed.with_columns(
        pl.col("year").str.extract(r"(\d{4})").cast(pl.Int32).alias("year")
    )

    # create value type col
    df_transformed = df_transformed.with_columns(pl.lit("target").alias("value_type"))

    current_run.log_info(f"Transformed {df_transformed.height} rows.")

    return df_transformed


def assign_indicator_codes(
    df_transformed: pl.DataFrame, indicators_metadata: pl.DataFrame
) -> pl.DataFrame:
    """ "
    Assign indicator codes to the transformed dataframe by matching indicator names with the
    reference map, and applying manual mapping for unmatched indicators.
    """
    # Assign indicator codes based on indicator names using the mapping dictionary from config
    df_transformed = df_transformed.with_columns(
        pl.col("indicator_name_original")
        .str.strip_chars()
        .alias("indicator_name_original")
    )
    df_transformed = df_transformed.with_columns(
        pl.col("indicator_name_original")
        .map_elements(
            lambda x: config.indicator_name_code_mapping.get(x, None),
            return_dtype=pl.String,
        )
        .alias("indicator_code_formatted")
    )

    # now if an indicator has its clean name starting with "dont " it is a sub-indicator.
    # For those, use the dataset sorted by their original order, and assign the last code seen + the suffix
    # "1", "2", etc.
    df_transformed = df_transformed.sort("original_order")
    df_transformed = df_transformed.with_columns(
        pl.col("indicator_name_original")
        # remove special character "-"
        .str.replace_all(r"- ", "")
        .str.to_lowercase()
        .str.starts_with("dont ")
        .alias("is_sub_indicator")
    )
    df_transformed = df_transformed.with_columns(
        pl.when(pl.col("is_sub_indicator").not_())
        .then(pl.col("indicator_code_formatted"))
        .otherwise(None)
        .fill_null(strategy="forward")
        .alias("parent_code")
    )
    is_new_name = (
        (
            (pl.col("is_sub_indicator"))
            & (
                pl.col("indicator_name_original")
                != pl.col("indicator_name_original").shift()
            )
        )
        .fill_null(False)
        .cast(pl.Int32)
    )
    df_transformed = df_transformed.with_columns(
        is_new_name.cum_sum().over("parent_code").alias("sub_suffix")
    )
    df_transformed = df_transformed.with_columns(
        pl.when(pl.col("is_sub_indicator").not_())
        .then(pl.col("indicator_code_formatted"))
        .otherwise(pl.col("parent_code") + pl.col("sub_suffix").cast(pl.Utf8))
        .alias("indicator_code_final")
    ).drop("parent_code", "sub_suffix", "is_sub_indicator")

    # join with metadata to retrieve indicator names (old and new), units (old and new) and status
    df_transformed = df_transformed.join(
        indicators_metadata.select(
            [
                pl.col("code").alias("indicator_code_final"),
                pl.col("designation").alias("indicator_name_old"),
                pl.col("designation_v2").alias("indicator_name_new"),
                pl.col("unite").alias("unit_old"),
                pl.col("unite_v2").alias("unit_new"),
                pl.col("note"),
            ]
        ),
        on="indicator_code_final",
        how="left",
    )

    # create indicator_status col (separately for years 2025 and 2026-2027)
    df_transformed = df_transformed.with_columns(
        pl.when(~pl.col("year").is_in([2021, 2026, 2027]))
        .then(pl.lit("unchanged"))
        .when(pl.col("note").str.contains(r"(?i)indicateur supprimé"))
        .then(pl.lit("deleted"))
        .when(pl.col("note").str.contains(r"(?i)nouveau nom d'indicateur"))
        .then(pl.lit("renamed"))
        .when(pl.col("note").str.contains(r"(?i)indicateur changé"))
        .then(pl.lit("renamed and unit changed"))
        .when(pl.col("note").str.contains(r"(?i)nouvel indicateur"))
        .then(pl.lit("new"))
        .otherwise(pl.lit("unchanged"))
        .alias("indicator_status")
    )

    # assign final indicators' names and units based on indicator status
    df_transformed = df_transformed.with_columns(
        # name
        pl.when(pl.col("indicator_status").is_in(["unchanged", "deleted"]))
        .then(pl.col("indicator_name_old"))
        .when(
            pl.col("indicator_status").is_in(
                ["renamed", "renamed and unit changed", "new"]
            )
        )
        .then(pl.col("indicator_name_new"))
        .otherwise(pl.lit(None))
        .alias("indicator_name_final"),
        # unit
        pl.when(pl.col("indicator_status").is_in(["unchanged", "deleted"]))
        .then(pl.col("unit_old"))
        .when(
            pl.col("indicator_status").is_in(
                ["renamed", "renamed and unit changed", "new"]
            )
        )
        .then(pl.col("unit_new"))
        .otherwise(pl.lit(None))
        .alias("unit_final"),
    )

    # only keep relevant cols
    df_transformed = df_transformed.select(
        [
            pl.col("indicator_code_final").alias("indicator_code"),
            pl.col("indicator_name_final").alias("indicator_name"),
            pl.col("unit_final").alias("unit_ref"),
            pl.col("unit_original"),
            pl.col("country"),
            pl.col("value"),
            pl.col("year"),
            pl.col("value_type"),
            pl.col("indicator_status"),
        ]
    )

    return df_transformed


def clean_results_values(cdr_df_results: pl.DataFrame) -> pl.DataFrame:
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
    # select result type values
    cdr_df_results = cdr_df_results.filter(pl.col("value_type") == "result")

    # clean country names based on config mapping
    cdr_df_results = cdr_df_results.with_columns(
        pl.col("country")
        .map_elements(
            lambda x: config.country_name_mapping.get(x, None), return_dtype=pl.String
        )
        .alias("country")
    )

    # convert values to standard values based on original unit column
    cdr_df_results = cdr_df_results.with_columns(
        pl.when(pl.col("unit_original").str.contains("(?i)million"))
        .then(pl.col("value") * 1_000_000)
        .when(pl.col("unit_original").str.contains("(?i)millier"))
        .then(pl.col("value") * 1_000)
        .otherwise(pl.col("value"))
    )

    # add suffix -01-01 to year and convert to date format to fit kobo data format
    cdr_df_results = cdr_df_results.with_columns(
        pl.col("year")
        .cast(pl.String)
        .map_elements(lambda x: f"{x}-01-01", return_dtype=pl.String)
        .alias("date")
    )

    # add level column (level 2 (Pays) except when country contains 'Régional', in which case level is 1 (Régional))
    cdr_df_results = cdr_df_results.with_columns(
        level=pl.when(pl.col("country").str.contains("Régional")).then(1).otherwise(2)
    )

    # add project col
    cdr_df_results = cdr_df_results.with_columns(pl.lit("PRAPS2").alias("project"))

    # clean column names
    cdr_df_results = cdr_df_results.select(
        [
            pl.col("indicator_code"),
            pl.col("indicator_name"),
            pl.col("unit_ref").alias("unit"),
            pl.col("year"),
            pl.col("date"),
            pl.col("project"),
            pl.col("level"),
            pl.col("country"),
            pl.col("value"),
            pl.col("indicator_status"),
        ]
    )

    return cdr_df_results


def clean_target_values(cdr_2026_2027_df: pl.DataFrame) -> pl.DataFrame:
    """
    Combine target values from 2025 and 2026-2027 CDRs into a single dataframe
    """
    # restrict both df to target values only
    cdr_2026_2027_targets_df = cdr_2026_2027_df.filter(pl.col("value_type") == "target")

    # sort indicator code, country and year
    cdr_2026_2027_targets_df = cdr_2026_2027_targets_df.sort(
        ["indicator_code", "country", "year"], descending=True
    )

    # create 'Composante' column based on indicator code (use config file)
    cdr_2026_2027_targets_df = cdr_2026_2027_targets_df.with_columns(
        pl.col("indicator_code")
        .map_elements(
            lambda x: next(
                (k for k, v in config.composante_indicator_mapping.items() if x in v),
                None,
            ),
            return_dtype=pl.String,
        )
        .alias("Composante")
    )

    # create "cumulative values" col taking boolean value false
    cdr_2026_2027_targets_df = cdr_2026_2027_targets_df.with_columns(
        pl.lit(False).alias("cumulative values")
    )

    # standardize values based on unit
    cdr_2026_2027_targets_df = cdr_2026_2027_targets_df.with_columns(
        pl.when(pl.col("unit_original").str.contains("(?i)millier"))
        .then(pl.col("value") * 1_000)
        .otherwise(pl.col("value"))
    )

    # select and rename relevant columns
    cdr_2026_2027_targets_df = cdr_2026_2027_targets_df.select(
        [
            pl.col("indicator_code").alias("Code"),
            pl.col("indicator_name").alias("Indicateur_Name"),
            pl.col("country").alias("Pays"),
            pl.col("Composante"),
            pl.col("year").alias("année"),
            pl.col("value").alias("valeur"),
            pl.col("unit_original").alias("unite"),
            pl.col("cumulative values"),
            pl.col("indicator_status"),
        ]
    )

    return cdr_2026_2027_targets_df


def combine_targets(
    cdr_targets_old: pl.DataFrame, cdr_2026_2027_targets_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Combine old target values (pre-2026) with new targets from revised CDR (2021 updated + 2026-2027) into a single dataframe
    """
    # remove updated data from old targets df (i.e. all target values for years 2021, 2026 and 2027)
    cdr_targets_old = cdr_targets_old.filter(
        ~(pl.col("année").is_in([2021, 2026, 2027]))
    )
    cdr_targets_old = cdr_targets_old.with_columns(
        pl.col("année").cast(pl.Int32),
        pl.col("valeur").cast(pl.Float64),
    )
    cdr_targets_old = cdr_targets_old.with_columns(
        pl.lit("unchanged").alias("indicator_status")
    )

    combined_targets_df = pl.concat(
        [cdr_targets_old, cdr_2026_2027_targets_df], how="diagonal"
    )

    # harmonize unit
    combined_targets_df = combined_targets_df.with_columns(
        pl.when(pl.col("unite").str.contains("(?i)nombre"))
        .then(pl.lit("count"))
        .when(pl.col("unite").str.contains("(?i)hectare|ha"))
        .then(pl.lit("surface"))
        .when(pl.col("unite").str.contains("(?i)tonne"))
        .then(pl.lit("weight"))
        .when(pl.col("unite").str.contains("(?i)pourcent"))
        .then(pl.lit("percent"))
        .when(pl.col("unite").str.contains("(?i)oui"))
        .then(pl.lit("boolean"))
        .otherwise(pl.col("unite"))
        .alias("unite")
    )

    # fill in missing indicator status by 'unchanged' (these are the ones that are still used from the old CDR)
    combined_targets_df = combined_targets_df.with_columns(
        pl.col("indicator_status").fill_null("unchanged")
    )

    return combined_targets_df


def save_output(df: pl.DataFrame, dir_name: str, file_name: str):
    """
    Save the processed dataframe to a parquet file in the specified directory and log the output file.
    """
    output_dir = Path(workspace.files_path, dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_name}.parquet"

    df.write_parquet(output_file)
    current_run.add_file_output(output_file.as_posix())


def push_to_db(df: pl.DataFrame, db_name: str) -> bool:
    current_run.log_info(f"Writing to database table `{db_name}`...")
    try:
        df.write_database(
            db_name, connection=workspace.database_url, if_table_exists="replace"
        )
        current_run.log_info(f"Writing to database table `{db_name}` ({len(df)} rows)")

        if get_environment() == Environment.CLOUD_PIPELINE:
            current_run.add_database_output(db_name)

        return True
    except Exception as e:
        msg = f"Error while writing to database table `{db_name}`: {e}"
        current_run.log_error(msg)
        raise


if __name__ == "__main__":
    process_cdr()
