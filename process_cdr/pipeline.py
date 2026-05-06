import polars as pl
from openhexa.sdk import current_run, parameter, pipeline, workspace

from pathlib import Path
import utils
import config


@pipeline("process-cdr", name="Traiter les données des CDR")
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
    default="data/cdr/indicators_metadata_v2.csv",
)
def process_cdr(
    cdr_raw_dir: str,
    cdr_processed_dir: str,
    cdr_indicators_ref_file: str,
):
    """
    Pipeline to process CDR Excel files and transform them into a long format Polars dataframe.
    """
    # process 2025 CDR
    cdr_indicators_map = import_indicators_map(cdr_indicators_ref_file)
    cdr_2025_raw = import_file(
        cdr_raw_dir, "CDR_PRAPS_2_CONSOLIDE_PAYS_CILSS_31_12_25_VF.xlsx"
    )
    cdr_2025_df = process_2025_CDR(cdr_2025_raw)
    cdr_2025_df = assign_indicator_codes(cdr_2025_df, cdr_indicators_map)

    # process 2026-2027 CDR
    cdr_2026_2027_raw = import_file(
        cdr_raw_dir, "PRAPS-2 Projet de CDR révisé - décembre 2025 VF 191225.xlsx"
    )
    cdr_2026_2027_df = process_2026_2027_CDR(cdr_2026_2027_raw)
    cdr_2026_2027_df = assign_indicator_codes(cdr_2026_2027_df, cdr_indicators_map)

    # create combined target df
    cdr_targets_df = combine_target_values(cdr_2025_df, cdr_2026_2027_df)

    # create cdr results df
    cdr_2025_results_df = clean_results_values(cdr_2025_df)

    # save outputs
    save_output(cdr_2025_results_df, cdr_processed_dir, "cdr_results_2025")
    save_output(cdr_targets_df, cdr_processed_dir, "CDR_Targets_v2")


def import_file(cdr_dir: str, file_name: str) -> pl.DataFrame:
    """Import most recent CDR file from the specified directory"""
    cdr_path = Path(workspace.files_path, cdr_dir)
    if not cdr_path.exists() or not cdr_path.is_dir():
        raise FileNotFoundError(
            f"Directory {cdr_dir} does not exist or is not a directory."
        )

    df_raw = pl.read_excel(Path(cdr_path, file_name), has_header=False)

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


def process_2025_CDR(cdr_2025_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Process the raw CDR dataframe to extract year, forward fill merged cells, and transform to
    long format.
    """
    current_run.log_info("Processing CDR raw data...")

    # drop first 3 rows
    cdr_2025_raw = cdr_2025_raw.slice(4)

    # create order column to keep track of original order for sub-indicators assignment later on
    cdr_2025_raw = cdr_2025_raw.with_row_index("original_order")

    # 2. Map and Forward Fill the whole dataframe first
    # This ensures that merged indicator names are captured before we slice/filter.
    df_full = cdr_2025_raw.select(
        [
            pl.col("column_1").alias("indicator_code_original"),
            pl.col("column_2").alias("indicator_name_original"),
            pl.col("column_4").alias("unit_original"),
            pl.col("column_5").alias("country"),
            pl.col("column_7").alias("target_value"),
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

    # recalculate original_order as the min original order for each combination of indicator_name_original and indicator_code_original (this is to ensure that sub-indicators that are separated by a new parent indicator are correctly assigned to their parent indicator in the next steps)
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

    # Filter out rows where Valeur résultats is missing or contains the term "Résultats"
    df_transformed = df_transformed.filter(
        pl.col("result_value").is_not_null()
        & ~pl.col("result_value").str.contains("Résultats")
    )

    # Replace "Valeur_xxx" values of "oui" and "non" with 1 and 0 respectively
    for col in ["target_value", "result_value"]:
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
        pl.col("target_value").cast(pl.Float64, strict=False),
        pl.col("result_value").cast(pl.Float64, strict=False),
    )

    # convert from wide to long format
    df_transformed = df_transformed.unpivot(
        on=["target_value", "result_value"],
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

    # fill in missing country values
    df_transformed = df_transformed.with_columns(
        pl.when(pl.col("country").is_not_null())
        .then(pl.col("country"))
        .when((pl.col("indicator_code_original") == "IRI-6") & (pl.col("value") == 185))
        .then(pl.lit("MR"))
        .when(
            (pl.col("indicator_code_original") == "IRI-7") & (pl.col("value") == 5_576)
        )
        .then(pl.lit("MR"))
        .when((pl.col("indicator_code_original").is_in(["FA-2", "FA-21", "FA-22"])))
        .then(pl.lit("BF"))
        .when(pl.col("indicator_code_original").is_in(config.regional_indicators))
        .then(pl.lit("REGIONAL"))
        .otherwise(pl.lit("NE"))
        .alias("country")
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
            pl.col("column_7").alias("target_value_2022"),
            pl.col("column_8").alias("target_value_2023"),
            pl.col("column_9").alias("target_value_2024"),
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
        "target_value_2022",
        "target_value_2023",
        "target_value_2024",
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
    df_transformed: pl.DataFrame, cdr_indicators_map: pl.DataFrame
) -> pl.DataFrame:
    """ "
    Assign indicator codes to the transformed dataframe by matching indicator names with the
    reference map, and applying manual mapping for unmatched indicators.
    """
    # define relevant columns based on whether CDR is 2025 or 2026-2027
    code_col = "code_v2"
    name_col = "designation"
    new_name_col = "designation_v2"
    unit_col = "unite_v2"
    cdr_is_2025 = (
        df_transformed.select(pl.col("year").unique()).to_series().item(0) == 2025
    )

    # restrict the df_transformed to Indicateur name and code and the original order
    df_transformed_restricted = df_transformed.select(
        ["indicator_name_original", "indicator_code_original", "original_order"]
    ).unique()

    if not cdr_is_2025:
        name_col = name_col + "_v2"

    # first normalize the indicator names in both dataframes to allow matching
    df_transformed_restricted = df_transformed_restricted.with_columns(
        utils.normalize_indicator_column("indicator_name_original").alias(
            "indicator_name_clean"
        )
    )

    cdr_indicators_map = cdr_indicators_map.with_columns(
        utils.normalize_indicator_column(name_col).alias("indicator_name_clean")
    )

    # select relevant cols
    cdr_indicators_map = cdr_indicators_map.select(
        [
            pl.col("indicator_name_clean"),
            pl.col(code_col).alias("indicator_code_ref"),
            pl.col(name_col).alias("indicator_name_ref"),
            pl.col(new_name_col).alias("indicator_name_ref_new"),
            pl.col(unit_col).alias("unit_ref"),
            pl.col("note"),
        ]
    )

    # drop sub-indicators (those starting with the term "Dont ") (these will be calculated later on)
    cdr_indicators_map_reduced = cdr_indicators_map.filter(
        ~pl.col("indicator_name_clean").str.starts_with("dont ")
    ).drop(["note"])

    # join with reference map
    df_transformed_restricted = df_transformed_restricted.join(
        cdr_indicators_map_reduced,
        on=["indicator_name_clean"],
        how="left",
    )

    # perform manual matching on remaining unmatched indicators using the mapping defined in config.py
    for indicator_name, code in config.missing_indicator_code_mapping.items():
        df_transformed_restricted = df_transformed_restricted.with_columns(
            pl.when(pl.col("indicator_name_original") == indicator_name)
            .then(pl.lit(code))
            .otherwise(pl.col("indicator_code_ref"))
            .alias("indicator_code_ref")
        )

    # now if an indicator has its clean name starting with "dont " it is a sub-indicator.
    # For those, use the dataset sorted by their original order, and assign the last code seen + the suffix
    # "1", "2", etc.
    df_transformed_restricted = df_transformed_restricted.sort("original_order")
    df_transformed_restricted = df_transformed_restricted.with_columns(
        pl.col("indicator_name_clean")
        .str.starts_with("dont ")
        .alias("is_sub_indicator")
    )
    df_transformed_restricted = df_transformed_restricted.with_columns(
        pl.when(pl.col("is_sub_indicator").not_())
        .then(pl.col("indicator_code_ref"))
        .otherwise(None)
        .fill_null(strategy="forward")
        .alias("parent_code")
    )
    is_new_name = (
        (
            (pl.col("is_sub_indicator"))
            & (pl.col("indicator_name_clean") != pl.col("indicator_name_clean").shift())
        )
        .fill_null(False)
        .cast(pl.Int32)
    )
    df_transformed_restricted = df_transformed_restricted.with_columns(
        is_new_name.cum_sum().over("parent_code").alias("sub_suffix")
    )
    df_transformed_restricted = df_transformed_restricted.with_columns(
        pl.when(pl.col("is_sub_indicator").not_())
        .then(pl.col("indicator_code_ref"))
        .otherwise(pl.col("parent_code") + pl.col("sub_suffix").cast(pl.Utf8))
        .alias("indicator_code_final")
    )

    # merge back on reference map to retrieve indicator name and unit for sub-indicators and failed matches
    df_transformed_restricted = df_transformed_restricted.join(
        cdr_indicators_map.select(
            [
                pl.col("indicator_code_ref").alias("indicator_code_final"),
                pl.col("indicator_name_ref_new").alias("indicator_name_ref_final"),
                pl.col("unit_ref").alias("unit_ref_final"),
                pl.col("note"),
            ]
        ),
        on="indicator_code_final",
        how="left",
    )

    # use the new indicator names for all indicators
    df_transformed_restricted = df_transformed_restricted.with_columns(
        pl.col("indicator_name_ref_final").alias("indicator_name_final"),
        pl.col("unit_ref_final"),
    )

    # retrict to relevant columns before merging back on the original df_transformed
    df_transformed_restricted = df_transformed_restricted.select(
        [
            "indicator_name_original",
            "indicator_code_original",
            "indicator_name_final",
            "indicator_code_final",
            "unit_ref_final",
            "note",
            "original_order",
        ]
    )

    # merge back onto original df by Indicateur_Name and Indicateur_Code to retrieve indicator names and units for all rows (including sub-indicators)
    cdr_columns = [
        "indicator_name_original",
        "indicator_code_original",
        "unit_original",
        "country",
        "value",
        "value_type",
        "year",
        "original_order",
    ]

    df_transformed = df_transformed.select(cdr_columns).join(
        df_transformed_restricted,
        on=["indicator_name_original", "indicator_code_original", "original_order"],
        how="left",
    )

    # drop indicators no longer relevant in cdr 2025
    if cdr_is_2025:
        df_transformed = df_transformed.filter(
            pl.col("note").is_null()
            | ~pl.col("note").str.contains(r"(?i)indicateur supprimé|indicateur changé")
        )

    # drop missing indicators
    df_transformed = df_transformed.filter(pl.col("indicator_code_final").is_not_null())

    # drop intermediate columns and rename final code column
    df_transformed = df_transformed.drop(
        [
            "indicator_name_original",
            "indicator_code_original",
            "note",
        ]
    ).rename(
        {
            "indicator_code_final": "indicator_code",
            "indicator_name_final": "indicator_name",
            "unit_ref_final": "unit_ref",
        }
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
            lambda x: config.country_name_mapping.get(x, x), return_dtype=pl.String
        )
        .alias("country")
    )

    # convert values to standard values based on original unit column
    cdr_df_results = cdr_df_results.with_columns(
        pl.when(pl.col("unit_original").str.contains("(?i)million"))
        .then(pl.col("value") * 1_000_000)
        .when(pl.col("unit_original").str.contains("(?i)millier"))
        .then(pl.col("value") * 1_000)
        .when(pl.col("unit_original").str.contains("(?i)pourcent"))
        .then(pl.col("value") / 100)
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
        ]
    )

    return cdr_df_results


def combine_target_values(
    cdr_2025_df: pl.DataFrame, cdr_2026_2027_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Combine target values from 2025 and 2026-2027 CDRs into a single dataframe
    """
    # restrict both df to target values only
    cdr_2025_targets_df = cdr_2025_df.filter(pl.col("value_type") == "target")
    cdr_2026_2027_targets_df = cdr_2026_2027_df.filter(pl.col("value_type") == "target")

    # concatenate the two dataframes
    combined_targets = pl.concat([cdr_2025_targets_df, cdr_2026_2027_targets_df])

    # sort indicator code, country and year
    combined_targets = combined_targets.sort(
        ["indicator_code", "country", "year"], descending=True
    )

    # create 'Composante' column based on indicator code (use config file)
    combined_targets = combined_targets.with_columns(
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
    combined_targets = combined_targets.with_columns(
        pl.lit(False).alias("cumulative_values")
    )

    # select and rename relevant columns
    combined_targets = combined_targets.select(
        [
            pl.col("indicator_code").alias("Code"),
            pl.col("indicator_name").alias("Indicateur_Name"),
            pl.col("country").alias("Pays"),
            pl.col("Composante"),
            pl.col("year").alias("année"),
            pl.col("value").alias("valeur"),
            pl.col("unit_original").alias("unite"),
            pl.col("cumulative_values"),
        ]
    )

    # clean unit column (remove trailing blanks, inner blanks, and white spaces (e.g. "hectares (milliers,                  valeur cumulée)" should be "hectares (milliers, valeur cumulée)")
    combined_targets = combined_targets.with_columns(
        pl.col("unite")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .map_elements(lambda x: config.unit_mapping.get(x, x))
        .alias("unite")
    )

    return combined_targets


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
