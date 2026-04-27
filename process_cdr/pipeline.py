import polars as pl
from openhexa.sdk import current_run, pipeline, workspace
from openhexa.sdk.utils import Environment, get_environment

import re
from pathlib import Path


@pipeline("process-cdr", name="process-cdr")
def process_cdr():
    """
    Pipeline to process CDR Excel file and transform it into a long format Polars dataframe.
    """
    cdr_raw = import_file(cdr_file)
    cdr_df = process(cdr_file)
    save_output(cdr_df, "cdr_transformed")


def import_file(cdr_dir: str) -> pl.DataFrame:
    """Import most recent CDR file from the specified directory"""
    cdr_path = Path(cdr_dir)
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


def process(df_raw: pl.DataFrame) -> pl.DataFrame:
    """Process the raw CDR dataframe to extract year, forward fill merged cells, and transform to long format."""
    current_run.log_info("Processing CDR raw data...")

    # 1. Identify Year from Row 4 (index 3) or surrounding rows
    year = None
    for r_idx in [2, 3, 4]:
        if r_idx >= df_raw.height:
            continue
        row = df_raw.row(r_idx)
        for cell in row:
            if cell and "CIBLE FIN" in str(cell):
                match = re.search(r"CIBLE FIN (\d{4})", str(cell))
                if match:
                    year = int(match.group(1))
                    break
        if year:
            break

    if not year:
        year = None

    # 2. Map and Forward Fill the whole dataframe first
    # This ensures that merged indicator names are captured before we slice/filter.
    df_full = df_raw.select(
        [
            pl.col("column_1").alias("Code"),
            pl.col("column_2").alias("Indicateur_Name"),
            pl.col("column_4").alias("Unité"),
            pl.col("column_5").alias("Pays"),
            pl.col("column_8").alias("Valeur"),
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

    # Filter rows where Pays or Valeur is null
    df_transformed = df_transformed.filter(
        pl.col("Pays").is_not_null() & pl.col("Valeur").is_not_null()
    )

    # Convert Valeur to numeric
    df_transformed = df_transformed.with_columns(
        pl.col("Valeur").cast(pl.Float64, strict=False)
    )

    current_run.log_info(f"Transformed {df_transformed.height} rows.")

    return df_transformed


def save_output(df: pl.DataFrame, table_name: str):
    output_dir = Path(workspace.files_path, "data/cdr/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{table_name}.parquet"

    df.write_parquet(output_file)
    current_run.add_file_output(output_file.as_posix())

    return df.as_posix()


if __name__ == "__main__":
    process_cdr()

# HEXA_SERVER_URL=https://app.openhexa.org
# HEXA_TOKEN=IjI1MGNjNTllLThhMWItNDRlMi1hMTBhLTE0NjY0ZWM1YmU5YiI:uRZW8ZFWqJEtmIhSCntwiF4GNQw6mPFKs5WJ9YIWmTE
# HEXA_WORKSPACE=praps-f5e786
