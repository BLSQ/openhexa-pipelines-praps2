"""Consolidate the 4 legacy CDR surveys plus the Mali/HCDR form into one dataset with a
computed INFRASTRUCTURE_ID, matching the business logic of
add_infrastructure_id/consolidate_kobo_forms.py -- but downloading live from Kobo instead
of reading static Excel exports, and organized as a proper OpenHexa pipeline.

Uses openhexa.toolbox.kobo.Api for auth/pagination, but *not* its to_dataframe()/
cast_values() convenience wrapper: cast_values() casts `select_one` fields to their
label, while every Mali-specific mapping in config.py (work_type_code_mapping_mali,
infra_type_mapping_mali, etc.) is keyed on the raw Kobo choice code -- confirmed against
live data. rename_columns() (group-prefix stripping, e.g. "group1/DATE" -> "DATE") is
still used, since config.py's mapping keys are the bare field names.

Pushing the consolidated result back into Kobo (what
add_infrastructure_id/push_to_kobo_simplified_form_db.py currently does) is intentionally
not part of this pipeline -- planned separately as push_to_consolidated_kobo_db.
"""

from pathlib import Path

import polars as pl
import xlsxwriter
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.kobo import Api
from openhexa.toolbox.kobo.utils import rename_columns

import config
import utils


@pipeline(name="consolider-les-anciens-formulaires-kobo")
@parameter(
    "output_dir",
    name="Dossier de sortie",
    help="Répertoire où enregistrer les fichiers consolidés",
    type=str,
    default="data/kobo/",
)
def consolidate_legacy_surveys(output_dir: str):
    """Download the 4 CDR + 1 Mali/HCDR legacy surveys, compute INFRASTRUCTURE_ID for
    each, consolidate them into one dataset, and write both the consolidated file and a
    validation workbook.
    """
    output_dir = Path(workspace.files_path, output_dir)
    raw_files = download(output_dir)
    transformed = transform(raw_files)
    df = consolidate(transformed)
    write_outputs(df, output_dir)


def get_survey_uid(api: Api, kobo_name: str) -> str:
    """Look up a survey's asset uid by its exact live Kobo project name."""
    for survey in api.surveys:
        if survey["name"] == kobo_name:
            return survey["uid"]
    raise ValueError(f"No Kobo survey found with name {kobo_name!r}")


def download(output_dir: Path) -> dict:
    """Download raw submissions for all 5 legacy surveys, writing raw/{name}.parquet.

    Returns {name: Path} for the written raw files.
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cdr_api = Api(url=config.kobo_connector_slug_cdr["url"])
    cdr_api.authenticate(token=config.kobo_connector_slug_cdr["token"])
    hcdr_api = Api(url=config.kobo_connector_slug_hcdr["url"])
    hcdr_api.authenticate(token=config.kobo_connector_slug_hcdr["token"])

    raw_files = {}
    for name, record in config.LEGACY_SURVEYS.items():
        api = hcdr_api if record["account"] == "hcdr" else cdr_api
        uid = get_survey_uid(api, record["kobo_name"])
        survey = api.get_survey(uid)

        df = pl.DataFrame(survey.get_data(), infer_schema_length=None)
        df = rename_columns(df)
        if "rootUuid" in df.columns and "meta/rootUuid" not in df.columns:
            df = df.rename({"rootUuid": "meta/rootUuid"})

        dst = raw_dir / f"{name}.parquet"
        df.write_parquet(dst)
        current_run.log_info(f"{name}: downloaded {len(df)} submissions")
        raw_files[name] = dst

    return raw_files


def transform(raw_files: dict) -> dict:
    """Transform each survey's raw parquet into the consolidated column shape."""
    transformed = {}
    for name, path in raw_files.items():
        df = pl.read_parquet(path)
        if config.LEGACY_SURVEYS[name]["account"] == "hcdr":
            df = transform_hcdr_survey(df, name)
        else:
            df = transform_cdr_survey(df, name)
        transformed[name] = df
        current_run.log_info(f"{name}: {len(df)} entries after transform")
    return transformed


def transform_cdr_survey(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Transform one of the 4 legacy CDR surveys into the consolidated column shape."""
    df = df.filter(~pl.col("_uuid").is_in(config.entry_errors))
    df = add_location_columns(df, name)

    infra_acronym = config.LEGACY_SURVEYS[name]["acronym"]
    lat_str = pl.col("LATITUDE").round(2).cast(pl.String).str.replace_all(r"[\.-]", "")
    lon_str = pl.col("LONGITUDE").round(2).cast(pl.String).str.replace_all(r"[\.-]", "")
    df = df.with_columns(
        pl.when(pl.col("LATITUDE").is_not_null() & pl.col("LONGITUDE").is_not_null())
        .then(pl.concat_str([pl.lit(f"{infra_acronym}_"), lat_str, lon_str]))
        .otherwise(pl.lit(None))
        .alias("INFRASTRUCTURE_ID")
    )
    df = resolve_infrastructure_id(df)

    df = df.with_columns(
        pl.lit(1).alias("TYPE"),  # CDR
        pl.lit(config.LEGACY_SURVEYS[name]["cdr_code"]).alias("CDR"),
        pl.lit(infra_acronym).alias("TYPE_ACRONYM"),
        pl.lit(None).cast(pl.Int64).alias("HCDR"),
        pl.lit(None).cast(pl.Utf8).alias("HCDRa"),
    )

    return select_and_rename(df)


def transform_hcdr_survey(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Transform the HCDR survey into the consolidated column shape."""
    df = df.filter(~pl.col("_uuid").is_in(config.entry_errors))
    df = add_location_columns(df, name)

    df = df.with_columns(
        pl.col(config.infra_type_col_mali)
        .replace_strict(config.infra_type_acronym_mapping_mali, default="AUTRE")
        .alias("TYPE_ACRONYM")
    )
    lat_str = pl.col("LATITUDE").round(2).cast(pl.String).str.replace_all(r"[\.-]", "")
    lon_str = pl.col("LONGITUDE").round(2).cast(pl.String).str.replace_all(r"[\.-]", "")
    df = df.with_columns(
        pl.when(pl.col("LATITUDE").is_not_null() & pl.col("LONGITUDE").is_not_null())
        .then(pl.concat_str([pl.col("TYPE_ACRONYM"), pl.lit("_"), lat_str, lon_str]))
        .otherwise(pl.lit(None))
        .alias("INFRASTRUCTURE_ID")
    )
    df = resolve_infrastructure_id(df)

    df = df.with_columns(
        pl.lit(2).alias("TYPE"),  # HCDR
        pl.lit(None).cast(pl.Int32).alias("CDR"),
    )
    other_types = [k for k, v in config.infra_type_mapping_mali.items() if v == 13]
    df = df.with_columns(
        pl.col(config.infra_type_col_mali)
        .replace_strict(config.infra_type_mapping_mali, default=None)
        .alias("HCDR"),
        pl.when(pl.col(config.infra_type_col_mali).is_in(other_types))
        .then(
            pl.col(config.infra_type_col_mali)
            .cast(pl.Utf8)
            .replace(config.infra_type_other_mali_cleaning)
        )
        .otherwise(None)
        .cast(pl.Utf8)
        .alias("HCDRa"),
    )

    df = _apply_if_present(
        df,
        config.collector_function_col_mali,
        lambda e: e.replace_strict(
            config.collector_function_code_mapping_mali, default=None
        ),
    )
    df = _apply_if_present(
        df, config.collector_contact_col_mali, lambda e: e.cast(pl.Utf8)
    )
    df = _apply_if_present(
        df, config.controller_contact_col_mali, lambda e: e.cast(pl.Utf8)
    )
    df = _apply_if_present(
        df, config.investigated_contact_col_mali, lambda e: e.cast(pl.Utf8)
    )
    df = _apply_if_present(
        df, config.reception_date_col_mali, lambda e: e.cast(pl.Date)
    )
    df = _apply_if_present(df, config.country_col_mali, lambda e: e.str.to_titlecase())
    df = _apply_if_present(
        df,
        config.work_type_col_mali,
        lambda e: e.replace_strict(config.work_type_code_mapping_mali, default=None),
    )
    df = _apply_if_present(
        df,
        config.work_duration_col_mali,
        lambda e: e.replace(config.work_duration_code_mapping_mali),
    )
    df = _apply_if_present(
        df,
        config.work_duration_col_mali,
        lambda e: pl.when(e.is_not_null())
        .then(
            e.cast(pl.Utf8)
            .str.replace_all(r"mois", "")
            .str.strip_chars()
            .cast(pl.Int64)
        )
        .otherwise(pl.lit(None)),
    )
    df = _apply_if_present(
        df,
        config.implementation_level_col_mali,
        lambda e: e.replace_strict(
            config.implementation_level_code_mapping_mali, default=None
        ),
    )
    df = _apply_if_present(
        df,
        config.work_completion_rate_col_mali,
        lambda e: e.replace_strict(
            config.work_completion_rate_code_mapping, default=None
        ),
    )

    return select_and_rename(df)


def _apply_if_present(df: pl.DataFrame, col: str, fn) -> pl.DataFrame:
    """Apply `fn(pl.col(col))` back onto `col`, or skip if `col` is absent from this
    batch (see the call site above for why that happens routinely for Mali fields).
    """
    if col not in df.columns:
        current_run.log_info(f"`{col}` absent from this batch -- skipping")
        return df
    return df.with_columns(fn(pl.col(col)).alias(col))


def add_location_columns(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Derive LATITUDE/LONGITUDE and a LUV6 geopoint string from Kobo's own reliable
    `_geolocation` field, and drop the survey's own named geopoint field.
    """
    df = df.drop(config.LEGACY_SURVEYS[name]["geoloc_col"], strict=False)
    lat = pl.col("_geolocation").list.get(0, null_on_oob=True)
    lon = pl.col("_geolocation").list.get(1, null_on_oob=True)
    has_location = lat.is_not_null() & lon.is_not_null()
    df = df.with_columns(
        pl.when(has_location).then(lat).otherwise(None).alias("LATITUDE"),
        pl.when(has_location).then(lon).otherwise(None).alias("LONGITUDE"),
    )
    df = df.with_columns(
        pl.when(pl.col("LATITUDE").is_not_null() & pl.col("LONGITUDE").is_not_null())
        .then(
            pl.concat_str(
                [
                    pl.col("LATITUDE").cast(pl.String),
                    pl.lit(" "),
                    pl.col("LONGITUDE").cast(pl.String),
                    pl.lit(" 0 0"),
                ]
            )
        )
        .otherwise(pl.lit(None))
        .alias("LUV6")
    )
    return df


def resolve_infrastructure_id(df: pl.DataFrame) -> pl.DataFrame:
    """Identify geographic duplicates and apply manual corrections.

    Identical for every source survey once INFRASTRUCTURE_ID has been computed.
    """
    df = utils.identify_duplicates(
        df, column_latitude="LATITUDE", column_longitude="LONGITUDE", min_distance=1.5
    )
    uuid_to_infra_map = dict(zip(df["_uuid"], df["INFRASTRUCTURE_ID"]))
    source_to_infra_map = {
        src_uuid: uuid_to_infra_map[target_uuid]
        for src_uuid, target_uuid in config.duplicates_manual_correction_mapping.items()
        if target_uuid in uuid_to_infra_map
    }
    df = df.with_columns(
        INFRASTRUCTURE_ID=pl.col("_uuid").replace_strict(
            source_to_infra_map, default=pl.col("INFRASTRUCTURE_ID")
        )
    )
    return df


def select_and_rename(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only the columns config.all_forms_cols_mapping declares a target for, and
    rename them to the consolidated form's field names.
    """
    if "_validation_status" in df.columns and isinstance(
        df.schema["_validation_status"], pl.Struct
    ):
        df = df.with_columns(pl.col("_validation_status").struct.json_encode())

    df = df.select(
        [
            col
            for col in df.columns
            if col in config.all_forms_cols_mapping
            and config.all_forms_cols_mapping[col] != ""
        ]
    )
    df = df.rename({col: config.all_forms_cols_mapping[col] for col in df.columns})
    return df


def consolidate(transformed: dict) -> pl.DataFrame:
    """Concatenate every survey's transformed dataframe, logging column differences and
    reconciling any dtype mismatches for columns shared across surveys first.
    """
    dfs = dict(transformed)
    names = list(dfs.keys())

    dtypes_by_col = {}
    for df in dfs.values():
        for col, dtype in df.schema.items():
            dtypes_by_col.setdefault(col, set()).add(dtype)

    for col, dtypes in dtypes_by_col.items():
        if len(dtypes) <= 1:
            continue
        target_dtype = pl.Int64
        for name in names:
            if col not in dfs[name].columns:
                continue
            dtype = dfs[name].schema[col]
            if isinstance(dtype, (pl.List, pl.Struct)):
                target_dtype = pl.Utf8
                break
            non_null = dfs[name][col].drop_nulls()
            if len(non_null) == 0:
                continue
            try:
                non_null.cast(pl.Int64, strict=True)
            except Exception:
                target_dtype = pl.Utf8
                break

        current_run.log_info(
            f"Column `{col}` has mixed dtypes across surveys ({dtypes}) -- casting to {target_dtype}"
        )
        for name in names:
            if col in dfs[name].columns and dfs[name].schema[col] != target_dtype:
                dfs[name] = dfs[name].with_columns(pl.col(col).cast(target_dtype))

    columns_by_name = {name: set(df.columns) for name, df in dfs.items()}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            only_in_a = columns_by_name[name_a] - columns_by_name[name_b]
            only_in_b = columns_by_name[name_b] - columns_by_name[name_a]
            if only_in_a:
                current_run.log_info(
                    f"Columns in {name_a} not in {name_b}: {only_in_a}"
                )
            if only_in_b:
                current_run.log_info(
                    f"Columns in {name_b} not in {name_a}: {only_in_b}"
                )

    df = pl.concat(list(dfs.values()), how="diagonal")
    current_run.log_info(f"Consolidated {len(df)} records from {len(names)} surveys")
    return df


def write_outputs(df: pl.DataFrame, output_dir: Path):
    """Write the consolidated dataset and a validation workbook (per-country + Mali_HCDR
    sheets).
    """
    output_folder = output_dir / "surveys"
    output_folder.mkdir(parents=True, exist_ok=True)

    consolidated_path = output_folder / "CONSOLIDATED_DB_with_infrastructure_id.xlsx"
    consolidated_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_excel(consolidated_path)
    current_run.log_info(f"Wrote {consolidated_path} ({len(df)} records)")

    df_validation = df.with_columns(
        pl.when(pl.col("INFRASTRUCTURE_ID").is_duplicated())
        .then(pl.lit("Oui"))
        .otherwise(pl.lit("Non"))
        .alias("Doublon")
    )
    df_validation = df_validation.sort(["INFRASTRUCTURE_ID", "starttime"])
    ordered_cols = [
        col for col in config.name_label_mapping.keys() if col in df_validation.columns
    ]
    missing_cols = [col for col in df_validation.columns if col not in ordered_cols]
    df_validation = df_validation.select(ordered_cols + missing_cols)
    df_validation = df_validation.rename(
        {col: config.name_label_mapping.get(col, col) for col in df_validation.columns}
    )

    validation_path = (
        output_folder / "CONSOLIDATED_DB_with_infrastructure_id_validation.xlsx"
    )
    with xlsxwriter.Workbook(str(validation_path)) as wb:
        df_hcdr = df_validation.filter(pl.col("8) Nature de l'indicateur") == 2)
        if len(df_hcdr) > 0:
            df_hcdr.write_excel(workbook=wb, worksheet="Mali_HCDR")
            current_run.log_info(f"{len(df_hcdr)} records for Mali_HCDR")

        for country in df_validation["20) Pays"].unique():
            df_country = df_validation.filter(
                (pl.col("20) Pays") == country)
                & (pl.col("8) Nature de l'indicateur") == 1)
            )
            df_country.write_excel(workbook=wb, worksheet=country)
            current_run.log_info(f"{len(df_country)} records for {country}")

    current_run.log_info(f"Wrote {validation_path}")


if __name__ == "__main__":
    consolidate_legacy_surveys()
