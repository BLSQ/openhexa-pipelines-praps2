import polars as pl
import xlsxwriter
import config
import utils

## CDR INFRASTRUCTURES
df_list = []
for file in config.INPUT_FILES:
    print(f"Reading data from Excel workbook: {file}...")
    df = pl.read_excel(file)

    # 1. Drop entries identified as errors by client (based on UUID)
    df = df.filter(~pl.col("_uuid").is_in(config.entry_errors))

    # 2. Extract and format the Latitude and Longitude
    geoloc_col = config.geoloc_cols_mapping.get(file.split("/")[-1].split(".")[0])
    lat_str = (
        (
            df[f"_{geoloc_col}_latitude"].round(2)
        )  # we use 2 decimals to round the coordinates to ~1.5km precision
        .cast(pl.String)
        .str.replace_all(r"[\.-]", "")
    )
    lon_str = (
        (df[f"_{geoloc_col}_longitude"].round(2))
        .cast(pl.String)
        .str.replace_all(r"[\.-]", "")
    )

    # 3. Build the final INFRASTRUCTURE_ID column

    # special rule for Mali form: the infrastructure type is recorded in the colum infra_type_col_mali in congig file
    if "FICHE AIRE D'ABATTAGE, ETAL, MAGASINS, LATRINES_MALI" in file:
        infra_type_col = config.infra_type_col_mali
        df = df.with_columns(
            pl.col(infra_type_col)
            .replace(config.infra_type_acronym_mapping_mali, default="AUTRE")
            .alias("TYPE_ACRONYM")
        )
        df = df.with_columns(
            pl.when(pl.col(geoloc_col).is_not_null() & (pl.col(geoloc_col) != ""))
            .then(
                pl.concat_str(
                    [
                        pl.col("TYPE_ACRONYM"),
                        pl.lit("_"),
                        lat_str,
                        lon_str,
                    ]
                )
            )
            .otherwise(pl.lit(None))  # Leaves the ID blank if there is no GPS data
            .alias("INFRASTRUCTURE_ID")
        )
    else:
        infra_acronym = config.infra_acronym_mapping.get(
            file.split("/")[-1].split(".")[0]
        )
        df = df.with_columns(
            pl.when(pl.col(geoloc_col).is_not_null() & (pl.col(geoloc_col) != ""))
            .then(
                pl.concat_str(
                    [
                        pl.lit(f"{infra_acronym}_"),
                        lat_str,
                        lon_str,
                    ]
                )
            )
            .otherwise(pl.lit(None))  # Leaves the ID blank if there is no GPS data
            .alias("INFRASTRUCTURE_ID")
        )

    # Identify duplicate INFRASTRUCTURE_ID based on geographic coordinates (with a minimum distance of 1.5km)
    df = utils.identify_duplicates(
        df,
        column_latitude=f"_{geoloc_col}_latitude",
        column_longitude=f"_{geoloc_col}_longitude",
        min_distance=1.5,
    )

    # apply manual corrections to duplicates based on UUID
    uuid_to_infra_map = dict(zip(df["_uuid"], df["INFRASTRUCTURE_ID"]))
    source_to_infra_map = {
        src_uuid: uuid_to_infra_map[target_uuid]
        for src_uuid, target_uuid in config.duplicates_manual_correction_mapping.items()
        if target_uuid in uuid_to_infra_map
    }

    df = df.with_columns(
        INFRASTRUCTURE_ID=pl.col("_uuid").replace(
            source_to_infra_map, default=pl.col("INFRASTRUCTURE_ID")
        )
    )

    # # create geometry columns
    # df = df.with_columns(
    #     pl.col(f"{geoloc_col}").alias("geometry"),
    #     pl.col(f"_{geoloc_col}_latitude").alias("latitude"),
    #     pl.col(f"_{geoloc_col}_longitude").alias("longitude"),
    # )

    # Add extra cols from the new Kobo form (TYPE, CDR, HCDR, HCDRa)
    if "FICHE AIRE D'ABATTAGE, ETAL, MAGASINS, LATRINES_MALI" in file:
        df = df.with_columns(
            pl.lit(2).alias("TYPE"),  # HCDR
            pl.lit(None).cast(pl.Int32).alias("CDR"),  # CDR col --> None
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

    else:
        df = df.with_columns(
            pl.lit(1).alias("TYPE"),  # CDR
        )

        for file_name, value_code in config.infra_type_mapping.items():
            if file_name in file:
                df = df.with_columns(
                    pl.lit(value_code).alias("CDR"),
                    pl.lit(config.infra_acronym_mapping[file_name]).alias(
                        "TYPE_ACRONYM"
                    ),
                    pl.lit(None).cast(pl.Int64).alias("HCDR"),
                    pl.lit(None).cast(pl.Utf8).alias("HCDRa"),
                )

    # adjust column types in mali form to match with other forms
    if "FICHE AIRE D'ABATTAGE, ETAL, MAGASINS, LATRINES_MALI" in file:
        df = df.with_columns(
            pl.col(config.collector_function_col_mali)
            .replace(config.collector_function_code_mapping_mali, default=None)
            .alias(config.collector_function_col_mali)
        )
        df = df.with_columns(
            pl.col(config.collector_contact_col_mali)
            .cast(pl.Utf8)
            .alias(config.collector_contact_col_mali),
            pl.col(config.controller_contact_col_mali)
            .cast(pl.Utf8)
            .alias(config.controller_contact_col_mali),
            pl.col(config.investigated_contact_col_mali)
            .cast(pl.Utf8)
            .alias(config.investigated_contact_col_mali),
            pl.col(config.reception_date_col_mali)
            .cast(pl.Date)
            .alias(config.reception_date_col_mali),
        )

        # add capital to first letter of country name
        df = df.with_columns(
            pl.col(config.country_col_mali)
            .str.to_titlecase()
            .alias(config.country_col_mali)
        )

        # encode values for work type col
        df = df.with_columns(
            pl.col(config.work_type_col_mali)
            .replace(config.work_type_code_mapping_mali, default=None)
            .alias(config.work_type_col_mali)
        )

        # set work duration col to int
        df = df.with_columns(
            pl.col(config.work_duration_col_mali)
            .replace(config.work_duration_code_mapping_mali)
            .alias(config.work_duration_col_mali)
        )
        df = df.with_columns(
            pl.when(pl.col(config.work_duration_col_mali).is_not_null())
            .then(
                pl.col(config.work_duration_col_mali)
                .cast(pl.Utf8)
                .str.replace_all(r"mois", "")
                .str.strip_chars()
                .cast(pl.Int64)
            )
            .otherwise(pl.lit(None))
            .alias(config.work_duration_col_mali)
        )

        # encode values for implementation level col
        df = df.with_columns(
            pl.col(config.implementation_level_col_mali)
            .replace(config.implementation_level_code_mapping_mali, default=None)
            .alias(config.implementation_level_col_mali)
        )

        # encode values for work completion rate col
        df = df.with_columns(
            pl.col(config.work_completion_rate_col_mali)
            .replace(config.work_completion_rate_code_mapping, default=None)
            .alias(config.work_completion_rate_col_mali)
        )

    # adjust column names to the simplified form (only keep columns whose values is not "")
    df = df.select(
        [
            col
            for col in df.columns
            if col in config.all_forms_cols_mapping
            and config.all_forms_cols_mapping[col] != ""
        ]
    )
    updated_dict = {col: config.all_forms_cols_mapping[col] for col in df.columns}
    df = df.rename(updated_dict)
    df_list.append(df)

# 4. Save the updated dataset
# compare columns between each df to make sure they are the same
columns_pe = set(df_list[0].columns)
columns_pv = set(df_list[1].columns)
columns_mb = set(df_list[2].columns)
columns_uv = set(df_list[3].columns)

cols_in_pe_not_in_pv = columns_pe - columns_pv
cols_in_pe_not_in_mb = columns_pe - columns_mb
cols_in_pe_not_in_uv = columns_pe - columns_uv
cols_in_pv_not_in_mb = columns_pv - columns_mb
cols_in_pv_not_in_uv = columns_pv - columns_uv
cols_in_mb_not_in_uv = columns_mb - columns_uv

print(f"Columns in PE not in PV: {cols_in_pe_not_in_pv}")
print(f"Columns in PE not in MB: {cols_in_pe_not_in_mb}")
print(f"Columns in PE not in UV: {cols_in_pe_not_in_uv}")
print(f"Columns in PV not in MB: {cols_in_pv_not_in_mb}")
print(f"Columns in PV not in UV: {cols_in_pv_not_in_uv}")
print(f"Columns in MB not in UV: {cols_in_mb_not_in_uv}")

# concatenate and save
df = pl.concat(df_list, how="diagonal")
file_path = config.OUTPUT_PATH + "CONSOLIDATED_DB_with_infrastructure_id.xlsx"
df.write_excel(file_path)
print(f"Successfully processed {len(df)} records and saved to {file_path}.")

# create version of consolidated form for validation purposes.
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
final_ordered_cols = ordered_cols + missing_cols
df_validation = df_validation.select(final_ordered_cols)
df_validation = df_validation.rename(
    {col: config.name_label_mapping.get(col, col) for col in df_validation.columns}
)
file_path_validation = (
    config.OUTPUT_PATH + "CONSOLIDATED_DB_with_infrastructure_id_validation.xlsx"
)

with xlsxwriter.Workbook(file_path_validation) as wb:
    # Mali_HCDR
    df_hcdr = df_validation.filter(pl.col("8) Nature de l'indicateur") == 2)
    if len(df_hcdr) > 0:
        df_hcdr.write_excel(workbook=wb, worksheet="Mali_HCDR")
        print(f"Successfully processed {len(df_hcdr)} records for Mali_HCDR.")

    # Per-country sheets (CDR only)
    for country in df_validation["20) Pays"].unique():
        df_country = df_validation.filter(
            (pl.col("20) Pays") == country) & (pl.col("8) Nature de l'indicateur") == 1)
        )
        df_country.write_excel(workbook=wb, worksheet=country)
        print(f"Successfully processed {len(df_country)} records for {country}.")

print(f"Successfully saved all sheets to {file_path_validation}")
