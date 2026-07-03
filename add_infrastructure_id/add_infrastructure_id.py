import polars as pl
import config

# 1. Load the Excel file
df = pl.read_excel(config.INPUT_FILE)

# 2. Extract and format the Latitude and Longitude
# Split the space-separated GPS string into a list

# compute string latitude/longitude rounded to nearest 2 decimals without "." or "-"
lat_str = (df["_LPE7_latitude"].round(2)).cast(pl.String).str.replace_all(r"[\.-]", "")
lon_str = (df["_LPE7_longitude"].round(2)).cast(pl.String).str.replace_all(r"[\.-]", "")

# 3. Build the final INFRASTRUCTURE_ID column
df = df.with_columns(
    pl.when(pl.col("LPE7").is_not_null() & (pl.col("LPE7") != ""))
    .then(
        pl.concat_str(
            [
                pl.lit("PE_"),
                pl.col("LPE3").cast(
                    pl.String
                ),  # Cast to string in case LPE3 is purely numeric
                pl.lit("_"),
                lat_str,
                lon_str,
            ]
        )
    )
    .otherwise(pl.lit(None))  # Leaves the ID blank if there is no GPS data
    .alias("INFRASTRUCTURE_ID")
)

# create geometry columns
df = df.with_columns(
    pl.col("LPE7").alias("geometry"),
    pl.col("_LPE7_latitude").alias("latitude"),
    pl.col("_LPE7_longitude").alias("longitude"),
)

# Add extra cols
df = df.with_columns(
    pl.lit(1).alias("TYPE"),  # CDR
)

for col_name, value_code in config.infra_type_mapping.items():
    if col_name in df.columns:
        df = df.with_columns(
            pl.lit(value_code).alias("CDR"),
            pl.lit(config.infra_acronym_mapping[col_name]).alias("TYPE_ACRONYM"),
            pl.lit("").alias("HCDR"),
        )

# adjust column names to the simplified form (only keep columns whose values is not "")
df = df.select(
    [
        col
        for col in df.columns
        if col in config.mapping_dict_water_points
        and config.mapping_dict_water_points[col] != ""
    ]
)
updated_dict = {col: config.mapping_dict_water_points[col] for col in df.columns}
df = df.rename(updated_dict)

# 4. Save the updated dataset
df.write_excel(config.OUTPUT_FILE)
print(f"Successfully processed {len(df)} records and saved to {config.OUTPUT_FILE}.")
