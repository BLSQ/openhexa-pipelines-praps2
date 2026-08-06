import json
from itertools import combinations
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import polars as pl
from openhexa.sdk import current_run
from openhexa.toolbox.kobo import Api
from openhexa.toolbox.kobo.utils import get_fields_mapping, to_dataframe

from legacy_schema import COLUMN_RENAMES, LEGACY_COLUMNS

OUTPUT_TABLES = [
    "marches_a_betail",
    "parcs_de_vaccination",
    "points_d_eau",
    "unites_veterinaires",
    "infrastructures_hors_cdr",
]

# CDR code/label pairs used to split the consolidated survey; matched on either since
# to_dataframe() may return the raw choice code or its french label (see CLAUDE.md §3.2).
CDR_INFRA_LIST = {
    "marches_a_betail": ("1", "Marché à bétail"),
    "parcs_de_vaccination": ("2", "Parc de vaccination"),
    "points_d_eau": ("3", "Point d'eau"),
    "unites_veterinaires": ("4", "Unité vétérinaire"),
}

# TYPE_ACRONYM is a stable, accent-free cross-check for the CDR/HCDR-based split (§3.2).
TYPE_ACRONYM_BY_CDR_TABLE = {
    "marches_a_betail": "MB",
    "parcs_de_vaccination": "PV",
    "points_d_eau": "PE",
    "unites_veterinaires": "UV",
}
HCDR_ACRONYMS = {
    "AA",
    "CCML",
    "EB",
    "MAB",
    "CB",
    "QED",
    "AQ",
    "UT",
    "CTSZC",
    "RBPF",
    "SAS",
    "CCP",
    "AUTRE",
}


def download_survey_data(api: Api, uid: str, name: str, dst_file: Path) -> Path:
    """Download survey data from KoboToolbox."""
    survey = api.get_survey(uid)
    df = to_dataframe(survey)

    # Some columns may have empty structs (e.g. _validation_status when all values are
    # empty) which cannot be written to Parquet. Replace with null String columns.
    empty_struct_cols = [col for col in df.columns if df[col].dtype == pl.Struct([])]
    if empty_struct_cols:
        df = df.with_columns(
            pl.lit(None).cast(pl.String).alias(col) for col in empty_struct_cols
        )

    df.write_parquet(dst_file)
    current_run.log_info(f"Downloaded {name} survey data ({len(df)} entries)")
    return dst_file


def download_survey_fields(api: Api, uid: str, name: str, dst_file: Path) -> Path:
    """Download survey fields metadata from KoboToolbox."""
    survey = api.get_survey(uid)
    mapping = get_fields_mapping(survey)
    mapping.write_parquet(dst_file)
    mapping.write_excel(Path(dst_file.as_posix().replace(".parquet", ".xlsx")))
    current_run.log_info(
        f"Downloaded {name} survey fields metadata ({len(mapping)} entries)"
    )
    return dst_file


PROGRESS = {
    "fiche_simplifiee_infrastructures": "STMB15",
}

STATE = {
    "fiche_simplifiee_infrastructures": "STMB5",
}

PICTURES = {
    "fourrage_cultive": ["LFC7a", "LFC7b", "LFC7c", "LFC7d"],
    "gestion_durable_des_paysages": ["LODURA7a", "LODURA7b", "LODURA7c", "LODURA7d"],
    "activites_generatrices_de_revenus": ["LAGR7a", "LAGR7b", "LAGR7c", "LAGR7d"],
    "sous_projets_innovants": ["LINO7a", "LINO7b", "LINO7c", "LINO7d"],
    "fiche_simplifiee_infrastructures": ["LUV7a", "LUV7b", "LUV7c", "LUV7d"],
}

GEO_COLUMNS = {
    "fourrage_cultive": {
        2: "LFC1",
        3: "LFC2",
        4: "LFC3",
        5: "LFC4",
        6: "LFC5",
        7: "LFC6",
    },
    "sous_projets_innovants": {
        2: "LINO1",
        3: "LINO2",
        4: "LINO3",
        5: "LINO4",
        6: "LINO5",
        7: "LINO6",
    },
    "gestion_durable_des_paysages": {
        2: "LODURA1",
        3: "LODURA2",
        4: "LODURA3",
        5: "LODURA4",
        6: "LODURA5",
        7: "LODURA6",
    },
    "activites_generatrices_de_revenus": {
        2: "LAGR1",
        3: "LAGR2",
        4: "LAGR3",
        5: "LAGR4",
        6: "LAGR5",
        7: "LAGR6",
    },
    "fiche_simplifiee_infrastructures": {
        2: "LUV1",
        3: "LUV2",
        4: "LUV3",
        5: "LUV4",
        6: "LUV5",
        7: "LUV6",
    },
}

# These legacy surveys never had a Kobo-native infrastructure identifier (unlike the
# consolidated survey, which computes INFRASTRUCTURE_ID itself). For those, assign one
# based on geographic proximity via identify_duplicates()
SURVEYS_WITH_GEO_ASSIGNED_ID = {
    "fourrage_cultive",
    "sous_projets_innovants",
    "gestion_durable_des_paysages",
    "activites_generatrices_de_revenus",
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance (in km) between two points."""
    R = 6372.8  # earth radius

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c


def group_pairs(pairs: Sequence[Tuple[int, int]]) -> Sequence[List[int]]:
    """Group pairs that intersect.

    Examples
    --------
    >>> group_pairs([(1, 3), (3, 2), (4, 6), (1, 2), (7, 8)])
    [[1, 2, 3], [4, 6], [7, 8]]
    """
    dup_flat = [i for d in pairs for i in d]
    dup_flat = list(sorted(set(dup_flat)))

    groups = []
    for i in dup_flat:
        group = [i]
        for pair in pairs:
            if i in pair:
                for j in pair:
                    if i != j:
                        group.append(j)
        group = sorted(group)
        if group not in groups:
            groups.append(group)

    return groups


def reassign_ids(
    src_indexes: Sequence[int], duplicate_groups: Sequence[List[int]]
) -> Dict[int, int]:
    """Re-assign unique IDs of duplicates to 1st of the group.

    Return a mapping.
    """
    mapping = {i: i for i in src_indexes}
    for group in duplicate_groups:
        for i in group:
            mapping[i] = group[0]
    return mapping


def identify_duplicates(
    df: pl.DataFrame,
    column_latitude: str = "LATITUDE",
    column_longitude: str = "LONGITUDE",
    min_distance: float = 1.5,
) -> pl.DataFrame:
    """Identify duplicate rows in source dataframe.

    Duplicates are identified based on geographic coordinates.
    A unique ID will be assigned to each row (with duplicated unique ID
    for duplicated rows).

    Parameters
    ----------
    df : dataframe
        Input dataframe.
    column_latitude : str
        Dataframe column with latitude values.
    column_longitude : str
        Dataframe column with longitude values.
    min_distance : float (default=1)
        Min. distance between two points for not being identified as
        duplicates (in kilometers).

    Return
    ------
    dataframe
        Output dataframe with duplicated dropped.
    """
    df = df.with_row_index(name="infrastructure_id")
    pairs = []

    def _check_coords(row: dict, column_latitude: str, column_longitude: str) -> bool:
        """Check availability of coordinates."""
        lat = row.get(column_latitude)
        lon = row.get(column_longitude)
        if lat is not None and lon is not None:
            return True
        return False

    df = df.with_columns(
        pl.concat_str(
            [
                pl.col(column_latitude).round(1).cast(pl.String),
                pl.col(column_longitude).round(1).cast(pl.String),
            ],
            separator="_",
        ).alias(
            "column_localite"
        )  # create a column with rounded coordinates to identify localities (2 decimals ~ 1.1km)
    )
    for localite in df["column_localite"].unique():
        if not localite:
            continue
        df_ = df.filter(pl.col("column_localite") == localite)
        if len(df_) < 2:
            continue
        for row1, row2 in combinations(df_.iter_rows(named=True), 2):
            if not _check_coords(
                row1, column_latitude, column_longitude
            ) or not _check_coords(row2, column_latitude, column_longitude):
                continue
            lat1 = row1[column_latitude]
            lon1 = row1[column_longitude]
            lat2 = row2[column_latitude]
            lon2 = row2[column_longitude]
            distance = haversine(lat1, lon1, lat2, lon2)
            if distance <= min_distance:
                pairs.append((row1["infrastructure_id"], row2["infrastructure_id"]))

    if not len(pairs):
        return df

    groups = group_pairs(pairs)
    mapping = reassign_ids(df["infrastructure_id"], groups)
    df = df.with_columns(pl.col("infrastructure_id").replace(mapping))

    return df


def drop_duplicates(
    df: pl.DataFrame,
    column_unique_id: str = "infrastructure_id",
    column_date: str = "DATE",
) -> pl.DataFrame:
    """Drop duplicate rows based on unique infrastructure ID and date.

    `column_unique_id` only exists for surveys that went through the (now-removed)
    identify_duplicates()/with_row_index() step at some point in their history -- the
    standalone legacy surveys (fourrage_cultive, sous_projets_innovants,
    gestion_durable_des_paysages, activites_generatrices_de_revenus) never did, so there
    is nothing to dedup on; skip rather than raise.
    """
    df = df.sort(column_date, descending=True)
    if column_unique_id not in df.columns:
        current_run.log_info(
            f"'{column_unique_id}' not in this survey's columns, skipping deduplication"
        )
        return df
    df = df.unique(subset=column_unique_id, keep="first")
    return df


def drop_blocked_submissions(df: pl.DataFrame) -> pl.DataFrame:
    """Drop blocked/partial submissions.

    When an enumerator types an IDT that already exists, the form skips ahead and the
    submission is saved with no identifier (and no location/works-tracking data). These
    are data-entry rejects, not real infrastructures, and must not reach any split.
    Must run before drop_duplicates: two blocked rows both have a null
    infrastructure_id, and `unique(subset=...)` treats null == null, so without this
    filter they would collapse into a single row.
    """
    if "infrastructure_id" not in df.columns:
        return df

    is_blocked = pl.col("infrastructure_id").is_null() | (
        pl.col("infrastructure_id").str.strip_chars() == ""
    )
    blocked = df.filter(is_blocked)
    if len(blocked):
        idt_values = (
            blocked["IDT"].drop_nulls().unique().to_list()
            if "IDT" in blocked.columns
            else []
        )
        current_run.log_warning(
            f"Dropping {len(blocked)} blocked/partial submission(s) with no infrastructure_id "
            f"(duplicate IDT values entered by enumerators: {idt_values})"
        )

    return df.filter(~is_blocked)


def serialize(value):
    """Serialize structs and lists columns into JSON."""
    if type(value) in (pl.Struct, dict):
        if value:
            return json.dumps(value, ensure_ascii=False)
        else:
            return None
    elif type(value) is pl.Series:
        value = value.to_list()
        if len(value):
            return json.dumps(value, ensure_ascii=False)
        else:
            return None
    elif type(value) is pl.List:
        if len(value):
            return json.dumps(value, ensure_ascii=False)
        else:
            return None
    else:
        raise ValueError(f"Cannot serialize object type {type(value)}")


def _add_url_prefix(fname: str) -> str:
    """Replace file names of attachments with public URLs."""
    if fname:
        return f"https://storage.googleapis.com/hexa-public-praps/{fname}"
    else:
        return "https://storage.googleapis.com/hexa-public-praps/placeholder.png"


def transform_survey(df: pl.DataFrame, name: str):
    # parse _validation_status and _geolocation columns
    df = df.with_columns(
        [
            pl.col("_validation_status")
            .map_elements(lambda x: x.get("label"), return_dtype=str)
            .alias("validation_status"),
            pl.col("_geolocation")
            .map_elements(lambda x: x[0] if all(x) else None, return_dtype=float)
            .alias("LATITUDE"),
            pl.col("_geolocation")
            .map_elements(lambda x: x[1] if all(x) else None, return_dtype=float)
            .alias("LONGITUDE"),
        ]
    )

    # Canonicalize the identifier column to lowercase "infrastructure_id" everywhere.
    if "INFRASTRUCTURE_ID" in df.columns:
        df = df.rename({"INFRASTRUCTURE_ID": "infrastructure_id"})

    # rename geographic columns with standard names
    if name in GEO_COLUMNS:
        available = []
        for lvl, field in GEO_COLUMNS[name].items():
            if field in df.columns:
                available.append((lvl, field))
            else:
                current_run.log_warning(
                    f"{name}: geographic field '{field}' (level_{lvl}) is absent from this "
                    "batch of submissions, skipping"
                )
        if available:
            df = df.with_columns(
                [pl.col(field).alias(f"level_{lvl}") for lvl, field in available]
            )

    # rename state and progress columns with consistent names
    if STATE.get(name) and STATE.get(name) in df.columns:
        df = df.with_columns(pl.col(STATE[name]).alias("STATE"))
    if PROGRESS.get(name) and PROGRESS.get(name) in df.columns:
        df = df.with_columns(pl.col(PROGRESS[name]).alias("PROGRESS"))

    # serialize struct and list columns
    df = df.with_columns(
        [
            pl.col(c).map_elements(serialize, return_dtype=pl.String)
            for c in df.columns
            if df[c].dtype in [pl.Struct, pl.List]
        ]
    )

    # replace picture urls with public ones
    # (skip fields absent from this batch, same reasoning as the GEO_COLUMNS guard above)
    columns = PICTURES.get(name)
    if columns:
        for col in columns:
            if col not in df.columns:
                current_run.log_warning(
                    f"{name}: picture field '{col}' is absent from this batch of "
                    "submissions, skipping"
                )
                continue
            df = df.with_columns(
                pl.col(col).map_elements(
                    _add_url_prefix, skip_nulls=False, return_dtype=pl.String
                )
            )

    # legacy surveys with no Kobo-native infrastructure identifier: assign one based on
    # geographic proximity (points within min_distance km share an ID). Not applied to
    # fiche_simplifiee_infrastructures -- INFRASTRUCTURE_ID is already computed inside
    # Kobo for that survey (renamed to infrastructure_id above).
    if name in SURVEYS_WITH_GEO_ASSIGNED_ID:
        df = identify_duplicates(
            df,
            column_latitude="LATITUDE",
            column_longitude="LONGITUDE",
            min_distance=1.5,
        )

    # drop duplicates based on unique infrastructure ID and date
    df_no_duplicates = drop_duplicates(
        df, column_unique_id="infrastructure_id", column_date="DATE"
    )

    return df, df_no_duplicates


def split_consolidated(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Split the consolidated survey dataframe into the 5 output tables.

    The 4 CDR-based tables are matched on `CDR` (code or label, per CLAUDE.md §3.2); the
    5th, `infrastructures_hors_cdr`, is every row where `HCDR` is non-empty. Returns all
    entries of OUTPUT_TABLES, including empty frames.
    """
    for col in ("CDR", "HCDR", "TYPE_ACRONYM", "_id"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.String).alias(col))

    checks = df.with_columns(
        [
            pl.col("CDR").str.strip_chars().is_in([code, label]).alias(name)
            for name, (code, label) in CDR_INFRA_LIST.items()
        ]
        + [
            (
                pl.col("HCDR").str.strip_chars().is_not_null()
                & (pl.col("HCDR").str.strip_chars() != "")
            ).alias("infrastructures_hors_cdr")
        ]
    )
    checks = checks.with_columns(pl.sum_horizontal(OUTPUT_TABLES).alias("_match_count"))

    no_split = checks.filter(pl.col("_match_count") == 0)
    if len(no_split):
        current_run.log_warning(
            f"{len(no_split)} submission(s) matched no output table (empty CDR and HCDR); "
            f"sample _id: {no_split['_id'].head(5).to_list()}"
        )

    multi_split = checks.filter(pl.col("_match_count") > 1)
    if len(multi_split):
        current_run.log_warning(
            f"{len(multi_split)} submission(s) matched more than one output table; "
            f"sample _id: {multi_split['_id'].head(5).to_list()}"
        )

    splits: Dict[str, pl.DataFrame] = {}
    for name in OUTPUT_TABLES:
        split_df = df.filter(checks[name])
        splits[name] = split_df
        current_run.log_info(f"split_consolidated: {name} -> {len(split_df)} entries")

    # cross-check TYPE_ACRONYM consistency
    for name, expected in TYPE_ACRONYM_BY_CDR_TABLE.items():
        mismatched = splits[name].filter(
            pl.col("TYPE_ACRONYM").is_not_null() & (pl.col("TYPE_ACRONYM") != expected)
        )
        if len(mismatched):
            current_run.log_warning(
                f"{name}: {len(mismatched)} row(s) have TYPE_ACRONYM != '{expected}' "
                f"(e.g. {mismatched['TYPE_ACRONYM'].unique().to_list()[:5]})"
            )

    hcdr_mismatched = splits["infrastructures_hors_cdr"].filter(
        pl.col("TYPE_ACRONYM").is_not_null()
        & ~pl.col("TYPE_ACRONYM").is_in(list(HCDR_ACRONYMS))
    )
    if len(hcdr_mismatched):
        current_run.log_warning(
            f"infrastructures_hors_cdr: {len(hcdr_mismatched)} row(s) have an unexpected "
            f"TYPE_ACRONYM (e.g. {hcdr_mismatched['TYPE_ACRONYM'].unique().to_list()[:5]})"
        )

    return splits


def conform_to_legacy_schema(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Rename, null-fill and reorder columns so `df` matches the pre-consolidation
    column shape of output table `name` (see legacy_schema.py).

    `name`s with no legacy predecessor (infrastructures_hors_cdr) only go through the
    rename step, if `COLUMN_RENAMES` defines one.
    """
    rename_map = {}
    for src, dst in COLUMN_RENAMES.get(name, {}).items():
        if src not in df.columns:
            continue
        if dst in df.columns:
            current_run.log_warning(
                f"{name}: rename target '{dst}' (from '{src}') already exists in the "
                "consolidated frame; keeping the existing column, skipping this rename"
            )
            continue
        rename_map[src] = dst
    if rename_map:
        df = df.rename(rename_map)

    legacy_cols = LEGACY_COLUMNS.get(name)
    if legacy_cols is None:
        return df

    filled = []
    for col, dtype in legacy_cols.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
            filled.append(col)
        elif df[col].dtype != dtype:
            try:
                df = df.with_columns(pl.col(col).cast(dtype))
            except Exception:
                current_run.log_warning(
                    f"{name}: column '{col}' has dtype {df[col].dtype}, expected legacy "
                    f"dtype {dtype}; keeping source dtype"
                )

    if filled:
        current_run.log_info(
            f"{name}: {len(filled)} legacy column(s) no longer collected, filled as null: "
            f"{filled}"
        )

    extra_cols = [c for c in df.columns if c not in legacy_cols]
    df = df.select(list(legacy_cols.keys()) + extra_cols)

    assert set(legacy_cols) <= set(df.columns)
    return df


def concatenate_snapshots(
    df: pl.DataFrame, column_unique_id: str = "infrastructure_id"
) -> pl.DataFrame:
    """Create a dataframe that concatenate yearly snapshots of mapped infrastructures.

    Falls back to Kobo's own "_id" (always present, always unique) when
    `column_unique_id` doesn't exist -- see drop_duplicates() for why that happens.
    Every submission is then its own entity (no cross-submission collapsing), which is
    the correct degradation when there is no real entity identity to reconcile against.
    """
    if column_unique_id not in df.columns:
        current_run.log_info(
            f"'{column_unique_id}' not in this survey's columns, using '_id' for snapshots"
        )
        column_unique_id = "_id"

    snapshots = []
    for year in range(df["DATE"].min().year, df["DATE"].max().year + 1):
        snapshots.append(
            df.filter(pl.col("DATE").dt.year() <= year)
            .sort(by="DATE")
            .unique(subset=[column_unique_id], keep="last")
            .with_columns(pl.lit(year).alias("over_year"))
        )

    return pl.concat(snapshots)
