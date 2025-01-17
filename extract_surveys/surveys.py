import json
from itertools import combinations
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import polars as pl
from openhexa.sdk import current_run
from openhexa.toolbox.kobo import Api
from openhexa.toolbox.kobo.utils import get_fields_mapping, to_dataframe

SURVEYS = [
    ("aHjhdBWbXzGf6d9bYm88Hq", "indicateurs_regionaux"),
    ("aCHSjHDuphcc2KjCgqAyxP", "indicateurs_pays"),
    ("a67gG6NTYHrxeEJkRDzk3q", "marches_a_betail"),
    ("a8VM8vXBA5vC2RvZ5g3MK4", "parcs_de_vaccination"),
    ("aQiqocfeJbxAd4zRehcXEP", "points_d_eau"),
    ("aRSr2SdzvpPEmnn6cJZXhK", "unites_veterinaires"),
    ("a59pdEM78L8WRtTTfpHGom", "fourrage_cultive"),
    ("a6gNzGeR2ScNZtYapFhTxJ", "sous_projets_innovants"),
    ("aEBHXt3rCHYxifEZLBMPU4", "gestion_durable_des_paysages"),
    ("aKjQ9Ak8rsPhtgxMup8mm6", "activites_generatrices_de_revenus"),
]


def download_survey_data(api: Api, uid: str, name: str, dst_file: Path) -> Path:
    """Download survey data from KoboToolbox."""
    survey = api.get_survey(uid)
    df = to_dataframe(survey)
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


def download_surveys(api_url: str, api_token: str, output_dir: str):
    """Download survey data from KoboToolbox."""
    api = Api(api_url)
    api.authenticate(api_token)

    for uid, fname in SURVEYS:
        survey = api.get_survey(uid)
        df = to_dataframe(survey)
        fpath = Path(output_dir, "raw", f"{fname}.parquet")
        df.write_parquet(fpath)

        mapping = get_fields_mapping(survey)
        mapping.write_parquet(Path(output_dir, "fields", f"{fname}_fields.parquet"))


PROGRESS = {
    "marches_a_betail": "STMB15",
    "parcs_de_vaccination": "STVAC15",
    "points_d_eau": "STPE15",
    "unites_veterinaires": "STUV15",
}

STATE = {
    "marches_a_betail": "STMB5",
    "parcs_de_vaccination": "STVAC5",
    "points_d_eau": "STPE5",
    "unites_veterinaires": "STUV5",
}

PICTURES = {
    "marches_a_betail": ["LMB7a", "LMB7b", "LMB7c", "LMB7d"],
    "parcs_de_vaccination": ["LVAC8a", "LVAC8b", "LVAC8c", "LVAC8d"],
    "points_d_eau": ["LPE8a", "LPE8b", "LPE8c", "LPE8d"],
    "unites_veterinaires": ["LUV7a", "LUV7b", "LUV7c", "LUV7d"],
    "fourrage_cultive": ["LFC7a", "LFC7b", "LFC7c", "LFC7d"],
    "gestion_durable_des_paysages": ["LODURA7a", "LODURA7b", "LODURA7c", "LODURA7d"],
    "activites_generatrices_de_revenus": ["LAGR7a", "LAGR7b", "LAGR7c", "LAGR7d"],
    "sous_projets_innovants": ["LINO7a", "LINO7b", "LINO7c", "LINO7d"],
}

GEO_COLUMNS = {
    "indicateurs_pays": {2: "DATE4"},
    "marches_a_betail": {
        2: "LMB1",
        3: "LMB2",
        4: "LMB3",
        5: "LMB4",
        6: "LMB5",
        7: "LMB6",
    },
    "parcs_de_vaccination": {
        2: "LVAC1",
        3: "LVAC2",
        4: "LVAC3",
        5: "LVAC4",
        6: "LVAC5",
        7: "LVAC6",
    },
    "points_d_eau": {2: "LPE1", 3: "LPE2", 4: "LPE3", 5: "LPE4", 6: "LPE5", 7: "LPE7"},
    "unites_veterinaires": {
        2: "LUV1",
        3: "LUV2",
        4: "LUV3",
        5: "LUV4",
        6: "LUV5",
        7: "LUV6",
    },
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
    column_localite: str = "localite",
    column_coords: str = "coordinates",
    min_distance: float = 1.0,
) -> pl.DataFrame:
    """Identify duplicate rows in source dataframe.

    Duplicates are identified based on localite and geographic coordinates.
    A unique ID will be assigned to each row (with duplicated unique ID
    for duplicated rows).

    Parameters
    ----------
    df : dataframe
        Input dataframe.
    column_localite : str
        Dataframe column with localité name.
    column_coords : str
        Dataframe column with coordinates.
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

    def _check_coords(row: dict, column_coords: str) -> bool:
        """Check availability of coordinates."""
        coords = row.get(column_coords)
        if coords:
            if coords.get("coordinates"):
                return True
        return False

    for localite in df[column_localite].unique():
        if not localite:
            continue
        df_ = df.filter(pl.col(column_localite) == localite)
        if len(df_) < 2:
            continue
        for row1, row2 in combinations(df_.iter_rows(named=True), 2):
            if not _check_coords(row1, column_coords) or not _check_coords(
                row2, column_coords
            ):
                continue
            lat1 = row1[column_coords]["coordinates"][0]
            lon1 = row1[column_coords]["coordinates"][1]
            lat2 = row2[column_coords]["coordinates"][0]
            lon2 = row2[column_coords]["coordinates"][1]
            distance = haversine(lat1, lon1, lat2, lon2)
            if distance <= min_distance:
                pairs.append((row1["infrastructure_id"], row2["infrastructure_id"]))

    if not len(pairs):
        return df

    groups = group_pairs(pairs)
    mapping = reassign_ids(df["infrastructure_id"], groups)
    df = df.with_columns(pl.col("infrastructure_id").replace(mapping))

    return df


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

    # rename geographic columns with standard names
    if name in GEO_COLUMNS:
        df = df.with_columns(
            [
                pl.col(field).alias(f"level_{lvl}")
                for lvl, field in GEO_COLUMNS[name].items()
            ]
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
    columns = PICTURES.get(name)
    if columns:
        for col in columns:
            df = df.with_columns(
                pl.col(col).map_elements(
                    _add_url_prefix, skip_nulls=False, return_dtype=pl.String
                )
            )

    if name in ("indicateurs_regionaux", "indicateurs_pays"):
        return df

    # identify unique infrastructures
    df = identify_duplicates(
        df=df.with_columns(pl.col("level_7").str.json_decode()),
        column_localite="level_6",
        column_coords="level_7",
        min_distance=1,
    )
    df = df.with_columns(pl.col("level_7").struct.json_encode())

    return df


def concatenate_snapshots(
    df: pl.DataFrame, column_unique_id: str = "infrastructure_id"
) -> pl.DataFrame:
    """Create a dataframe that concatenate yearly snapshots of mapped infrastructures."""
    snapshots = []
    for year in range(df["DATE"].min().year, df["DATE"].max().year + 1):
        snapshots.append(
            df.filter(pl.col("DATE").dt.year() <= year)
            .sort(by="DATE")
            .unique(subset=[column_unique_id], keep="last")
            .with_columns(pl.lit(year).alias("over_year"))
        )

    return pl.concat(snapshots)
