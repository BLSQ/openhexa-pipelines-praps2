import polars as pl
from math import asin, cos, radians, sin, sqrt
from typing import Dict, List, Sequence, Tuple
from itertools import combinations


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
    min_distance: float = 1.0,
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
                pairs.append((row1["INFRASTRUCTURE_ID"], row2["INFRASTRUCTURE_ID"]))

    if not len(pairs):
        return df

    groups = group_pairs(pairs)
    mapping = reassign_ids(df["INFRASTRUCTURE_ID"], groups)
    df = df.with_columns(pl.col("INFRASTRUCTURE_ID").replace(mapping))
    df = df.drop("column_localite")  # drop the temporary column

    return df
