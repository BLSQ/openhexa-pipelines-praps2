import logging
from itertools import combinations
from math import asin, cos, radians, sin, sqrt
from typing import Dict, List, Sequence, Tuple

import polars as pl


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


def reassign_ids(src_indexes: Sequence[int], duplicate_groups: Sequence[List[int]]) -> Dict[int, int]:
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
            if not _check_coords(row1, column_coords) or not _check_coords(row2, column_coords):
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


def ir_1(indicateurs_pays: pl.DataFrame) -> pl.DataFrame:
    """IR-1: Taux de couverture vaccinale PPCB."""
    rows = []

    df = indicateurs_pays.filter((pl.col("DATE9").is_null().not_()) & (pl.col("DATE8").is_null().not_()))

    for row in df.iter_rows(named=True):
        if row["DATE9"] > row["DATE8"]:
            logging.warning('IR-1: vaccinated count is higher than total' f' ({row["DATE4"]}, {row["DATE5"]})')

        rows.append(
            {
                "date": f"{row['DATE5']}-01-01",
                "level": 2,
                "country": row["DATE4"],
                "indicator_code": "IR-1",
                "value": round(row["DATE9"] / row["DATE8"], 3),
            }
        )

    logging.info(f"IR-1: computed {len(rows)} values")
    return pl.DataFrame(rows)


def ir_2(indicateurs_pays: pl.DataFrame) -> pl.DataFrame:
    """IR-2: Nombre de petits ruminants vaccinés et marqués contre la PPR

    Parameters
    ----------
    indicateurs_pays : dataframe
        FICHE_INDI_CDR_ PAYS_1KBT_4_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    rows = []

    src = indicateurs_pays.filter(pl.col("IR-2").is_not_null())

    for row in src.iter_rows(named=True):
        value = row["IR-2"]

        # value sometimes expressed in millions
        # todo: check with PRAPS team
        if value <= 500:
            value *= 1000000

        rows.append(
            {
                "date": f"{row['DATE5']}-01-01",
                "level": 2,
                "country": row["DATE4"],
                "indicator_code": "IR-2",
                "value": value,
            }
        )

    df = pl.DataFrame(rows)

    df = df.sort(by="date").with_columns(pl.col("value").cum_sum().alias("cumulated_value").over("country"))

    logging.info(f"IR-2: computed {len(df)} values")

    return df


def ir_3(paysages: pl.DataFrame) -> pl.DataFrame:
    """IR-3: Superficie de terres sous pratique de gestion durable des paysages

    Parameters
    ----------
    paysages : dataframe
        FICHE SUPERFICIE SOUS PRATIQUE DE GESTI...AYSAGES PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    rows = []

    for row in paysages.iter_rows(named=True):
        value = 0

        # 22a si 23 et 24 et 27 et 25 = oui
        if row.get("CRDURA6c"):
            if (
                row.get("CRDURA7") == "Oui"
                and row.get("CRDURA8") == "Oui"
                and row.get("CRDURA11") == "Oui"
                and row.get("CRDURA9")
            ):
                value += float(row["CRDURA6c"])

        # 42a si 40 = oui
        if row.get("CRDURA26c"):
            if row.get("CRDURA24") == "Oui":
                value += float(row["CRDURA26c"])

        # 58a si 56 = oui
        if row.get("CRDURA43c"):
            if row.get("CRDURA41") == "Oui":
                value += float(row["CRDURA43c"])

        # 73a si 71 = oui
        if row.get("CRDURA58c"):
            if row.get("CRDURA56") == "Oui":
                value += float(row["CRDURA58c"])

        rows.append(
            {
                "indicator_code": "IR-3",
                "date": row["DATE"],
                "level": 6,
                "country": row["LODURA1"],
                "region": row["LODURA2"],
                "province": row["LODURA3"],
                "commune": row["LODURA4"],
                "localite": row["LODURA5"],
                "coordinates": row["LODURA6"],
                "value": value,
            }
        )

    logging.info(f"IR-3: computed {len(rows)} values")
    df = pl.DataFrame(rows)

    df = df.sort(by="date").with_columns(pl.col("value").cum_sum().alias("cumulated_value").over("country"))

    return df


def ir_4(indicateurs_pays: pl.DataFrame) -> pl.DataFrame:
    """IR-4: Accroissement des revenus des ménages pastoraux générés par l’appui du projet

    Parameters
    ----------
    indicateurs_pays : dataframe
        FICHE_INDI_CDR_ PAYS_1KBT_4_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = indicateurs_pays.filter(pl.col("IR-4").is_not_null()).select(
        [
            pl.lit("IR-4").alias("indicator_code"),
            pl.format("{}-01-01", pl.col("DATE5")).alias("date"),
            pl.lit(2).alias("level"),
            pl.col("DATE4").alias("country"),
            pl.col("IR-4").alias("value"),
        ]
    )
    logging.info(f"IR-4: computed {len(df)} values")
    return df


def iri_1(indicateurs_pays: pl.DataFrame) -> pl.DataFrame:
    """IRI-1: Niveau de mise en œuvre des plans nationaux stratégiques pour la PPR et la PPCB

    Parameters
    ----------
    indicateurs_pays : dataframe
        FICHE_INDI_CDR_ PAYS_1KBT_4_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = indicateurs_pays.filter(pl.col("IRI-1").is_not_null()).select(
        [
            pl.lit("IRI-1").alias("indicator_code"),
            pl.format("{}-01-01", pl.col("DATE5")).alias("date"),
            pl.lit(2).alias("level"),
            pl.col("DATE4").alias("country"),
            pl.col("IRI-1").alias("value") / 100,
        ]
    )
    logging.info(f"IRI-1: computed {len(df)} values")
    return df


def iri_2(unites_veterinaires: pl.DataFrame) -> pl.DataFrame:
    """IRI-2: Unités vétérinaires construites ou réhabilitées par le projet et
    fonctionnelles dans les zones ciblées

    Parameters
    ----------
    unites_veterinaires : dataframe
        FICHE UNITE VETERINAIRE PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = unites_veterinaires.filter(
        (pl.col("STUV5") == "Réception provisoire sans réserve")
        | (
            (pl.col("STUV5") == "Réception définitive")
            & (pl.col("CFUV1") == "Oui")
            & (pl.col("CFUV2") == "Oui")
            & (pl.col("CFUV3") == "Oui")
            & (pl.col("CFUV4") == "Oui")
        )
    ).select(
        [
            pl.lit("IRI-2").alias("indicator_code"),
            pl.col("DATE").alias("date"),
            pl.lit(6).alias("level"),
            pl.col("LUV1").alias("country"),
            pl.col("LUV2").alias("region"),
            pl.col("LUV3").alias("province"),
            pl.col("LUV4").alias("commune"),
            pl.col("LUV5").alias("localite"),
            pl.col("LUV6").alias("coordinates"),
            pl.lit(1).alias("value"),
        ]
    )
    logging.info(f"IRI-2: computed {len(df)} values")
    return df


def iri_3(parcs_de_vaccination: pl.DataFrame) -> pl.DataFrame:
    """IRI-3: Parcs de vaccination construits ou réhabilités par le projet dans les zones ciblées

    Parameters
    ----------
    parcs_de_vaccination : dataframe
        FICHE PARC DE VACCINATION PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = parcs_de_vaccination.filter(
        (pl.col("STVAC5") == "Réception provisoire sans réserve")
        | ((pl.col("STVAC5") == "Réception définitive") & (pl.col("IGVAC1") == "Oui") & (pl.col("IGVAC2") == "Oui"))
    ).select(
        [
            pl.lit("IRI-3").alias("indicator_code"),
            pl.col("DATE").alias("date"),
            pl.lit(6).alias("level"),
            pl.col("LVAC1").alias("country"),
            pl.col("LVAC2").alias("region"),
            pl.col("LVAC3").alias("province"),
            pl.col("LVAC4").alias("commune"),
            pl.col("LVAC5").alias("localite"),
            pl.col("LVAC6").alias("coordinates"),
            pl.lit(1).alias("value"),
        ]
    )
    logging.info(f"IRI-3: computed {len(df)} values")
    return df


def iri_5(paysages: pl.DataFrame) -> pl.DataFrame:
    """IRI-5: Comités fonctionnels pour la gestion durable des territoires facilitant la
    mobilité mis en place ou appuyés par le projet

    Parameters
    ----------
    paysages : dataframe
        FICHE SUPERFICIE SOUS PRATIQUE DE GESTI...AYSAGES PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = paysages.filter(pl.col("CRDURA11") == "Oui").select(
        [
            pl.lit("IRI-5").alias("indicator_code"),
            pl.format("{}-01-01", "CRDURA12").alias("date"),
            pl.lit(6).alias("level"),
            pl.col("LODURA1").alias("country"),
            pl.col("LODURA2").alias("region"),
            pl.col("LODURA3").alias("province"),
            pl.col("LODURA4").alias("commune"),
            pl.col("LODURA5").alias("localite"),
            pl.col("LODURA6").alias("coordinates"),
            pl.lit(1).alias("value"),
        ]
    )
    logging.info(f"IRI-5: computed {len(df)} values")
    return df


def iri_6(points_d_eau: pl.DataFrame) -> pl.DataFrame:
    """IRI-6: Point d'eau fonctionnels accessibles aux (agro) pasteurs sur les axes de
    déplacement et sur les nouveaux parcours de transhumance appuyés par le projet

    Parameters
    ----------
    points_d_eau : dataframe
        FICHE POINT D_EAU PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = points_d_eau.filter(
        (pl.col("STPE5") == "Réception provisoire sans réserve")
        | (
            (pl.col("STPE5") == "Réception définitive")
            & (pl.col("LPE6") == "Oui")
            & (pl.col("CFPE2") == "Oui")
            & (pl.col("CFPE4") == "Oui")
            & (pl.col("CFPE6") == "Oui")
        )
    ).select(
        [
            pl.lit("IRI-6").alias("indicator_code"),
            pl.col("DATE").alias("date"),
            pl.lit(6).alias("level"),
            pl.col("LPE1").alias("country"),
            pl.col("LPE2").alias("region"),
            pl.col("LPE3").alias("province"),
            pl.col("LPE4").alias("commune"),
            pl.col("LPE5").alias("localite"),
            pl.col("LPE7").alias("coordinates"),
            pl.lit(1).alias("value"),
        ]
    )
    logging.info(f"IRI-6: computed {len(df)} values")
    return df


def iri_8(marches: pl.DataFrame) -> pl.DataFrame:
    """IRI-8: Marchés opérationnels selon les critères définis réhabilités et
    construits grâce au projet sur les couloirs régionaux

    Parameters
    ----------
    marches : dataframe
        FICHE MARCHES A BETAIL PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = marches.filter(
        (pl.col("STMB5") == "Réception provisoire sans réserve")
        | (
            (pl.col("STMB5") == "Réception définitive")
            & (pl.col("CMOP1") == "Oui")
            & (pl.col("CMOP3") == "Oui")
            & (pl.col("CMOP4") == "Oui")
            & (pl.col("CMOP6") == "Oui")
        )
    ).select(
        [
            pl.lit("IRI-8").alias("indicator_code"),
            pl.col("DATE").alias("date"),
            pl.lit(6).alias("level"),
            pl.col("LMB1").alias("country"),
            pl.col("LMB2").alias("region"),
            pl.col("LMB3").alias("province"),
            pl.col("LMB4").alias("commune"),
            pl.col("LMB5").alias("localite"),
            pl.col("LMB6").alias("coordinates"),
            pl.lit(1).alias("value"),
        ]
    )
    logging.info(f"IRI-8: computed {len(df)} values")
    return df


def iri_9(indicateurs_pays: pl.DataFrame) -> pl.DataFrame:
    """IRI-9: Taux d'exécution des plans d'actions élaborés par les organisations pastorales
    faîtières (part appuyée par le projet)

    Parameters
    ----------
    indicateurs_pays : dataframe
        FICHE_INDI_CDR_ PAYS_1KBT_4_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = indicateurs_pays.filter(pl.col("IRI-9").is_not_null()).select(
        [
            pl.lit("IRI-9").alias("indicator_code"),
            pl.format("{}-01-01", "DATE5").alias("date"),
            pl.lit(2).alias("level"),
            pl.col("DATE4").alias("country"),
            pl.col("IRI-9").alias("value") / 100,
        ]
    )
    logging.info(f"IRI-9: computed {len(df)} values")
    return df


def iri_10(projects: pl.DataFrame) -> pl.DataFrame:
    """IRI-10: Bénéficiaires des sous-projets innovants de valorisation des filières pastorales
    promus par le projet

    Parameters
    ----------
    projects : dataframe
        FICHE SOUS PROJETS INNOVANTS PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = (
        projects.with_columns((pl.col("VAINO6").fill_null(0) + pl.col("VAINO13").fill_null(0)).alias("value"))
        .select(
            [
                pl.lit("IRI-10").alias("indicator_code"),
                pl.col("DATE").alias("date"),
                pl.lit(6).alias("level"),
                pl.col("LINO1").alias("country"),
                pl.col("LINO2").alias("region"),
                pl.col("LINO3").alias("province"),
                pl.col("LINO4").alias("commune"),
                pl.col("LINO5").alias("localite"),
                pl.col("LINO6").alias("coordinates"),
                pl.col("value"),
            ]
        )
        .filter(pl.col("value") > 0)
    )
    logging.info(f"IRI-10: computed {len(df)} values")
    return df


def iri_101(projects: pl.DataFrame) -> pl.DataFrame:
    """IRI-101: Dont jeunes 18-24 ans

    Parameters
    ----------
    projects : dataframe
        FICHE SOUS PROJETS INNOVANTS PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = (
        projects.with_columns(
            (
                pl.col("VAINO9").fill_null(0) + pl.col("VAINO11").fill_null(0) + pl.col("VAINO15") + pl.col("VAINO17")
            ).alias("value")
        )
        .select(
            [
                pl.lit("IRI-101").alias("indicator_code"),
                pl.col("DATE").alias("date"),
                pl.lit(6).alias("level"),
                pl.col("LINO1").alias("country"),
                pl.col("LINO2").alias("region"),
                pl.col("LINO3").alias("province"),
                pl.col("LINO4").alias("commune"),
                pl.col("LINO5").alias("localite"),
                pl.col("LINO6").alias("coordinates"),
                pl.col("value"),
            ]
        )
        .filter(pl.col("value") > 0)
    )
    logging.info(f"IRI-101: computed {len(df)} values")
    return df


def iri_102(projects: pl.DataFrame) -> pl.DataFrame:
    """IRI-102: Dont jeunes 25-40 ans

    Parameters
    ----------
    projects : dataframe
        FICHE SOUS PROJETS INNOVANTS PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = (
        projects.with_columns(
            (
                pl.col("VAINO10").fill_null(0) + pl.col("VAINO12").fill_null(0) + pl.col("VAINO16") + pl.col("VAINO18")
            ).alias("value")
        )
        .select(
            [
                pl.lit("IRI-102").alias("indicator_code"),
                pl.col("DATE").alias("date"),
                pl.lit(6).alias("level"),
                pl.col("LINO1").alias("country"),
                pl.col("LINO2").alias("region"),
                pl.col("LINO3").alias("province"),
                pl.col("LINO4").alias("commune"),
                pl.col("LINO5").alias("localite"),
                pl.col("LINO6").alias("coordinates"),
                pl.col("value"),
            ]
        )
        .filter(pl.col("value") > 0)
    )
    logging.info(f"IRI-102: computed {len(df)} values")
    return df


def iri_103(projects: pl.DataFrame) -> pl.DataFrame:
    """IRI-103: Dont femmes

    Parameters
    ----------
    projects : dataframe
        FICHE SOUS PROJETS INNOVANTS PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = (
        projects.with_columns((pl.col("VAINO7").fill_null(0) + pl.col("VAINO14").fill_null(0)).alias("value"))
        .select(
            [
                pl.lit("IRI-103").alias("indicator_code"),
                pl.col("DATE").alias("date"),
                pl.lit(6).alias("level"),
                pl.col("LINO1").alias("country"),
                pl.col("LINO2").alias("region"),
                pl.col("LINO3").alias("province"),
                pl.col("LINO4").alias("commune"),
                pl.col("LINO5").alias("localite"),
                pl.col("LINO6").alias("coordinates"),
                pl.col("value"),
            ]
        )
        .filter(pl.col("value") > 0)
    )
    logging.info(f"IRI-103: computed {len(df)} values")
    return df


def iri_13(activites: pl.DataFrame) -> pl.DataFrame:
    """IRI-13: Bénéficiaires directs d'activités génératrices de revenus promues par le projet

    Parameters
    ----------
    activites : dataframe
        FICHE ACTIVITES GENERATRICES DE REVENUS (AGR) PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = activites.filter(pl.col("VAAGR6").is_not_null()).select(
        [
            pl.lit("IRI-13").alias("indicator_code"),
            pl.col("DATE").alias("date"),
            pl.lit(6).alias("level"),
            pl.col("LAGR1").alias("country"),
            pl.col("LAGR2").alias("region"),
            pl.col("LAGR3").alias("province"),
            pl.col("LAGR4").alias("commune"),
            pl.col("LAGR5").alias("localite"),
            pl.col("LAGR6").alias("coordinates"),
            pl.col("VAAGR6").alias("value"),
        ]
    )
    logging.info(f"IRI-13: computed {len(df)} values")
    return df


def iri_131(activites: pl.DataFrame) -> pl.DataFrame:
    """IRI-131: Dont jeunes 18-24 ans

    Parameters
    ----------
    activites : dataframe
        FICHE ACTIVITES GENERATRICES DE REVENUS (AGR) PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = (
        activites.with_columns((pl.col("VAAGR8").fill_null(0) + pl.col("VAAGR10").fill_null(0)).alias("value"))
        .select(
            [
                pl.lit("IRI-131").alias("indicator_code"),
                pl.col("DATE").alias("date"),
                pl.lit(6).alias("level"),
                pl.col("LAGR1").alias("country"),
                pl.col("LAGR2").alias("region"),
                pl.col("LAGR3").alias("province"),
                pl.col("LAGR4").alias("commune"),
                pl.col("LAGR5").alias("localite"),
                pl.col("LAGR6").alias("coordinates"),
                pl.col("value"),
            ]
        )
        .filter(pl.col("value") > 0)
    )
    logging.info(f"IRI-131: computed {len(df)} values")
    return df


def iri_132(activites: pl.DataFrame) -> pl.DataFrame:
    """IRI-132: Dont jeunes 25-40 ans

    Parameters
    ----------
    activites : dataframe
        FICHE ACTIVITES GENERATRICES DE REVENUS (AGR) PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = (
        activites.with_columns((pl.col("VAAGR9").fill_null(0) + pl.col("VAAGR11").fill_null(0)).alias("value"))
        .select(
            [
                pl.lit("IRI-132").alias("indicator_code"),
                pl.col("DATE").alias("date"),
                pl.lit(6).alias("level"),
                pl.col("LAGR1").alias("country"),
                pl.col("LAGR2").alias("region"),
                pl.col("LAGR3").alias("province"),
                pl.col("LAGR4").alias("commune"),
                pl.col("LAGR5").alias("localite"),
                pl.col("LAGR6").alias("coordinates"),
                pl.col("value"),
            ]
        )
        .filter(pl.col("value") > 0)
    )
    logging.info(f"IRI-132: computed {len(df)} values")
    return df


def iri_133(activites: pl.DataFrame) -> pl.DataFrame:
    """IRI-133: Dont femmes

    Parameters
    ----------
    activites : dataframe
        FICHE ACTIVITES GENERATRICES DE REVENUS (AGR) PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = activites.filter(pl.col("VAAGR7").is_not_null()).select(
        [
            pl.lit("IRI-133").alias("indicator_code"),
            pl.col("DATE").alias("date"),
            pl.lit(6).alias("level"),
            pl.col("LAGR1").alias("country"),
            pl.col("LAGR2").alias("region"),
            pl.col("LAGR3").alias("province"),
            pl.col("LAGR4").alias("commune"),
            pl.col("LAGR5").alias("localite"),
            pl.col("LAGR6").alias("coordinates"),
            pl.col("VAAGR6").alias("value"),
        ]
    )
    logging.info(f"IRI-133: computed {len(df)} values")
    return df


def iri_14(indicateurs_pays: pl.DataFrame) -> pl.DataFrame:
    """IRI-14: Cadres techniques et scientifiques formés sur le pastoralisme
    (dont formations diplômantes)


    Parameters
    ----------
    indicateurs_pays : dataframe
        FICHE_INDI_CDR_ PAYS_1KBT_4_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = indicateurs_pays.filter(pl.col("IRI-14").is_not_null()).select(
        [
            pl.lit("IRI-14").alias("indicator_code"),
            pl.format("{}-01-01", "DATE5").alias("date"),
            pl.lit(2).alias("level"),
            pl.col("DATE4").alias("country"),
            pl.col("IRI-14").alias("value"),
        ]
    )
    logging.info(f"IRI-14: computed {len(df)} values")
    return df


def iri_141(indicateurs_pays: pl.DataFrame) -> pl.DataFrame:
    """IRI-141: Dont femmes

    Parameters
    ----------
    indicateurs_pays : dataframe
        FICHE_INDI_CDR_ PAYS_1KBT_4_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = indicateurs_pays.filter(pl.col("IRI-14-1").is_not_null()).select(
        [
            pl.lit("IRI-141").alias("indicator_code"),
            pl.format("{}-01-01", "DATE5").alias("date"),
            pl.lit(2).alias("level"),
            pl.col("DATE4").alias("country"),
            pl.col("IRI-14-1").alias("value"),
        ]
    )
    logging.info(f"IRI-141: computed {len(df)} values")
    return df


def iri_15(indicateurs_pays: pl.DataFrame) -> pl.DataFrame:
    """IRI-15: Paramètres spécifiques ou pastoralisme pérennisé dans le système
    d'alerte précoce national

    Parameters
    ----------
    indicateurs_pays : dataframe
        FICHE_INDI_CDR_ PAYS_1KBT_4_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = indicateurs_pays.filter(pl.col("IRI-15").is_not_null()).select(
        [
            pl.lit("IRI-15").alias("indicator_code"),
            pl.format("{}-01-01", "DATE5").alias("date"),
            pl.lit(2).alias("level"),
            pl.col("DATE4").alias("country"),
            pl.col("IRI-15").alias("value") == "Oui",
        ]
    )
    logging.info(f"IRI-15: computed {len(df)} values")
    return df


def iri_16(
    parcs_de_vaccination: pl.DataFrame,
    gestion_durable: pl.DataFrame,
    points_d_eau: pl.DataFrame,
    marches_a_betail: pl.DataFrame,
) -> pl.DataFrame:
    """IRI-16: Comité de gestion ayant au moins 15% de femmes participant activement

    Parameters
    ----------
    parcs_de_vaccination : dataframe
        FICHE PARC DE VACCINATION PRAPS 2_14072023 _ KoboToolbox
    gestion_durable : dataframe
        FICHE SUPERFICIE SOUS PRATIQUE DE GESTI...AYSAGES PRAPS 2_14072023 _ KoboToolbox
    points_d_eau : dataframe
        FICHE POINT D_EAU PRAPS 2_14072023 _ KoboToolbox
    marches_a_betail : dataframe
        FICHE MARCHES A BETAIL PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """

    if "IGVAC12A" in parcs_de_vaccination.columns:
        df1 = (
            parcs_de_vaccination.filter(pl.col("IGVAC9") == "Oui")
            .with_columns(
                pl.when(
                    (pl.col("IGVAC11") > 0)
                    & (pl.col("IGVAC12") >= (pl.col("IGVAC11") * 0.15))
                    & (
                        (pl.col("IGVAC12A").list.contains("Président (e)"))
                        | (pl.col("IGVAC12A").list.contains("Sécrétaire (principal-e)"))
                        | (pl.col("IGVAC12A").list.contains("Trésorier (ère)"))
                    )
                )
                .then(1)
                .otherwise(0)
                .alias("councils with women")
            )
            .select(
                [
                    pl.lit("IRI-16").alias("indicator_code"),
                    pl.col("DATE").alias("date"),
                    pl.lit(6).alias("level"),
                    pl.col("LVAC1").alias("country"),
                    pl.col("LVAC2").alias("region"),
                    pl.col("LVAC3").alias("province"),
                    pl.col("LVAC4").alias("commune"),
                    pl.col("LVAC5").alias("localite"),
                    pl.col("LVAC6").alias("coordinates"),
                    pl.col("councils with women").alias("numerator"),
                    pl.lit(1).alias("denominator"),
                    pl.col("councils with women").alias("value"),
                ]
            )
        )

    else:
        df1 = None

    if "CRDURA17" in gestion_durable.columns:
        df2 = (
            gestion_durable.filter(pl.col("CRDURA11") == "Oui")
            .with_columns(
                pl.when(
                    (pl.col("CRDURA13") > 0)
                    & (pl.col("CRDURA14") >= (pl.col("CRDURA13") * 0.15))
                    & (
                        (pl.col("CRDURA17").list.contains("Président (e)"))
                        | (pl.col("CRDURA17").list.contains("Sécrétaire (principal-e)"))
                        | (pl.col("CRDURA17").list.contains("Trésorier (ère)"))
                    )
                )
                .then(1)
                .otherwise(0)
                .alias("councils with women")
            )
            .select(
                [
                    pl.lit("IRI-16").alias("indicator_code"),
                    pl.col("DATE").alias("date"),
                    pl.lit(6).alias("level"),
                    pl.col("LODURA1").alias("country"),
                    pl.col("LODURA2").alias("region"),
                    pl.col("LODURA3").alias("province"),
                    pl.col("LODURA4").alias("commune"),
                    pl.col("LODURA5").alias("localite"),
                    pl.col("LODURA6").alias("coordinates"),
                    pl.col("councils with women").alias("numerator"),
                    pl.lit(1).alias("denominator"),
                    pl.col("councils with women").alias("value"),
                ]
            )
        )

    else:
        df2 = None

    if "IGPE11A3" in points_d_eau.columns:
        df3 = (
            points_d_eau.filter(pl.col("IGPE6") == "Oui")
            .with_columns(
                pl.when(
                    (pl.col("IGPE10") > 0)
                    & (pl.col("IGPE11") >= (pl.col("IGPE10") * 0.15))
                    & (
                        (pl.col("IGPE11A3").list.contains("Président (e)"))
                        | (pl.col("IGPE11A3").list.contains("Sécrétaire (principal-e)"))
                        | (pl.col("IGPE11A3").list.contains("Trésorier (ère)"))
                    )
                )
                .then(1)
                .otherwise(0)
                .alias("councils with women")
            )
            .select(
                [
                    pl.lit("IRI-16").alias("indicator_code"),
                    pl.col("DATE").alias("date"),
                    pl.lit(6).alias("level"),
                    pl.col("LPE1").alias("country"),
                    pl.col("LPE2").alias("region"),
                    pl.col("LPE3").alias("province"),
                    pl.col("LPE4").alias("commune"),
                    pl.col("LPE5").alias("localite"),
                    pl.col("LPE7").alias("coordinates"),
                    pl.col("councils with women").alias("numerator"),
                    pl.lit(1).alias("denominator"),
                    pl.col("councils with women").alias("value"),
                ]
            )
        )

    else:
        df3 = None

    if "IGMBA" in marches_a_betail.columns:
        df4 = (
            marches_a_betail.filter(pl.col("IGMB5") == "Oui")
            .with_columns(
                pl.when(
                    (pl.col("IGMB7") > 0)
                    & (pl.col("IGMB8") >= (pl.col("IGMB7") * 0.15))
                    & (
                        (pl.col("IGMBA").list.contains("Président (e)"))
                        | (pl.col("IGMBA").list.contains("Sécrétaire (principal-e)"))
                        | (pl.col("IGMBA").list.contains("Trésorier (ère)"))
                    )
                )
                .then(1)
                .otherwise(0)
                .alias("councils with women")
            )
            .select(
                [
                    pl.lit("IRI-16").alias("indicator_code"),
                    pl.col("DATE").alias("date"),
                    pl.lit(6).alias("level"),
                    pl.col("LMB1").alias("country"),
                    pl.col("LMB2").alias("region"),
                    pl.col("LMB3").alias("province"),
                    pl.col("LMB4").alias("commune"),
                    pl.col("LMB5").alias("localite"),
                    pl.col("LMB6").alias("coordinates"),
                    pl.col("councils with women").alias("numerator"),
                    pl.lit(1).alias("denominator"),
                    pl.col("councils with women").alias("value"),
                ]
            )
        )

    else:
        df4 = None

    dataframes = [df for df in [df1, df2, df3, df4] if df is not None]
    df = pl.concat(dataframes)
    logging.info(f"IRI-16: computed {len(df)} values")
    return df


def iri_17(
    sous_projets: pl.DataFrame,
    activites: pl.DataFrame,
) -> pl.DataFrame:
    """IRI-17: Femmes ayant reçu des formations en gestion financière (%)

    Parameters
    ----------
    sous_projets : dataframe
        FICHE SOUS PROJETS INNOVANTS PRAPS 2_14072023 _ KoboToolbox
    activites : dataframe
        FICHE ACTIVITES GENERATRICES DE REVENUS (AGR) PRAPS 2_14072023 _ KoboToolbox

    Return
    ------
    dataframe
    """
    df1 = sous_projets.filter(pl.col("VAINO7") > 0).select(
        [
            pl.lit("IRI-17").alias("indicator_code"),
            pl.col("DATE").alias("date"),
            pl.lit(6).alias("level"),
            pl.col("LINO1").alias("country"),
            pl.col("LINO2").alias("region"),
            pl.col("LINO3").alias("province"),
            pl.col("LINO4").alias("commune"),
            pl.col("LINO5").alias("localite"),
            pl.col("LINO6").alias("coordinates"),
            pl.col("VAINO7").alias("denominator"),
            pl.col("VAINO8").fill_null(0).alias("numerator"),
            ((pl.col("VAINO8").fill_null(0) / pl.col("VAINO7")).round(3)).alias("value"),
        ]
    )

    df2 = activites.filter(pl.col("VAAGR7") > 0).select(
        [
            pl.lit("IRI-17").alias("indicator_code"),
            pl.col("DATE").alias("date"),
            pl.lit(6).alias("level"),
            pl.col("LAGR1").alias("country"),
            pl.col("LAGR2").alias("region"),
            pl.col("LAGR3").alias("province"),
            pl.col("LAGR4").alias("commune"),
            pl.col("LAGR5").alias("localite"),
            pl.col("LAGR6").alias("coordinates"),
            pl.col("VAAGR7").alias("denominator"),
            pl.col("VAAGR7A").fill_null(0).alias("numerator"),
            ((pl.col("VAAGR7A").fill_null(0) / pl.col("VAAGR7")).round(3)).alias("value"),
        ]
    )

    df = pl.concat([df1, df2])
    df = df.filter(pl.col("numerator") <= pl.col("denominator"))
    logging.info(f"IRI-17: computed {len(df)} values")
    return df


def iri_18(indicateurs_pays: pl.DataFrame) -> pl.DataFrame:
    """IRI-18: Agriculteurs ayant bénéficié d'actifs ou services agricoles

    Parameters
    ----------
    indicateurs_pays : dataframe
        FICHE_INDI_CDR_ PAYS_1KBT_4_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = indicateurs_pays.filter(pl.col("IRI-18").is_not_null()).select(
        [
            pl.lit("IRI-18").alias("indicator_code"),
            pl.format("{}-01-01", "DATE5").alias("date"),
            pl.lit(2).alias("level"),
            pl.col("DATE4").alias("country"),
            pl.col("IRI-18").alias("value"),
        ]
    )
    logging.info(f"IRI-18: computed {len(df)} values")
    return df


def iri_181(indicateurs_pays: pl.DataFrame) -> pl.DataFrame:
    """IRI-181: Dont femmes (30%)

    Parameters
    ----------
    indicateurs_pays : dataframe
        FICHE_INDI_CDR_ PAYS_1KBT_4_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    df = indicateurs_pays.filter(pl.col("IRI-18-1").is_not_null()).select(
        [
            pl.lit("IRI-181").alias("indicator_code"),
            pl.format("{}-01-01", "DATE5").alias("date"),
            pl.lit(2).alias("level"),
            pl.col("DATE4").alias("country"),
            pl.col("IRI-18-1").alias("value"),
        ]
    )
    logging.info(f"IRI-181: computed {len(df)} values")
    return df


def reg_int_1(indicateurs_regionaux: pl.DataFrame) -> pl.DataFrame:
    """Reg-Int-1: Comité Vétérinaire Régional opérationnel

    Parameters
    ----------
    indicateurs_regionaux : dataframe
        FICHE_ CDR INDI_REGIONAL_KBT 14_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    if "Reg-Int-1" not in indicateurs_regionaux.columns:
        return pl.DataFrame()

    df = indicateurs_regionaux.filter(pl.col("Reg-Int-1").is_not_null()).select(
        [
            pl.lit("Reg Int 1").alias("indicator_code"),
            pl.format("{}-01-01", "IND5").alias("date"),
            pl.lit(1).alias("level"),
            pl.lit("Régional").alias("country"),
            pl.col("Reg-Int-1").alias("value"),
        ]
    )
    logging.info(f"Reg Int 1: computed {len(df)} values")
    return df


def reg_int_2(indicateurs_regionaux: pl.DataFrame) -> pl.DataFrame:
    """Reg-Int-2: Accords bilatéraux et multilatéraux facilitant une transhumance
    pacifique établis grâce au projet

    Parameters
    ----------
    indicateurs_regionaux : dataframe
        FICHE_ CDR INDI_REGIONAL_KBT 14_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    if "Reg-Int-2" not in indicateurs_regionaux.columns:
        return pl.DataFrame()

    df = indicateurs_regionaux.filter(pl.col("Reg-Int-2").is_not_null()).select(
        [
            pl.lit("Reg Int 2").alias("indicator_code"),
            pl.format("{}-01-01", "IND5").alias("date"),
            pl.lit(1).alias("level"),
            pl.lit("Régional").alias("country"),
            pl.col("Reg-Int-2").alias("value"),
        ]
    )
    logging.info(f"Reg Int 2: computed {len(df)} values")
    return df


def reg_int_4(indicateurs_regionaux: pl.DataFrame) -> pl.DataFrame:
    """Reg-Int-4: Barrières commerciales suivies dans des zones de commercialisation
    transfrontalière sélectionnées et diffusées par le projet

    Parameters
    ----------
    indicateurs_regionaux : dataframe
        FICHE_ CDR INDI_REGIONAL_KBT 14_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    if "Reg-Int-4" not in indicateurs_regionaux.columns:
        return pl.DataFrame()

    df = indicateurs_regionaux.filter(pl.col("Reg-Int-4").is_not_null()).select(
        [
            pl.lit("Reg Int 4").alias("indicator_code"),
            pl.format("{}-01-01", "IND5").alias("date"),
            pl.lit(1).alias("level"),
            pl.lit("Régional").alias("country"),
            pl.col("Reg-Int-4").alias("value"),
        ]
    )
    logging.info(f"Reg Int 4: computed {len(df)} values")
    return df


def reg_int_5(indicateurs_regionaux: pl.DataFrame) -> pl.DataFrame:
    """Reg-Int-5: Capacités institutionnelles nationales et régionales renforcées pour
    élaborer des politiques et stratégies d'élevage tenant compte du climat

    Parameters
    ----------
    indicateurs_regionaux : dataframe
        FICHE_ CDR INDI_REGIONAL_KBT 14_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    if "Reg-Int-5" not in indicateurs_regionaux.columns:
        return pl.DataFrame()

    df = indicateurs_regionaux.filter(pl.col("Reg-Int-5").is_not_null()).select(
        [
            pl.lit("Reg Int 5").alias("indicator_code"),
            pl.format("{}-01-01", "IND5").alias("date"),
            pl.lit(1).alias("level"),
            pl.lit("Régional").alias("country"),
            pl.col("Reg-Int-5").alias("value"),
        ]
    )
    logging.info(f"Reg Int 5: computed {len(df)} values")
    return df


def reg_int_6(indicateurs_regionaux: pl.DataFrame) -> pl.DataFrame:
    """Reg-Int-6: Capacité régionale renforcée pour mener des analyses prospectives sur
    le secteur de l'élevage

    Parameters
    ----------
    indicateurs_regionaux : dataframe
        FICHE_ CDR INDI_REGIONAL_KBT 14_07_23_PRAPS_2_VF _ KoboToolbox

    Return
    ------
    dataframe
    """
    if "Reg-Int-6" not in indicateurs_regionaux.columns:
        return pl.DataFrame()

    df = indicateurs_regionaux.filter(pl.col("Reg-Int-6").is_not_null()).select(
        [
            pl.lit("Reg Int 6").alias("indicator_code"),
            pl.format("{}-01-01", "IND5").alias("date"),
            pl.lit(1).alias("level"),
            pl.lit("Régional").alias("country"),
            pl.col("Reg-Int-6").alias("value"),
        ]
    )
    logging.info(f"Reg Int 6: computed {len(df)} values")
    return df


def load_praps1_data(fname: str) -> pl.DataFrame:
    """Load and transform PRAPS1 data from CSV."""
    COUNTRIES = {
        "BF": "Burkina-Faso",
        "MR": "Mauritanie",
        "SN": "Sénégal",
        "ML": "Mali",
        "REGIONAL": "Régional",
        "NE": "Niger",
        "TD": "Tchad",
    }

    df = pl.read_csv(fname).select(
        [
            pl.col("Code").alias("indicator_code"),
            pl.format("{}-01-01", pl.col("année")).alias("date"),
            pl.when(pl.col("Pays") == "REGIONAL").then(1).otherwise(2).alias("level"),
            pl.col("Pays").replace(COUNTRIES).alias("country"),
            pl.lit("PRAPS1").alias("project"),
            pl.col("valeur").alias("value"),
        ]
    )

    # percent values between 0 and 1
    df = df.with_columns(
        pl.when(pl.col("indicator_code").is_in(["IR-1", "IRI-17", "IRI-1", "IRI-9", "Reg Int 7"]))
        .then(pl.col("value") / 100)
        .otherwise(pl.col("value"))
        .alias("value")
    )

    return df


def combine_indicators(
    indicateurs_regionaux: pl.DataFrame,
    indicateurs_pays: pl.DataFrame,
    gestion_durable: pl.DataFrame,
    unites_veterinaires: pl.DataFrame,
    parcs_de_vaccination: pl.DataFrame,
    points_d_eau: pl.DataFrame,
    marches_a_betail: pl.DataFrame,
    sous_projets: pl.DataFrame,
    activites: pl.DataFrame,
    praps1: pl.DataFrame,
) -> pl.DataFrame:
    """Compute PRAPS2 indicators from survey data and concatenate PRAPS1 indicator values."""
    dataframes = [
        ir_1(indicateurs_pays),
        ir_2(indicateurs_pays),
        ir_3(gestion_durable),
        ir_4(indicateurs_pays),
        iri_1(indicateurs_pays),
        iri_2(unites_veterinaires),
        iri_3(parcs_de_vaccination),
        iri_5(gestion_durable),
        iri_6(points_d_eau),
        iri_8(marches_a_betail),
        iri_9(indicateurs_pays),
        iri_10(sous_projets),
        iri_101(sous_projets),
        iri_102(sous_projets),
        iri_103(sous_projets),
        iri_13(activites),
        iri_131(activites),
        iri_132(activites),
        iri_133(activites),
        iri_14(indicateurs_pays),
        iri_141(indicateurs_pays),
        iri_15(indicateurs_pays),
        iri_16(parcs_de_vaccination, gestion_durable, points_d_eau, marches_a_betail),
        iri_17(sous_projets, activites),
        iri_18(indicateurs_pays),
        iri_181(indicateurs_pays),
        reg_int_1(indicateurs_regionaux),
        reg_int_2(indicateurs_regionaux),
        reg_int_4(indicateurs_regionaux),
        reg_int_5(indicateurs_regionaux),
        reg_int_6(indicateurs_regionaux),
    ]

    praps2 = pl.concat([df for df in dataframes if not df.is_empty()], how="diagonal_relaxed")
    praps2 = praps2.with_columns(pl.lit("PRAPS2").alias("project"))
    df = pl.concat([praps1, praps2], how="diagonal_relaxed")

    return df


def join_metadata(df: pl.DataFrame, indicators_metadata: pl.DataFrame) -> pl.DataFrame:
    """Join indicators metadata (unit, full name, etc) to dataframe."""
    df = df.join(other=indicators_metadata, how="left", left_on="indicator_code", right_on="code").select(
        [
            pl.col("indicator_code"),
            pl.col("designation").alias("indicator_name"),
            pl.col("unite").alias("unit"),
            pl.col("date").str.slice(0, 4).cast(int),
            pl.col("project"),
            pl.col("level"),
            pl.col("country"),
            pl.col("region"),
            pl.col("province"),
            pl.col("commune"),
            pl.col("localite"),
            pl.col("coordinates"),
            pl.col("numerator"),
            pl.col("denominator"),
            pl.col("value"),
        ]
    )

    return df


def aggregate_counts(df: pl.DataFrame, agg_columns: Sequence[str]) -> pl.DataFrame:
    return (
        df.filter(pl.col("unit").is_in(["count", "weight", "surface"]))
        .group_by(agg_columns)
        .agg(pl.col("value").sum())
    )


def aggregate_ratios(df: pl.DataFrame, agg_columns: Sequence[str]) -> pl.DataFrame:
    return (
        df.filter(
            (pl.col("unit") == "percent") & (pl.col("numerator").is_not_null()) & (pl.col("denominator").is_not_null())
        )
        .group_by(agg_columns)
        .agg([pl.col("numerator").sum(), pl.col("denominator").sum()])
        .with_columns((pl.col("numerator") / pl.col("denominator")).alias("value"))
    )


def aggregate_ratios_without_num_den(df: pl.DataFrame, agg_columns: Sequence[str]) -> pl.DataFrame:
    return (
        df.filter((pl.col("unit") == "percent") & ((pl.col("numerator").is_null()) | (pl.col("denominator").is_null())))
        .group_by(agg_columns)
        .agg(pl.col("value").mean())
    )


def aggregate_bools(df: pl.DataFrame, agg_columns: Sequence[str]) -> pl.DataFrame:
    return (
        df.filter(pl.col("unit") == "boolean")
        .group_by(agg_columns)
        .agg(pl.col("value").cast(pl.Boolean).all().cast(pl.Float64))
    )


def spatial_aggregation(df: pl.DataFrame) -> pl.DataFrame:
    """Apply spatial aggregation for each geographic level."""
    # from lvl 6 (localité) to lvl 3 (region)
    AGG_COLUMNS = [
        "indicator_code",
        "indicator_name",
        "unit",
        "date",
        "project",
        "level",
        "country",
        "region",
    ]
    df_lvl6 = df.filter(pl.col("level") == 6)
    counts = aggregate_counts(df=df_lvl6, agg_columns=AGG_COLUMNS)
    ratios = aggregate_ratios(df=df_lvl6, agg_columns=AGG_COLUMNS)
    ratios_ = aggregate_ratios_without_num_den(df=df_lvl6, agg_columns=AGG_COLUMNS)
    bools = aggregate_bools(df=df_lvl6, agg_columns=AGG_COLUMNS)
    per_region = pl.concat([counts, ratios, ratios_, bools], how="diagonal_relaxed").with_columns(
        pl.lit(3).alias("level")
    )
    df = pl.concat([df, per_region], how="diagonal_relaxed")

    # from lvl3 (region) to lvl2 (country)
    agg_columns = [c for c in AGG_COLUMNS if c != "region"]
    df_lvl3 = df.filter(pl.col("level") == 3)
    counts = aggregate_counts(df=df_lvl3, agg_columns=agg_columns)
    ratios = aggregate_ratios(df=df_lvl3, agg_columns=agg_columns)
    ratios_ = aggregate_ratios_without_num_den(df=df_lvl3, agg_columns=agg_columns)
    bools = aggregate_bools(df=df_lvl3, agg_columns=agg_columns)
    per_country = pl.concat([counts, ratios, ratios_, bools], how="diagonal_relaxed").with_columns(
        pl.lit(2).alias("level")
    )
    df = pl.concat([df, per_country], how="diagonal_relaxed")

    # from lvl2 (country) to lvl1 (region)
    agg_columns = [c for c in AGG_COLUMNS if c not in ("region", "country")]
    df_lvl2 = df.filter(pl.col("level") == 2)
    counts = aggregate_counts(df=df_lvl2, agg_columns=agg_columns)
    ratios = aggregate_ratios(df=df_lvl2, agg_columns=agg_columns)
    ratios_ = aggregate_ratios_without_num_den(df=df_lvl2, agg_columns=agg_columns)
    bools = aggregate_bools(df=df_lvl2, agg_columns=agg_columns)
    regional = pl.concat([counts, ratios, ratios_, bools], how="diagonal_relaxed").with_columns(
        [pl.lit(1).alias("level"), pl.lit("Régional").alias("country")]
    )
    df = pl.concat([df, regional], how="diagonal_relaxed")

    # combine all geographic levels into a single dataframe
    df = (
        df.filter(pl.col("level") <= 3)
        .unique(
            subset=["indicator_code", "date", "level", "country", "region"],
            keep="first",
            maintain_order=True,
        )
        .select(
            [
                "indicator_code",
                "indicator_name",
                "unit",
                "date",
                "project",
                "level",
                "country",
                "region",
                "numerator",
                "denominator",
                "value",
            ]
        )
        .sort(["indicator_code", "date", "country", "region"])
    )

    return df


def fill_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    """Fill missing values in indicators dataframe."""
    YEARS = list(range(2021, df["date"].max() + 1))
    COUNTRIES = list(df.filter(pl.col("level") == 2)["country"].unique())

    rows = []

    for code in df["indicator_code"].unique():
        df_ = df.filter(pl.col("indicator_code") == code)
        name = df_["indicator_name"][0]
        unit = df_["unit"][0]

        for year in YEARS:
            if year <= 2021:
                project = "PRAPS1"
            else:
                project = "PRAPS2"

            baserow = {
                "indicator_code": code,
                "indicator_name": name,
                "unit": unit,
                "date": year,
                "project": project,
                "level": None,
                "country": None,
                "region": None,
                "numerator": None,
                "denominator": None,
                "value": None,
            }

            row = baserow.copy()
            row["level"] = 1
            row["country"] = "Régional"
            rows.append(row)

            for country in COUNTRIES:
                if df_["level"].max() >= 2:
                    row = baserow.copy()
                    row["level"] = 2
                    row["country"] = country
                    rows.append(row)

                if df_["level"].max() >= 3:
                    for region in df_.filter(pl.col("country") == country)["region"].unique():
                        if not region:
                            continue

                        row = baserow.copy()
                        row["level"] = 3
                        row["country"] = country
                        row["region"] = region
                        rows.append(row)

    df_nulls = pl.DataFrame(data=rows, schema=df.schema)
    df = pl.concat([df, df_nulls], how="diagonal_relaxed").unique(
        subset=["indicator_code", "date", "country", "region"],
        keep="first",
        maintain_order=True,
    )
    return df


def cumulate_counts(df: pl.DataFrame) -> pl.DataFrame:
    """Add a `cumulated_value_praps2` column with cumulative sum of counts."""
    counts = pl.concat(
        [
            df.filter(pl.col("unit").is_in(["surface", "weight", "count"]))
            .filter(pl.col("level") == level)
            .sort(by="date")
            .with_columns(
                [
                    pl.col("value")
                    .fill_null(0)
                    .cum_sum()
                    .over(["indicator_code", "country", "region"])
                    .alias("cumulated_value"),
                    pl.when(pl.col("project") == "PRAPS2")
                    .then(pl.col("value"))
                    .otherwise(None)
                    .fill_null(0)
                    .cum_sum()
                    .over(["indicator_code", "country", "region"])
                    .alias("cumulated_value_praps2"),
                ]
            )
            for level in [1, 2, 3]
        ]
    )

    return counts.with_columns(
        pl.when(pl.col("project") == "PRAPS1")
        .then(None)
        .otherwise(pl.col("cumulated_value_praps2"))
        .alias("cumulated_value_praps2")
    )


def cumulate_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """Cumulate numerators and denominators to add a new column with cumulated values for ratios.

    Add 4 columns:
        * `cumulated_numerator` (cumulative sum)
        * `cumulated_denominator` (cumulative sum)
        * `cumulated_value` and `cumulated_value_praps2` (ratio)
    """
    ratios = pl.concat(
        [
            df.filter((pl.col("unit") == "percent") & (pl.col("project") == "PRAPS2") & (pl.col("level") == level))
            .sort(by="date")
            .with_columns(
                [
                    pl.col("numerator")
                    .fill_null(0)
                    .cum_sum()
                    .over(["indicator_code", "country", "region"])
                    .alias("cumulated_numerator"),
                    pl.col("denominator")
                    .fill_null(0)
                    .cum_sum()
                    .over(["indicator_code", "country", "region"])
                    .alias("cumulated_denominator"),
                ]
            )
            for level in [1, 2, 3]
        ]
    )

    ratios = pl.concat(
        [
            ratios,
            df.filter((pl.col("unit") == "percent") & (pl.col("project") == "PRAPS1")),
        ],
        how="diagonal_relaxed",
    )

    return ratios.with_columns(
        [
            (pl.col("cumulated_numerator") / pl.col("cumulated_denominator")).alias("cumulated_value"),
            (pl.col("cumulated_numerator") / pl.col("cumulated_denominator")).alias("cumulated_value_praps2"),
        ]
    )


def cumulate_bools(df: pl.DataFrame) -> pl.DataFrame:
    """Cumulate boolean indicator values.

    As of now, the function just add two columns `cumulated_value` and `cumulated_value_praps2`
    which uses the original value without any calculation.
    """
    return df.filter(pl.col("unit") == "boolean").with_columns(
        [
            pl.col("value").alias("cumulated_value"),
            pl.col("value").alias("cumulated_value_praps2"),
        ]
    )


def cumulate_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Cumulate indicator values over years."""
    counts = cumulate_counts(df)
    ratios = cumulate_ratios(df)
    bools = cumulate_bools(df)
    cumul = pl.concat([counts, ratios, bools], how="diagonal_relaxed")
    return cumul.sort(by=["indicator_code", "country", "region", "date"])


def retro_compatibility(df: pl.DataFrame) -> pl.DataFrame:
    """Rename columns and transform values for retro-compatibility with the Power BI
    dashboard.
    """
    return df.select(
        [
            pl.col("indicator_code"),
            pl.col("indicator_name"),
            pl.col("unit"),
            pl.col("date").alias("year"),
            pl.format("{}-01-01", pl.col("date")).alias("date"),
            pl.col("project").alias("project"),
            pl.col("level").alias("level"),
            pl.col("country"),
            pl.col("region"),
            pl.col("numerator"),
            pl.col("denominator"),
            pl.col("value"),
            pl.col("cumulated_numerator").alias("cumulative_numerator"),
            pl.col("cumulated_denominator").alias("cumulative_denominator"),
            pl.col("cumulated_value").alias("cumulative_value"),
            pl.col("cumulated_value_praps2").alias("cumulative_value_praps2"),
        ]
    )
