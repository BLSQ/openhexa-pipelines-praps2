from pathlib import Path

import polars as pl
import xlsxwriter
from openhexa.sdk import current_run, parameter, pipeline, workspace


@pipeline("generate-excel-cdr", name="generate-excel-cdr")
@parameter(
    "targets_fp",
    name="Cibles CDR",
    help="Fichier contenant les cibles CDR",
    type=str,
    default="data/targets/CDR_Targets.csv",
)
@parameter(
    "cdr_dir",
    name="Dossier CDR",
    help="Répertoire où le CDR est enregistré",
    type=str,
    default="data/cdr",
)
@parameter(
    "dst_file",
    name="Fichier de sortie",
    help="Fichier Excel de sortie",
    type=str,
    default="data/cdr/praps_cadre_de_resultat.xlsx",
)
def generate_excel_cdr(targets_fp: str, cdr_dir: str, dst_file: str):
    targets_fp = Path(workspace.files_path, targets_fp)
    cdr_dir = Path(workspace.files_path, cdr_dir)
    dst_file = Path(workspace.files_path, dst_file)
    generate(targets_fp=targets_fp, cdr_dir=cdr_dir, dst_file=dst_file)


COUNTRIES = {
    "Burkina-Faso": "BF",
    "Mali": "ML",
    "Mauritanie": "MR",
    "Niger": "NE",
    "Sénégal": "SN",
    "Tchad": "TD",
    "Régional": "REGIONAL",
}

SECTIONS = [
    {
        "name": "ODP",
        "label": "Indicateurs ODP par objectif/résultat",
        "components": [
            {"name": "Actifs soutenus et maintenus", "indicators": ["IR-1", "IR-2"]},
            {"name": "Ecosystèmes soutenus et maintenus", "indicators": ["IR-3"]},
            {"name": "Moyens soutenus et maintenus", "indicators": ["IR-4"]},
        ],
    },
    {
        "name": "IRI",
        "label": "Indicateurs de résultats intermédiaires par composante",
        "components": [
            {
                "name": "Amélioration de la santé animale et contrôle des médicaments vétérinaires",
                "indicators": ["IRI-1", "IRI-2", "IRI-3", "IRI-4", "Reg Int 1"],
            },
            {
                "name": "Gestion durable des paysages et amélioration de la gouvernance",
                "indicators": [
                    "IRI-5",
                    "IRI-6",
                    "IRI-7",
                    "IRI-FA",
                    "Reg Int 2",
                    "Reg Int 3",
                ],
            },
            {
                "name": "Amélioration des chaînes de valeur du bétail",
                "indicators": [
                    "IRI-8",
                    "IRI-9",
                    "IRI-10",
                    "IRI-101",
                    "IRI-102",
                    "IRI-103",
                    "Reg Int 4",
                ],
            },
            {
                "name": "Amélioration de l'inclusion sociale et économique, femmes et jeunes",
                "indicators": ["IRI-11", "IRI-111", "IRI-112", "IRI-113", "IRI-12"],
            },
            {
                "name": "Amélioration de l'inclusion sociale et économique, femmes et jeunes (suite)",
                "indicators": ["IRI-13", "IRI-131", "IRI-132", "IRI-133"],
            },
            {
                "name": "Coordination de projet, renforcement institutionnel, et prévention et réponse aux urgences",
                "indicators": ["IRI-14", "IRI-141", "IRI-15", "IRI-16", "IRI-17"],
            },
            {
                "name": "Coordination de projet, renforcement institutionnel, et prévention et réponse aux urgences (suite)",
                "indicators": [
                    "IRI-18",
                    "IRI-181",
                    "IRI-19",
                    "Reg Int 5",
                    "Reg Int 6",
                    "Reg Int 7",
                ],
            },
        ],
    },
]


def get_target(
    df: pl.DataFrame, indicator: str, country: str, year: int
) -> int | float:
    """Get indicator target for a given country and year."""
    try:
        row = df.row(
            by_predicate=(pl.col("Code") == indicator)
            & (pl.col("Pays") == country)
            & (pl.col("année") == year),
            named=True,
        )
    except pl.exceptions.NoRowsReturnedError:
        return None
    return row.get("valeur")


def get_value(df: pl.DataFrame, indicator: str, country: str, year: int) -> int | float:
    """Get indicator value for a given country and year."""
    # country name from 2-letters code
    mapping = {v: k for k, v in COUNTRIES.items()}
    country = mapping[country]

    try:
        row = df.row(
            by_predicate=(pl.col("indicator_code") == indicator)
            & (pl.col("country") == country)
            & (pl.col("year") == year),
            named=True,
        )
        cum_value = row.get("cumulative_value")
        value = row.get("value")
        if cum_value is not None and not indicator.startswith("Reg"):
            return cum_value
        else:
            if isinstance(value, float):
                return round(value, 1)
            return value
    except pl.exceptions.NoRowsReturnedError:
        return None


def generate(targets_fp: Path, cdr_dir: Path, dst_file: Path):
    targets = pl.read_csv(targets_fp)
    current_run.log_info(f"Loaded {len(targets)} targets")
    indicators = pl.read_parquet(cdr_dir / "indicateurs.parquet")
    indicators = indicators.fill_nan(None).filter(pl.col("level") <= 2)
    current_run.log_info(f"Loaded {len(indicators)} indicators")
    meta = pl.read_csv(cdr_dir / "indicators_metadata.csv")

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(dst_file.absolute().as_posix())
    sheet = workbook.add_worksheet("Cadre de résultats")

    default_fmt = workbook.add_format(
        {"text_wrap": 1, "align": "left", "valign": "vcenter"}
    )

    country_fmt = workbook.add_format(
        {"text_wrap": 1, "valign": "vcenter", "align": "left"}
    )

    value_fmt = workbook.add_format(
        {"font_size": 10, "align": "center", "valign": "vcenter"}
    )

    target_fmt = workbook.add_format(
        {
            "font_size": 10,
            "align": "center",
            "valign": "vcenter",
            "font_color": "#9e9e9e",
        }
    )

    section_header_fmt = workbook.add_format(
        {
            "bold": 1,
            "fg_color": "#fff9c4",
            "text_wrap": 1,
            "align": "left",
            "valign": "vcenter",
        }
    )

    section_header_year_fmt = workbook.add_format(
        {
            "bold": 1,
            "fg_color": "#fff9c4",
            "text_wrap": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )

    component_fmt = workbook.add_format(
        {
            "bold": 1,
            "fg_color": "#bbdefb",
            "text_wrap": 1,
            "align": "left",
            "valign": "vcenter",
        }
    )

    indicator_fmt = workbook.add_format(
        {"bold": 1, "text_wrap": 1, "align": "left", "valign": "vcenter"}
    )

    ratio_fmt = workbook.add_format(
        {"align": "center", "valign": "vcenter", "font_size": 10, "num_format": 9}
    )

    row = 0
    col = 0
    for section in SECTIONS:
        col = 0

        # section headers
        sheet.merge_range(row, col, row + 1, col, section["name"], section_header_fmt)
        sheet.merge_range(
            row, col + 1, row + 1, col + 1, section["label"], section_header_fmt
        )
        sheet.merge_range(
            row, col + 2, row + 1, col + 2, "Indicateur corporate", section_header_fmt
        )
        sheet.merge_range(
            row, col + 3, row + 1, col + 3, "Unité de mesure", section_header_fmt
        )
        sheet.merge_range(row, col + 4, row + 1, col + 4, "Pays", section_header_fmt)

        # section headers: years
        sheet.merge_range(row, 5, row, 5 + 1, "2021", section_header_year_fmt)
        sheet.write(row + 1, 5, "Cible", section_header_year_fmt)
        sheet.write(row + 1, 6, "Valeur", section_header_year_fmt)

        sheet.merge_range(row, 7, row, 7 + 2, "2022", section_header_year_fmt)
        sheet.write(row + 1, 7, "Cible", section_header_year_fmt)
        sheet.write(row + 1, 8, "Valeur", section_header_year_fmt)
        sheet.write(row + 1, 9, "%", section_header_year_fmt)

        sheet.merge_range(row, 10, row, 10 + 2, "2023", section_header_year_fmt)
        sheet.write(row + 1, 10, "Cible", section_header_year_fmt)
        sheet.write(row + 1, 11, "Valeur", section_header_year_fmt)
        sheet.write(row + 1, 12, "%", section_header_year_fmt)

        sheet.merge_range(row, 13, row, 13 + 2, "2024", section_header_year_fmt)
        sheet.write(row + 1, 13, "Cible", section_header_year_fmt)
        sheet.write(row + 1, 14, "Valeur", section_header_year_fmt)
        sheet.write(row + 1, 15, "%", section_header_year_fmt)

        sheet.merge_range(row, 16, row, 16 + 2, "2025", section_header_year_fmt)
        sheet.write(row + 1, 16, "Cible", section_header_year_fmt)
        sheet.write(row + 1, 17, "Valeur", section_header_year_fmt)
        sheet.write(row + 1, 18, "%", section_header_year_fmt)

        sheet.merge_range(row, 19, row, 19 + 2, "2026", section_header_year_fmt)
        sheet.write(row + 1, 19, "Cible", section_header_year_fmt)
        sheet.write(row + 1, 20, "Valeur", section_header_year_fmt)
        sheet.write(row + 1, 21, "%", section_header_year_fmt)

        sheet.merge_range(row, 22, row, 22 + 2, "2027", section_header_year_fmt)
        sheet.write(row + 1, 22, "Cible", section_header_year_fmt)
        sheet.write(row + 1, 23, "Valeur", section_header_year_fmt)
        sheet.write(row + 1, 24, "%", section_header_year_fmt)

        row += 2

        for component in section["components"]:
            col = 0
            sheet.merge_range(row, col, row, col + 29, component["name"], component_fmt)
            row += 1

            for indicator in component["indicators"]:
                col = 0

                if indicator.startswith("Reg"):
                    nrows = 1  # regional only
                    sheet.write(row, 0, indicator, indicator_fmt)
                else:
                    nrows = 7  # all countries + regional
                    sheet.merge_range(
                        row, col, row + nrows - 1, col, indicator, indicator_fmt
                    )

                col += 1

                try:
                    label = meta.row(
                        by_predicate=pl.col("code") == indicator, named=True
                    )["designation"]
                except pl.exceptions.NoRowsReturnedError:
                    label = None

                try:
                    unit = targets.filter(pl.col("Code") == indicator)["unite"][0]
                except IndexError:
                    unit = None

                for text in [label, "", unit]:
                    if nrows > 1:
                        sheet.merge_range(
                            row, col, row + nrows - 1, col, text, default_fmt
                        )
                    else:
                        sheet.write(row, col, text, default_fmt)
                    col += 1

                col_ = col
                for country_name, country_code in COUNTRIES.items():
                    if indicator.startswith("Reg") and country_name != "Régional":
                        continue

                    sheet.write(row, col, country_name, country_fmt)
                    col += 1

                    for year in range(2021, 2028):
                        value = get_value(indicators, indicator, country_code, year)
                        target = get_target(targets, indicator, country_code, year)

                        if unit == "Pourcentage" and value is not None:
                            value *= 100

                        sheet.write(row, col, target, target_fmt)
                        col += 1
                        sheet.write(row, col, value, value_fmt)

                        if year >= 2022:
                            col += 1
                            if value is not None and target:
                                ratio = round(value / target, 1)
                            else:
                                ratio = None
                            sheet.write(row, col, ratio, ratio_fmt)

                        col += 1

                    col = col_
                    row += 1

                label = None
                unit = None

        row += 1

    sheet.set_column(0, 0, 15)
    sheet.set_column(1, 1, 50)
    sheet.set_column(2, 2, 10)
    sheet.set_column(3, 3, 20)
    sheet.set_column(4, 4, 30)
    sheet.set_column(5, 17, 10)

    workbook.close()

    current_run.log_info(f"Saved {dst_file.name}")
    current_run.add_file_output(dst_file.absolute().as_posix())


if __name__ == "__main__":
    generate_excel_cdr()
