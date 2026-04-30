import polars as pl


def normalize_indicator_column(column_name: str) -> pl.Expr:
    return (
        pl.col(column_name)
        .str.strip_chars()
        .str.to_lowercase()
        # 1. Remove Accents
        .str.replace_all(r"[éèêë]", "e")
        .str.replace_all(r"[àâä]", "a")
        .str.replace_all(r"[îï]", "i")
        .str.replace_all(r"[ôö]", "o")
        .str.replace_all(r"[ùûü]", "u")
        .str.replace_all(r"ç", "c")
        # 2. Protection: Keep singular words that end in S/X
        .str.replace_all(r"\bplus\b", "PLUS_TMP")
        .str.replace_all(r"\bmoins\b", "MOINS_TMP")
        .str.replace_all(r"\bprix\b", "PRIX_TMP")
        .str.replace_all(r"\bsous\b", "SOUS_TMP")
        .str.replace_all(r"\btaux\b", "TAUX_TMP")
        # 3. Singularize
        .str.replace_all(r"aux\b", "al")
        .str.replace_all(r"s\b", "")
        .str.replace_all(r"x\b", "")
        # 4. Restore Protected Words
        .str.replace_all("PLUS_TMP", "plus")
        .str.replace_all("MOINS_TMP", "moins")
        .str.replace_all("PRIX_TMP", "prix")
        .str.replace_all("SOUS_TMP", "sous")
        .str.replace_all("TAUX_TMP", "taux")
        # 5. Remove linking words
        .str.replace_all(r"\b(de|du|des|et|le|la|les|pour|au|ou)\b", " ")
        # 6. Final Special Character Cleanup
        .str.replace_all(r"[^\w\s]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        # 7. Restrict to first 4 words
        # This regex captures (Word + space) three times + the 4th word.
        # Everything after is discarded via the capture group $1.
        .str.replace_all(r"^((?:\w+\s*){1,4}).*", r"$1")
        .str.strip_chars()
    )
