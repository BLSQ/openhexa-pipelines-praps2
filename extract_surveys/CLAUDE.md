# CLAUDE.md — PRAPS `extract-surveys` pipeline

Instructions for Claude Code working on this repository.

---

## 1. Context

This is an [OpenHexa](https://github.com/BLSQ/openhexa) pipeline (`extract-surveys`) for the
**PRAPS-2** programme (World Bank / CILSS): it tracks pastoral infrastructures across 6 Sahel
countries. It downloads KoboToolbox survey submissions, transforms them, writes files to the
OpenHexa workspace, pushes tables to the OpenHexa PostgreSQL/PostGIS warehouse, and publishes
OpenHexa datasets.

Two files matter:

| File | Role |
| --- | --- |
| `pipeline.py` | OpenHexa pipeline definition + 4 tasks: `download`, `transform`, `push`, `update_datasets` |
| `surveys.py` | Helper library: survey download, per-survey constants, `transform_survey`, `drop_duplicates`, `concatenate_snapshots`, serialization helpers |

Downstream consumer that constrains everything below: a **Geonode server** renders each
infrastructure table on a digital web map using **hard-coded column names** in its layer
configuration. Renaming or dropping a column breaks the map. `push()` therefore also mirrors each
table into a legacy `PRAPS2_*` table that Geonode still reads.

---

## 2. Objective

Today the pipeline was designed around **11 separate Kobo surveys** (`SURVEYS` in both files, now
mostly commented out). It must instead consume **one consolidated Kobo survey** and *fan it back
out* into the table shapes downstream systems already expect.

**In:** one survey — `fiche_simplifiee_infrastructures`, uid `aRZ43wRerX8pCdo4XrKw4n`.
It merges what used to be 4 separate surveys, plus a 5th category of infrastructure that never had
its own survey.

**Out:** five sets of outputs (files + DB tables):

| Output name | Selected from consolidated survey by |
| --- | --- |
| `marches_a_betail` | `CDR` == `Marché à bétail` |
| `parcs_de_vaccination` | `CDR` == `Parc de vaccination` |
| `points_d_eau` | `CDR` == `Point d'eau` |
| `unites_veterinaires` | `CDR` == `Unité vétérinaire` |
| `infrastructures_hors_cdr` | `HCDR` is not empty (new table, no predecessor) |

**The hard requirement — "reconstruct":** each of the 4 pre-existing output tables must expose
**exactly the same column names it had before**, *even for columns that no longer exist in the
consolidated survey* (those become all-null columns of the right type), **plus** the new columns
that only exist in the consolidated survey. Extra columns are fine — Geonode ignores columns it
does not reference. Missing or renamed columns are a regression.

Note that "reconstruct" is **not** just adding null columns: the consolidated survey uses **one
field name per concept** where the 4 old surveys each used their own prefixed names (the location
block is `LUV1..LUV6` in the consolidated form, but was `LMB1..LMB6` in `marches_a_betail`,
`LPE1..LPE5/LPE7` in `points_d_eau`, `LVAC1..LVAC6` in `parcs_de_vaccination`; likewise `STMB*` vs
`STPE*` / `STVAC*` / `STUV*`). So the job per output table is: **rename** consolidated fields to that
survey's historical names, **add** the missing ones as typed nulls, and **keep** the consolidated-only
ones. §5 gives the two authoritative sources for both halves of that.

The 7 other historical surveys — `indicateurs_regionaux`, `indicateurs_pays`, `fourrage_cultive`,
`sous_projets_innovants`, `gestion_durable_des_paysages`, `activites_generatrices_de_revenus`,
`cultures_vivrieres` — are **out of scope**: do not download, transform, push, or publish them.
Remove the code paths that special-case them (see §7).

---

## 3. Decisions already taken (do not re-litigate)

1. **Legacy schemas are hard-coded** in `surveys.py` as an explicit per-table column → dtype
   mapping — *derived* from the two on-disk sources in §5, but committed as literal constants. No
   runtime database introspection and no reading of the reference parquet files at pipeline run time.
   Reading those sources is a **blocking prerequisite** (§5), not part of the pipeline.
2. **No `fiche_simplifiee_infrastructures` output.** The consolidated form is an intermediate only:
   no `fiche_simplifiee_infrastructures` DB table, no dataset, no `surveys/`/`geo/`/`snapshots/`
   artefact for it. Keeping the downloaded `raw/fiche_simplifiee_infrastructures.parquet` and
   `metadata/fiche_simplifiee_infrastructures_fields.*` is expected and required. **Superseded by
   §10.9: files + DB push were later added back for this survey too — dataset/Geonode mirror
   still excluded.**
3. **`infrastructures_hors_cdr`** gets full pipeline treatment (files, gpkg, DB table, snapshots)
   and is wired into `update_datasets` and the Geonode mirror **with clearly-marked placeholder
   constants** for the dataset UID and the mirror table name — Lionel will fill them in. The code
   must skip those steps gracefully (log, continue) while the placeholders are unfilled.
4. **Scope: `pipeline.py` and `surveys.py` only** (plus, if genuinely useful, one new module for the
   legacy schema constants). No opportunistic refactors, no reformatting untouched code, no new
   dependencies. Leave the commented-out duplicate-detection code (`haversine`, `group_pairs`,
   `identify_duplicates`, …) exactly where it is. **Superseded by §10.7: Lionel later asked for
   this block to be removed as part of a dead-code cleanup — it is gone as of that date.**
   *Reading* outside that scope is expected and required — the reference parquet files and the
   `add_infrastructure_id` pipeline's `config.py` (§5) are inputs. Do not **edit** anything outside
   the two files, and in particular do not modify the `add_infrastructure_id` pipeline.

---

## 4. Ground truth: the consolidated survey

Taken from the XLSForm **`FICHE_SIMPLIFIEE_INFRASTRUCTURES_final_config_v7.xlsx`** (contained in the pipeline folder `.../extract_surveys`) — form `id_string` `TBT`, `version` `3107261400`, default language `french`,
**85 form rows**. Use it as the reference for what the new
survey does and does not contain — but always confirm against the actual downloaded parquet, since
Kobo adds `_`-prefixed system columns, drops columns that were never answered by anyone, and the
deployed form may differ from this config file.


**Consequence you must handle — partial / blocked submissions.** When an enumerator types an
identifier at `IDT` that already exists, the rest of the form is skipped and the submission can be
saved containing only `group1` + `group2`: **no `LUV1..LUV6`, no `_geolocation`, no
`INFRASTRUCTURE_ID`, no `STMB*`, no photos.** Today's code assumes those exist. Concretely:

* `drop_duplicates(subset="INFRASTRUCTURE_ID")` would collapse *all* such rows into a single row
  (null == null in `unique`), and `concatenate_snapshots` would carry that artefact forward;
* `transform_survey`'s `pl.col("level_7").struct.json_encode()` fails if `LUV6` is missing from the
  frame or is not a struct column;
* the `GEO_COLUMNS` aliasing (`pl.col(field).alias(f"level_{lvl}")`) raises `ColumnNotFoundError` if
  a `LUV*` column is absent from the parquet entirely;
* `_geolocation` null → `LATITUDE`/`LONGITUDE` null → no geometry, so these rows must not reach the
  gpkg / PostGIS write.

Required handling: guard the `GEO_COLUMNS` / `level_7` steps with `if <col> in df.columns`, and drop
blocked/partial submissions **after** `transform_survey` and **before** the split — filter to rows
where `INFRASTRUCTURE_ID` is non-null and non-blank — logging the dropped count with
`current_run.log_warning`. These are data-entry rejects, not real infrastructures; they must not
reach any output table.

**Verified unchanged from v5:** the entire `choices` sheet (2716 rows, byte-identical set), so the
`TYPE` / `CDR` / `HCDR` code↔label tables in §4.2 remain valid. No field was removed or renamed.

### 4.1 Field inventory (in form order)

```
start           starttime
end             endtime
deviceid        deviceid
phonenumber     devicephonenum
note            praps2

group1 — IDENTIFICATION DU COLLECTEUR ET DE L'ENQUÊTE
  date          DATE
  text          IDUV1
  select_one    IDUV2          (9 choices)
  text          IDUV2A         relevant: IDUV2 = 9
  text          IDUV3
  text          IDUV7
  text          IDUV8
  text          IDUV9

group2 — TYPE / CATEGORY
  select_one    TYPE           1 = CDR, 2 = HORS CDR
  select_one    CDR            relevant: TYPE = 1
  select_one    IGPE0          relevant: CDR = 4  (type of veterinary unit)
  text          IGPE0a         relevant: IGPE0 = 7
  select_one    IGPE1          relevant: CDR = 3  (type of water point)
  text          IGPE1a         relevant: IGPE1 = 8
  select_multiple IGPE2        relevant: CDR = 3  (water-lifting mode)
  text          IGPE2a         relevant: IGPE2 includes 6
  select_one    HCDR           relevant: TYPE = 2
  text          STMB2
  select_one    STMB0          LIST: 1 = Oui, 2 = Non  (new site vs existing site)
  text          IDT            relevant: STMB0 = 1        (manual identifier entry)
  calculate     IDT_DUP_COUNT  [NEW v7] count of points_live rows whose INFRASTRUCTURE_ID = IDT
  note          IDT_DUP        [NEW v7] relevant: IDT != '' and IDT_DUP_COUNT > 0
  acknowledge   IDT_DUP_BLOCK  [NEW v7] required; relevant: IDT != '' and IDT_DUP_COUNT > 0
  text          STMB01         relevant: STMB0 = 2
  text          STMB02         relevant: STMB0 = 2
  text          STMB03         relevant: STMB0 = 2
  calculate     TYPE_ACRONYM   MB|PV|PE|UV for CDR 1-4, else HCDR acronym (see 4.3)

group2-1 — LOCATION        [v7] relevant: (IDT = '' or IDT_DUP_COUNT = 0)
  select_one    LUV1           Pays        (6 choices)
  select_one    LUV2           Region      (80)
  select_one    LUV3           Departement (359)
  select_one    LUV4           Commune     (2176)
  text          LUV5
  select_one_from_file points_live.csv  LUV6_existing   relevant: STMB0 = 2
  geopoint      LUV6_manual    relevant: STMB0 = 1
  geopoint      LUV6
  calculate     NEARBY_RADIUS  constant 1500 (metres)
  calculate     NEARBY_COUNT   nearby points_live entries, same LUV3 + same TYPE_ACRONYM
  calculate     NEARBY_IDS     comma-joined INFRASTRUCTURE_IDs of those
  calculate     BASE_ID        TYPE_ACRONYM + rounded lat/lon
  calculate     DUP_COUNT
  calculate     INFRASTRUCTURE_ID   = IDT, else LUV6_existing, else BASE_ID[_DUP_COUNT]
  note          IDT2
  calculate     NEARBY_MSG     [NEW v7] warning text built from NEARBY_COUNT / NEARBY_IDS
  note          IDT3           relevant: STMB0 = 1 and NEARBY_COUNT > 0

group3 — SUIVI DES TRAVAUX  [v7] relevant: (IDT = '' or IDT_DUP_COUNT = 0)
  select_one    STMB1          relevant: STMB0 = 2
  date          STMB3          relevant: STMB0 = 2
  integer       STMB4          relevant: STMB0 = 2
  select_one    STMB5          STUV5 list — works state (7 choices)
  date          STMB11         relevant: STMB5 = 5
  date          STMB12         relevant: STMB5 = 6
  date          STMB13         relevant: STMB5 = 7
  calculate     STMB14
  select_one    STMB15         STUV15 list — progress bracket (7 choices)
  note          CONFSITE
  select_one    CONFSITE1      (4 choices)
  select_one    DATTROL        relevant: IDUV2 != 4  (6 choices)
  text          CONFSITE2
  group3-1 — relevant: HCDR in (5, 9, 10)   [linear infrastructures]
    geotrace    STMB16
    calculate   STMB17
    calculate   STMB18
    select_one  note4

group3-2 — PHOTOS          [v7] relevant: (IDT = '' or IDT_DUP_COUNT = 0)
  note          LUV7
  image         LUV7a
  image         LUV7b
  image         LUV7c
  image         LUV7d
```

### 4.2 Split keys — choice codes and labels

`TYPE`: `1` = *CDR*, `2` = *HORS CDR*

`CDR`:

| code | label |
| --- | --- |
| 1 | `Marché à bétail` |
| 2 | `Parc de vaccination` |
| 3 | `Point d'eau` |
| 4 | `Unité vétérinaire` |

`HCDR` (13 choices — all of them route to `infrastructures_hors_cdr`):

| code | label |
| --- | --- |
| 1 | `Aires d'abattage` |
| 2 | `Centres collecte/Mini Laiteries` |
| 3 | `Etals de boucherie` |
| 4 | `Magasins aliment betail` |
| 5 | `Couloirs balisés (en Km)` |
| 6 | `Quai d'embarquement/débarquement` |
| 7 | `Aire de quarantaine` |
| 8 | `Unité de tannerie` |
| 9 | `Couloirs de transhumances sécuriés des zones conflictuelles` |
| 10 | `Réalisation des bandes des pare feux dans les zones de hautes prairies d’herbes` |
| 11 | `Sécurisation des aires de stationnement` |
| 12 | `Construction des cordons pierreux dans les bas fonds` |
| 13 | `Autre (à Préciser)` |

> **Codes vs labels — verify, do not assume.** `openhexa.toolbox.kobo.utils.to_dataframe()` may
> return either the raw choice code (`"2"`) or the French label (`"Parc de vaccination"`) for
> `select_one` fields. Read the toolbox source in the installed environment to confirm, and write
> the matcher to **accept both** (compare against code *and* label, after `strip()`), so the split
> is robust either way. Note the label strings contain accents and a typographic apostrophe
> (`Point d'eau`, `d’herbes`) — keep the files UTF-8 and copy the strings from this document
> rather than retyping them.

`surveys.py` already contains the skeleton for this:

```python
CDR_INFRA_LIST = {
    "marches_a_betail": "Marché à bétail",
    "parcs_de_vaccination": "Parc de vaccination",
    "points_d_eau": "Point d'eau",
    "unites_veterinaires": "Unité vétérinaire",
}
HCDR_INFRA_LIST = set()   # unused placeholder — replace with the HCDR rule
```

Extend `CDR_INFRA_LIST` values to carry the code as well (e.g. `{"marches_a_betail": ("1", "Marché à
bétail"), ...}`) or add a parallel code map — your choice, but keep one single source of truth for
the mapping.

`infrastructures_hors_cdr` = rows where `HCDR` is **not empty**: not null, not `""`, not whitespace.
Do not additionally filter on `TYPE == 2` — `HCDR` non-empty is the rule Lionel specified. (You may
`log_warning` on rows where the two disagree.)

### 4.3 `TYPE_ACRONYM` — use it as a cross-check

The form computes a category acronym that is also embedded in `INFRASTRUCTURE_ID` (via `BASE_ID`):

```
CDR  1 → MB    2 → PV    3 → PE    4 → UV
HCDR 1 → AA    2 → CCML  3 → EB    4 → MAB   5 → CB    6 → QED   7 → AQ
     8 → UT    9 → CTSZC 10 → RBPF 11 → SAS  12 → CCP  13 → AUTRE (fallback)
```

The split rule stays the one Lionel specified (`CDR` / `HCDR`). But `TYPE_ACRONYM` is a stable,
accent-free, single-token value, so use it as a **cross-check**: after splitting, assert that every
row of `marches_a_betail` has `TYPE_ACRONYM == "MB"` (`PV`, `PE`, `UV` respectively) and that every
row of `infrastructures_hors_cdr` has an HCDR acronym; `log_warning` on any mismatch. If you
discover that `to_dataframe()` returns something unexpected for `CDR`/`HCDR`, `TYPE_ACRONYM` is the
safe fallback split key — say so explicitly in your summary if you fall back to it.

---

## 5. BLOCKING PREREQUISITE — read the two reference sources

**Do not invent, guess, or reconstruct-from-memory either the old column names or the field
mapping.** Getting one wrong silently breaks a production web map. Both answers already exist on
disk. Read them **before writing any code**, and if you cannot find one of them, **stop and ask
Lionel** rather than shipping a plausible guess.

### 5.1 Source A — the 4 reference parquet files → *which* columns are mandatory

The `extract_surveys` pipeline folder contains the last-generated parquet file for each of the 4
pre-existing surveys:

```
marches_a_betail.parquet
parcs_de_vaccination.parquet
points_d_eau.parquet
unites_veterinaires.parquet
```

They sit in the same pipeline folder as `pipeline.py` / `surveys.py` (`.../extract_surveys`). If they
are not directly beside the code, search the repository/workspace for them by name before concluding
anything — check `data/kobo/surveys/` too, which is where the pipeline writes them at run time.

**The column set of each of these files is the mandatory column set of the corresponding output
table.** Inspect them with polars — schema only, do not load the data:

```python
import polars as pl
for name in ("marches_a_betail", "parcs_de_vaccination", "points_d_eau", "unites_veterinaires"):
    print(name, dict(pl.scan_parquet(f"{name}.parquet").collect_schema()))
```

Use both the **names and the dtypes**, and preserve the **column order** as it appears in the file.
Report the per-file column count in your summary so Lionel can sanity-check it.

If a `geo/{name}.gpkg` is also present, cross-check against it — that is what actually reaches
PostGIS and Geonode; flag any discrepancy instead of silently picking one.

### 5.2 Source B — `all_forms_cols_mapping` → *how* to rename consolidated fields

A sibling pipeline, **`add_infrastructure_id`**, has a `config.py` containing:

```python
all_forms_cols_mapping
```

This dict defines how each field of the consolidated survey relates to the fields of the 4 existing
surveys **plus** the new `infrastructures_hors_cdr` one. It is the authoritative answer to "what does
consolidated field `X` need to be called in table `Y`" — use it to build the rename map, do not
re-derive the correspondence yourself from field prefixes or labels.

Locate it with a search (e.g. `rg -l all_forms_cols_mapping`, or glob for
`**/add_infrastructure_id/config.py`); it is a sibling pipeline directory in the same workspace/repo.
Read the whole dict, and note carefully:

* which **direction** it maps (consolidated → legacy, or legacy → consolidated) and invert it if
  needed — state which direction you found in your summary;
* whether it is keyed per survey or flat, and whether one consolidated field maps to **different**
  names in different surveys (expected — that is the whole point);
* any consolidated field that maps to **nothing** for a given survey (leave it as a consolidated-only
  extra column), and any legacy column with **no** consolidated source (that one becomes a typed-null
  column);
* whether it contains an entry for `infrastructures_hors_cdr`, and if so, apply it — that table's
  columns are then *not* simply the raw consolidated names.

**Reconcile A and B and report the reconciliation.** After applying the rename map to the
consolidated columns, compare the result against each parquet's column set and report, per survey:
how many mandatory columns are satisfied by a renamed consolidated field, how many have no source
and become typed nulls, and how many consolidated-only extras are added. If a mandatory column is
neither in the mapping nor in the consolidated survey, list it explicitly — do not quietly null it
without saying so.

### 5.3 Encode the result as constants

Commit what you derived into `surveys.py` (or one new small module) as literal constants — the
pipeline must not read the reference parquet files or the other pipeline's `config.py` at run time:

```python
# Column schema of each output table as it existed BEFORE the survey consolidation.
# Source of truth for the Geonode layer configuration — DO NOT rename or remove entries.
# Derived on <date> from <exact paths of the parquet files read>.
LEGACY_COLUMNS: Dict[str, Dict[str, pl.DataType]] = {
    "marches_a_betail": {
        "DATE": pl.Date,
        "INFRASTRUCTURE_ID": pl.String,
        ...
    },
    ...
}

# Consolidated survey field -> historical field name, per output table.
# Derived from all_forms_cols_mapping in <exact path of add_infrastructure_id/config.py>.
COLUMN_RENAMES: Dict[str, Dict[str, str]] = {
    "marches_a_betail": {"LUV1": "LMB1", "LUV2": "LMB2", ...},
    ...
}
```

If either source cannot be found: implement the mechanism, leave the constant with an explicit `TODO`
and a single obvious place to paste the data, make the code `raise` on an empty schema rather than
silently emitting a wrong-shaped table, and say so in your summary.

`infrastructures_hors_cdr` has no historical table, so it has no `LEGACY_COLUMNS` entry — but it may
still have a `COLUMN_RENAMES` entry if `all_forms_cols_mapping` defines one (§5.2).

---

## 6. Implementation plan

### 6.1 `surveys.py`

* Keep `SURVEYS` as the single-entry source list (the consolidated survey), or rename it to something
  unambiguous like `SOURCE_SURVEY = ("aRZ43wRerX8pCdo4XrKw4n", "fiche_simplifiee_infrastructures")`
  — if you rename, update every reference in both files.
* Add a module-level list of the five output names, in a stable order:

  ```python
  OUTPUT_TABLES = [
      "marches_a_betail",
      "parcs_de_vaccination",
      "points_d_eau",
      "unites_veterinaires",
      "infrastructures_hors_cdr",
  ]
  ```
* `PROGRESS`, `STATE`, `PICTURES`, `GEO_COLUMNS` stay keyed by
  `"fiche_simplifiee_infrastructures"` — they are applied by `transform_survey` **before** the split
  (see 6.2). Their current values are already correct for the consolidated form: `STATE` → `STMB5`,
  `PROGRESS` → `STMB15`, `PICTURES` → `LUV7a..LUV7d`, `GEO_COLUMNS` → `LUV1..LUV6` as
  `level_2..level_7`. Delete the commented-out per-survey entries only if that keeps the diff clean;
  leaving them is acceptable.
* **New:** `split_consolidated(df: pl.DataFrame) -> Dict[str, pl.DataFrame]`
  * returns one frame per entry of `OUTPUT_TABLES` (include empty frames rather than omitting keys);
  * CDR splits match on code **or** label as described in §4.2;
  * HCDR split = `HCDR` non-null and non-blank;
  * `log_info` the row count of each split, and `log_warning` for rows that fall into **no** split
    (e.g. `TYPE = 1` with empty `CDR`) with the count and a couple of `_id` values, so bad
    submissions are visible without failing the run;
  * a row must not land in two splits — assert/log if it does.
* **New:** `conform_to_legacy_schema(df: pl.DataFrame, name: str) -> pl.DataFrame` — three ordered
  steps, all driven by the §5 constants:
  1. **Rename** — apply `COLUMN_RENAMES[name]` (from `all_forms_cols_mapping`, §5.2) to map
     consolidated field names to that survey's historical names. Only rename keys that are actually
     present in `df`; if a rename target collides with an existing column, `log_warning` and keep the
     legacy-named one (do not silently overwrite). Do not rename anything for a survey with no entry.
  2. **Fill** — for every column in `LEGACY_COLUMNS[name]` still missing after the rename: add it as
     an all-null column **cast to the declared dtype** (`pl.lit(None).cast(dtype).alias(col)`).
     `log_info` the list of columns filled this way — Lionel wants to know which questions the
     consolidated form no longer collects.
  3. **Order and keep** — legacy columns first, in the order they appear in the reference parquet,
     then the consolidated-only columns. Never drop a column: extras are explicitly wanted.
  * for columns present in both: keep the data; cast to the legacy dtype if that is safe, otherwise
    keep the source dtype and `log_warning` the mismatch (a type change on a Geonode-referenced
    column is worth surfacing);
  * post-condition worth asserting in code: `set(LEGACY_COLUMNS[name]) <= set(out.columns)`;
  * for a name with no legacy schema (`infrastructures_hors_cdr`), apply step 1 if a rename entry
    exists and return; there is nothing to fill.
* **New:** `drop_blocked_submissions(df: pl.DataFrame) -> pl.DataFrame` (name it as you like) —
  removes the v7 partial/blocked submissions described in §4.0: rows where `INFRASTRUCTURE_ID` is
  null or blank. `log_warning` the count dropped (and, if present, the `IDT` values, since those are
  the duplicate identifiers the enumerators tried to reuse — that list is useful to Lionel).
  Apply it **before** dedup and before the split.
* `transform_survey`, `drop_duplicates`, `concatenate_snapshots`, `serialize`, `_add_url_prefix`:
  keep their current behaviour, with two v7-driven robustness fixes:
  * the `GEO_COLUMNS` aliasing loop must skip fields that are not in `df.columns` (`log_warning`
    once per missing field) — Kobo omits columns that no submission ever answered, and v7 lets a
    submission stop before `group2-1`;
  * `pl.col("level_7").struct.json_encode()` must be guarded: only run it if `level_7` exists **and**
    its dtype is a `pl.Struct` (otherwise leave / null it and log).

  In `transform_survey` you may drop the now-unreachable
  `if name in ("indicateurs_regionaux", "indicateurs_pays"): return df, df` early-exit; keep
  everything else, including the `INFRASTRUCTURE_ID` / `DATE` dedup.

### 6.2 `pipeline.py`

* `SURVEYS` / `download`: unchanged in behaviour — one survey downloaded to
  `raw/fiche_simplifiee_infrastructures.parquet`, fields metadata to
  `metadata/fiche_simplifiee_infrastructures_fields.{parquet,xlsx}`.
* `transform`: **currently contains a bare `xxx` placeholder at line 107 — that is where this work
  goes, and it must not survive.** New shape:

  1. read `raw/fiche_simplifiee_infrastructures.parquet`; if absent, `log_warning` and return as today;
  2. call `surveys.transform_survey(survey, "fiche_simplifiee_infrastructures")` **once** on the full
     consolidated frame, so `LATITUDE`/`LONGITUDE`, `validation_status`, `level_2..level_7`,
     `STATE`/`PROGRESS`, struct/list serialization and picture URLs are computed once and identically
     for all outputs;
  2b. drop the v7 blocked/partial submissions (`drop_blocked_submissions`, §4.0) from both returned
     frames, before splitting;
  3. `surveys.split_consolidated(...)` on **both** returned frames (`df` with duplicates and
     `df_no_duplicates`) — or split once and dedup per split; either is fine as long as the
     `*_with_duplicates` / deduped distinction is preserved as it is today;
  4. per output name: `conform_to_legacy_schema(...)`, then write exactly the artefacts the old loop
     wrote, with the output name substituted for the survey name:
     `surveys/{name}_with_duplicates.parquet`, `surveys/{name}.parquet`, `surveys/{name}.xlsx`,
     `geo/{name}.gpkg`, `snapshots/{name}_snapshots.parquet`, plus `current_run.add_file_output(...)`
     under `Environment.CLOUD_PIPELINE` and the closing `log_info`;
  5. **do not** write any artefact named `fiche_simplifiee_infrastructures` under `surveys/`,
     `geo/`, or `snapshots/`.
* **Empty-split guard (important).** If a split is empty (or has no non-null geometry):
  `log_warning` and **skip** writing its gpkg and skip its DB push — do not let an empty frame
  replace a populated production table, and do not let `to_file()` / `concatenate_snapshots()` raise
  on an empty frame (note `concatenate_snapshots` calls `df["DATE"].min().year`, which fails on an
  empty or all-null column). One survey category having zero submissions in a country must not fail
  the whole run.
* `push`: iterate `OUTPUT_TABLES` instead of `SURVEYS`. Delete the
  `if name in ("indicateurs_regionaux", "indicateurs_pays", "cultures_vivrieres")` branch — every
  output now takes the PostGIS path (gpkg → `to_postgis(..., if_exists="replace")`) plus its
  `{name}_snapshots` table. Keep `current_run.add_database_output(...)` and the log lines.
* `push` — Geonode mirror: reduce `mapping` to the tables still produced:

  ```python
  mapping = {
      "PRAPS2_Marches_a_Betail": "marches_a_betail",
      "PRAPS2_Points_d_Eau": "points_d_eau",
      "PRAPS2_Unites_Veterinaires": "unites_veterinaires",
      "PRAPS2_Parcs_de_Vaccination": "parcs_de_vaccination",
      # TODO(Lionel): mirror table name for the new HCDR layer, e.g.
      # "PRAPS2_Infrastructures_Hors_CDR": "infrastructures_hors_cdr",
  }
  ```

  Drop the four entries whose source tables are no longer produced
  (`activites_generatrices_de_revenus`, `fourrage_cultive`, `gestion_durable_des_paysages`,
  `sous_projets_innovants`) — the loop would otherwise copy stale data or fail on a missing table.
  Wrap each mirror copy in `try/except` + `log_warning` so one bad layer cannot abort the task, and
  reuse the single `engine` already created at the top of `push` rather than creating a second one.
* `update_datasets`: trim `DATASETS` to the 4 surviving entries — keep their existing UIDs verbatim:

  ```python
  ("parcs_de_vaccination", "Parcs de vaccination", "parcs-de-vaccination-a6fbd3"),
  ("unites_veterinaires",  "Unités vétérinaires",  "unites-veterinaires-05ee52"),
  ("marches_a_betail",     "Marchés à Bétail",     "marches-a-betail-286942"),
  ("points_d_eau",         "Points d'Eau",         "points-d-eau-0935a6"),
  # TODO(Lionel): create the OpenHexa dataset and paste its UID here
  # ("infrastructures_hors_cdr", "Infrastructures Hors CDR", "TODO_DATASET_UID"),
  ```

  Skip any entry whose UID is a placeholder (`log_info` and `continue`) so an unfilled TODO cannot
  crash the task.
* `update_datasets` — fields metadata: the per-survey `metadata/{survey_name}_fields.xlsx` no longer
  exists; there is now one `metadata/fiche_simplifiee_infrastructures_fields.xlsx`. Attach that
  single consolidated file to every dataset version (the existing `[p for p in src_files if
  p.exists()]` filter means an untouched code path would silently stop shipping field metadata —
  that would be a silent regression, so handle it explicitly). Note `add_file(path, name)` takes an
  explicit name, so the file can be attached under a stable name.

---

## 7. Explicit do-nots

* Do not rename or drop any column of the 4 existing output tables. Additive only.
* Do not change the output table names, the file/directory layout under `data/kobo/`, the pipeline
  name/id `extract-surveys`, or its 3 parameters (`output_dir`, `push_to_db`, `overwrite`).
* Do not change the existing dataset UIDs or the `PRAPS2_*` mirror table names.
* Do not reintroduce the 7 out-of-scope surveys anywhere, including in `DATASETS`, `PICTURES`,
  `GEO_COLUMNS`, and the mirror mapping.
* Do not add runtime database introspection, new dependencies, or an alembic-style migration. The
  pipeline must not read the reference parquet files or import `add_infrastructure_id.config` at run
  time — those are design-time inputs, transcribed into constants.
* Do not re-derive the consolidated→legacy field correspondence from prefixes, labels, or intuition
  when `all_forms_cols_mapping` (§5.2) already answers it, and do not edit that dict.
* Do not uncomment or "restore" the duplicate-detection block in `surveys.py`. **Superseded by
  §10.7 — this block has since been removed entirely, not restored.**
* Do not leave the `xxx` placeholder, or any `print()` / debug leftovers.
* Do not reformat or reorder code you did not need to touch — keep the diff reviewable.

---

## 8. Verification before you report done

There is no Kobo credential and no warehouse in the dev environment, so verify offline:

1. `python -m py_compile pipeline.py surveys.py`, and `ruff check` / `ruff format --diff` if ruff is
   available (match the existing style: 88-col, double quotes).
2. Build a **synthetic fixture** — a small polars frame with the consolidated survey's columns
   (§4.1) plus the Kobo system columns the code touches (`_validation_status` struct with `label`,
   `_geolocation` list of 2 floats, `_id`), covering:
   * one row per `CDR` value, and two rows with non-empty `HCDR`;
   * one row with `TYPE = 1` and empty `CDR` (falls into no split);
   * one duplicate `INFRASTRUCTURE_ID` with two different `DATE`s;
   * one row with null `_geolocation`;
   * **[v7] one blocked/partial submission**: `IDT` filled, `IDT_DUP_COUNT > 0`, and everything from
     `group2-1` onwards null — including null `INFRASTRUCTURE_ID`, null `LUV6`/`level_7` and null
     `_geolocation`;
   * **[v7] a variant frame with the `LUV*` / `STMB*` / `LUV7*` columns absent entirely**, to prove
     the `df.columns` guards work (this is what a fresh form with few submissions looks like).

   Run `transform_survey` → `drop_blocked_submissions` → `split_consolidated` →
   `conform_to_legacy_schema` on it and assert:
   * the blocked submission is dropped, with a warning, and never reaches a split;
   * every split has the expected row count and no row appears in two splits;
   * `TYPE_ACRONYM` is consistent with the split each row landed in (§4.3);
   * for each of the 4 legacy tables, `set(LEGACY_COLUMNS[name]) <= set(result.columns)` and the
     legacy columns come first in their original order — assert against the column set read straight
     from the reference parquet (§5.1), not against a retyped copy of it;
   * the renames from `COLUMN_RENAMES` actually happened, and **carried their data**: pick two or
     three renamed fields per survey and assert the values survived the rename (a rename that
     produces a correctly-named all-null column is the failure mode to catch here);
   * columns absent from the consolidated survey exist and are all-null with the declared dtype;
   * consolidated-only columns (`TYPE`, `HCDR`, `CONFSITE1`, `DATTROL`, `IGPE0/1/2`, `STMB16/17/18`,
     `NEARBY_*`, `NEARBY_MSG`, `IDT_DUP_COUNT`, `BASE_ID`, `DUP_COUNT`, `TYPE_ACRONYM`, …) are
     present;
   * dedup kept the most recent `DATE` per `INFRASTRUCTURE_ID`;
   * an empty split produces warnings and no exception, and writes no gpkg;
   * the missing-columns variant runs end to end without raising.

   Keep this as a throwaway script under `/tmp` unless the repo already has a test suite — §3.4
   limits the scope to the two files.
3. Re-read your own diff and check it against §7 line by line.
4. In your final summary, state: the exact paths of the 4 reference parquet files and the
   `add_infrastructure_id/config.py` you read (§5), each file's column count, the direction
   `all_forms_cols_mapping` maps in, the per-survey reconciliation figures required by §5.2
   (renamed / null-filled / extra), whether Kobo returns choice codes or labels and how you verified
   it, how many blocked/partial submissions your filter would drop if you were able to inspect real
   data, every `TODO` placeholder you left and what Lionel must paste into it, and anything you had
   to assume.

## 9. Definition of done

* `transform` consumes only `raw/fiche_simplifiee_infrastructures.parquet` and produces the 5 output
  sets; the `xxx` placeholder is gone.
* The 4 legacy tables are column-name-compatible with their pre-consolidation versions — every column
  of the corresponding reference parquet (§5.1) is present — plus the new columns.
* Consolidated fields are renamed per `all_forms_cols_mapping` (§5.2), with data intact, and the
  reconciliation between the two sources is reported rather than assumed.
* `infrastructures_hors_cdr` is produced end to end, with clearly marked placeholders for its
  dataset UID and Geonode mirror name, and no crash while they are unfilled.
* Nothing referencing the 7 out-of-scope surveys remains in the executed code paths.
* An empty or geometry-less split logs a warning instead of failing the run or wiping a table.
* v7 blocked/partial submissions (null `INFRASTRUCTURE_ID`) are dropped with a warning and never
  reach an output table, and missing `LUV*` / `level_7` columns cannot raise.

---

## 10. Amendments after initial implementation (2026-08-05)

§1–§9 above is the *original* spec, exactly as given, and is left unedited for the historical
record. The pipeline was implemented against it, then run for real and adjusted based on what
that surfaced. Where this section and §1–§9 disagree, **this section wins.** Each entry below is
labeled as either an explicit instruction from Lionel, or a correction forced by something the
original spec's sources (§5) turned out not to fully account for.

### 10.1 `infrastructure_id` / `INFRASTRUCTURE_ID` case collision — Lionel's instruction

Running the pipeline raised `pyogrio.errors.FieldError: Error adding field 'INFRASTRUCTURE_ID' to
layer` on every `geo/{name}.gpkg` write. Root cause: each of the 4 legacy tables carries a dead,
always-null `infrastructure_id` column (lowercase, `UInt32` — a leftover row index from the
long-disabled `identify_duplicates()`), alongside the real, populated `INFRASTRUCTURE_ID`
(uppercase, `String`) inherited from the consolidated survey. GDAL's GPKG driver rejects two field
names that differ only by case — reproduced directly, outside this pipeline, with a 2-column
GeoDataFrame.

Lionel: *"I would like that the column gets renamed in lower case in the pipeline such that the
lower case version is used in all tables and gpkg/postGIS files."*

Implemented in `transform_survey()` (`surveys.py`): `INFRASTRUCTURE_ID` is renamed to
`infrastructure_id` immediately after computing `LATITUDE`/`LONGITUDE`, before anything else
touches it. `drop_duplicates()`'s internal call (inside `transform_survey`) and
`concatenate_snapshots()`'s call (in `pipeline.py`) now key on `infrastructure_id` (lowercase)
throughout. `LEGACY_COLUMNS["infrastructure_id"]` in `legacy_schema.py` is typed `pl.String` for
all 4 tables, **not** the `pl.UInt32` found in the raw reference parquet — a deliberate, documented
deviation from §5.3's literal-transcription rule, since there is now one identifier column instead
of two.

### 10.2 One stale entry in `all_forms_cols_mapping` — found during verification, not an instruction

Per §5.2, `all_forms_cols_mapping` is applied as-is, without re-deriving the correspondence. One
entry doesn't survive contact with the real reference parquet: it maps consolidated `IDUV2A` →
legacy `IDUV2a` (lowercase `a`) for `unites_veterinaires`, but `unites_veterinaires.parquet`'s real
column is `IDUV2A` (uppercase — already an identity, no rename needed). Applying the mapping
literally reintroduced the exact case collision from §10.1. Per §5.2's own principle ("flag any
discrepancy instead of silently picking one"), Source A (the reference parquet) wins: that one
`COLUMN_RENAMES["unites_veterinaires"]` entry was removed. Documented inline in `legacy_schema.py`.

### 10.3 `infrastructures_hors_cdr`'s `COLUMN_RENAMES` reverted to empty — found during verification, not an instruction

§5.2/§5.3 permitted (did not mandate) applying `all_forms_cols_mapping`'s Mali-pilot-derived entry
for `infrastructures_hors_cdr` (e.g. `DATE` → `_1_Date_de_la_collecte`, from the
`FICHE AIRE D'ABATTAGE, ETAL, MAGASINS, LATRINES_MALI` block). It was applied, then reverted:
renaming `DATE` away broke `concatenate_snapshots()`, which hard-codes `column_date="DATE"` for
every output table (`polars.exceptions.ColumnNotFoundError: "DATE" not found`). Unlike the 4
legacy tables, `infrastructures_hors_cdr` has no Geonode-facing predecessor that needs those
Mali-specific names, so there was no compatibility benefit to weigh against the breakage.
`COLUMN_RENAMES["infrastructures_hors_cdr"]` is now `{}`; the table keeps plain consolidated field
names.

### 10.4 The 4 non-CDR legacy surveys are back in scope — Lionel's instruction, supersedes §2 and §7

Lionel: *"I now want to re-adapt the code to preserve the other legacy source tables
`fourrage_cultive`, `sous_projets_innovants`, `gestion_durable_des_paysages`,
`activites_generatrices_de_revenus`. Please adapt the pipeline such that it also produces the same
outputs as before for these surveys."*

This **supersedes** §2's "7 other historical surveys ... are out of scope" and §7's "do not
reintroduce the 7 out-of-scope surveys anywhere" — for these 4 specifically.
`indicateurs_regionaux`, `indicateurs_pays`, and `cultures_vivrieres` were not named and remain out
of scope, still commented out everywhere.

These 4 never merged into the consolidated survey — they are independent Kobo surveys, downloaded,
transformed and pushed exactly as before the consolidation work, entirely bypassing
`split_consolidated`/`conform_to_legacy_schema`/`drop_blocked_submissions` (those stay
consolidated-survey-only concerns). Implementation:

* `pipeline.py`: `SURVEYS` split into `LEGACY_SURVEYS` (these 4, uncommented with their original
  UIDs) + `CONSOLIDATED_SURVEY`; `download()` needed no change since it already just iterates
  `SURVEYS`.
* `transform()`: a second loop over `LEGACY_SURVEYS` runs the plain pre-consolidation logic
  (`transform_survey` → write outputs), sharing the write logic with the consolidated-splits loop
  via a new `_write_survey_outputs()` helper (avoids duplicating the parquet/xlsx/gpkg/snapshots
  block across the two loops).
* `surveys.py`: uncommented the pre-existing (already-correct) `PICTURES` and `GEO_COLUMNS`
  entries for these 4 names. `PROGRESS`/`STATE` had no entries for them and still don't — those
  are CDR-form-specific fields these surveys never had.
* `push()` / `update_datasets()`: their `PRAPS2_*` mirror entries and dataset entries (original
  names/UIDs) restored. Fields-metadata attachment in `update_datasets()` is now survey-aware: the
  5 consolidated-derived tables share one `fiche_simplifiee_infrastructures_fields.xlsx`; these 4
  keep their own per-survey `{name}_fields.xlsx`, since they're still downloaded independently.
* Real bug found by testing against archived raw data for all 4: `drop_duplicates()` and
  `concatenate_snapshots()` both hard-coded `infrastructure_id` as the dedup key, but that column
  only ever existed for surveys that went through the now-disabled `identify_duplicates()` — these
  4 never did. Both functions now degrade gracefully (skip dedup / fall back to Kobo's own `_id`)
  instead of raising `ColumnNotFoundError` when the column is absent.

### 10.5 `_tags` / `_notes` on every table except `parcs_de_vaccination` — Lionel's instruction

Lionel: *"The GeoServer layers config identify two additional fields `_tags` and `_notes` supposed
to be present in all final tables except `PRAPS2_Parcs_de_Vaccination`. Verify that these fields
have not been dropped by error during the transformation process and if not, add these two extra
fields with missing values in each table relating to these surveys."*

Verified: not a transformation bug. `_tags`/`_notes` are absent from the *raw* Kobo download
itself, for every survey (consolidated and all 4 legacy) — `to_dataframe()` only emits a column
for a key present in at least one submission, and no submission in the current data has ever been
tagged or annotated via the Kobo UI. Confirmed this isn't a schema-inference-truncation artifact
either (`to_dataframe()` already runs with `infer_schema_length=None`, patched into the installed
`openhexa.toolbox.kobo.utils` in an earlier session — that patch lives in `site-packages`, not this
repo, and won't survive an environment rebuild). Since there is no source data to carry, both are
added as null (`pl.String`) at the one point every output table passes through regardless of
origin: `_write_survey_outputs()` in `pipeline.py`, via `GEONODE_EXTRA_COLUMNS = ["_tags",
"_notes"]` and `TABLES_WITHOUT_GEONODE_EXTRA_COLUMNS = {"parcs_de_vaccination"}`.

### 10.6 `surveys.py`'s own `SURVEYS` list had stale UIDs — Lionel's instruction

Lionel noticed `surveys.py`'s `SURVEYS` list still commented out the 4 surveys re-added in §10.4.
That list is dead code (see §10.7 — nothing calls it or the `download_surveys()` function that
used it; `pipeline.py` has its own `SURVEYS`/`LEGACY_SURVEYS`/`CONSOLIDATED_SURVEY` that actually
drives `download()`), but it was still misleading to read. Its UIDs for these 4 surveys also
didn't match the ones actually restored in `pipeline.py`'s `LEGACY_SURVEYS` (the real, verified
ones). Uncommented and corrected to match `pipeline.py` exactly, rather than uncommenting stale
values.

### 10.7 Dead-code removal — Lionel's instruction

Lionel asked to remove all dead code in the pipeline. Audited with `pyflakes` plus manual
call-site tracing across both files. Removed:

* `surveys.py`'s `download_surveys()` function and its module-level `SURVEYS` list (from §10.6) —
  unreachable: nothing called `download_surveys()`, and nothing but that function read `SURVEYS`.
  `pipeline.py`'s `download()` task has always used its own separate `SURVEYS` constant.
* The entire commented-out `haversine()` / `group_pairs()` / `reassign_ids()` /
  `identify_duplicates()` block, and the leftover commented call-site inside `transform_survey()`
  that invoked it.
* The imports that existed only to support that block: `itertools.combinations`, `math.{asin, cos,
  radians, sin, sqrt}`, and `typing.{List, Sequence, Tuple}` (`typing.Dict` is still used
  elsewhere and was kept).

This explicitly **supersedes** §3.4's "leave the commented-out duplicate-detection code exactly
where it is" and §7's "do not uncomment or restore the duplicate-detection block" — both are
struck through above with a pointer here. The block is not restored or reactivated, it is deleted;
if it's ever needed again, it exists in git history prior to this change.

### 10.8 `identify_duplicates()` restored, but only for the 4 legacy surveys — Lionel's instruction, partially reverses §10.7

Lionel: *"I notice that the infrastructure_id is no longer created for the surveys
`fourrage_cultive`, `sous_projets_innovants`, `gestion_durable_des_paysages`, and
`activites_generatrices_de_revenus`. For these surveys, I want to revert back to the previous
configuration which calculated an infrastructure_id using information on the geolocation of the
point to figure out whether it should have the same infrastructure_id as another point. ... implement it
only for these 4 (no need to apply it to the fiche_simplifiee_infrastructures survey as there the
infrastructure_id already exists as it is generated within Kobo)."*

§10.7 deleted `identify_duplicates()` (and `haversine()`/`group_pairs()`/`reassign_ids()`) as dead
code without realizing it was the *only* source of `infrastructure_id` for these 4 surveys — after
§10.4 restored them, they had no Kobo-native identifier and no other code path assigned one
(confirmed: `drop_duplicates()`/`concatenate_snapshots()`'s §10.4 fallback logic was masking the
gap by skipping dedup entirely / falling back to `_id`, silently, rather than raising).

Restored verbatim from the pre-§10.7 version of `surveys.py` (all 4 functions, and the imports
that exist only to support them: `itertools.combinations`, `math.{asin, cos, radians, sin, sqrt}`,
`typing.{List, Sequence, Tuple}`). Unlike before, the call site in `transform_survey()` is now
conditional — a new `SURVEYS_WITH_GEO_ASSIGNED_ID` set (the 4 legacy survey names) — instead of
running unconditionally for every survey as it originally did. `fiche_simplifiee_infrastructures`
is deliberately excluded: its `INFRASTRUCTURE_ID` is computed inside the Kobo form itself (§4.1)
and is already folded into `infrastructure_id` earlier in `transform_survey()` per §10.1.

Verified against the real archived raw data for all 4 surveys: `infrastructure_id` is populated
(`UInt32`, matching the dtype found in the historical reference parquets), submissions within 1km
of each other now correctly share an ID (e.g. `fourrage_cultive`: 444 raw submissions → 275
distinct geo-assigned IDs), and `drop_duplicates()`'s existing fallback from §10.4 is simply never
triggered for these 4 anymore since the column now exists. The consolidated survey's `String`-typed
`infrastructure_id` is confirmed untouched.

§10.7's "the block is not restored or reactivated, it is deleted" is now inaccurate for these 4
surveys specifically — it is restored, deliberately, exactly where §10.7 said it wouldn't be. Read
§10.7 and this entry together, not §10.7 alone.

### 10.9 `fiche_simplifiee_infrastructures` gets files + DB output after all — Lionel's instruction, supersedes §3 decision 2

Lionel noticed `push()` logged `"infrastructures_hors_cdr: no gpkg file, skipping database
push"` and asked why — that turned out to be a stale message from an earlier run, not a bug (the
gpkg existed and was valid by the time of the actual check). In the course of that he asked:
*"also make sure that the equivalent outputs get generated also for the fiche_simplifiee_
infrastructures data."*

Asked which parts of "equivalent outputs" he meant — file artefacts + DB push only, or also an
OpenHexa dataset entry, or also a Geonode `PRAPS2_*` mirror table — Lionel confirmed **files + DB
push only**. This **supersedes** §3's decision 2 ("No `fiche_simplifiee_infrastructures` output"),
but only partially: it still gets no dataset entry and no Geonode mirror, since it's an
intermediate form with no historical `PRAPS2_*` layer, and publishing it as its own dataset/layer
would be redundant with the 5 tables already split out of it.

Implementation: in `transform()`, right after `drop_blocked_submissions` and before
`split_consolidated`, `_write_survey_outputs(consolidated_name, df, df_no_duplicates, output_dir)`
writes the same 5 artefacts (`surveys/{name}_with_duplicates.parquet`, `surveys/{name}.parquet`,
`surveys/{name}.xlsx`, `geo/{name}.gpkg`, `snapshots/{name}_snapshots.parquet`) for the full
consolidated frame that every split gets — reusing the same helper means the same empty/
no-geometry guards apply automatically. In `push()`, `consolidated_name` is appended to the list
of names pushed to PostGIS (table + `_snapshots` table); `DATASETS` and the Geonode `mapping` in
`push()` are deliberately left untouched.

Verified against the real downloaded consolidated survey (1203 raw rows, 1 blocked submission
dropped, 786 valid geometries): all 5 file artefacts write correctly.