# CLAUDE.md — PRAPS `extract-surveys` pipeline

Instructions and reference for Claude Code working on this repository.

---

## 1. Context

This is an [OpenHexa](https://github.com/BLSQ/openhexa) pipeline (`extract-surveys`) for the
**PRAPS-2** programme (World Bank / CILSS): it tracks pastoral infrastructures across 6 Sahel
countries. It downloads KoboToolbox survey submissions, transforms them, writes files to the
OpenHexa workspace, pushes tables to the OpenHexa PostgreSQL/PostGIS warehouse, and publishes
OpenHexa datasets.

Three files matter:

| File | Role |
| --- | --- |
| `pipeline.py` | OpenHexa pipeline definition + 4 tasks: `download`, `transform`, `push`, `update_datasets` |
| `surveys.py` | Helper library: survey download, per-survey constants, `transform_survey`, the split/reconstruction logic, `drop_duplicates`, `concatenate_snapshots` |
| `legacy_schema.py` | Data only: the pre-migration column schema of each reconstructed table, and the field-rename map that populates it |

**Downstream consumer that constrains everything below:** a **Geonode server** renders each
infrastructure table on a digital web map using **hard-coded column names** in its layer
configuration. Renaming or dropping a column breaks the map. `push()` therefore also mirrors most
tables into a legacy `PRAPS2_*` table that Geonode still reads.

**Related pipeline:** `sync_attachments` (sibling directory) reads this pipeline's
`surveys/{name}.parquet` outputs to find and re-upload photo attachments to GCS. It is not part of
this pipeline's scope, but its `SURVEYS` list should stay in sync with what's actually produced
here (see §9).

---

## 2. Architecture

Kobo collects pastoral infrastructure data through **one consolidated survey**,
`fiche_simplifiee_infrastructures` (uid `aRZ43wRerX8pCdo4XrKw4n`, form `id_string` `TBT`), which
merges what used to be 4 separate surveys plus a 5th infrastructure category that never had its
own survey. Alongside it, **4 unrelated surveys** never merged into the consolidated form and are
still collected and processed independently: `fourrage_cultive`, `sous_projets_innovants`,
`gestion_durable_des_paysages`, `activites_generatrices_de_revenus`.

```
pipeline.py
  LEGACY_SURVEYS      = 4 standalone surveys (uid, name) -- never merged into the consolidated form
  CONSOLIDATED_SURVEY = (uid, "fiche_simplifiee_infrastructures")
  SURVEYS             = LEGACY_SURVEYS + [CONSOLIDATED_SURVEY]   # drives download()
```

`download()` fetches all 5 surveys' raw data + field metadata, unchanged in shape from Kobo.

`transform()` then produces **10 output tables**, each getting the same 5 artefacts
(`surveys/{name}_with_duplicates.parquet`, `surveys/{name}.parquet`, `surveys/{name}.xlsx`,
`geo/{name}.gpkg`, `snapshots/{name}_snapshots.parquet`) via the shared `_write_survey_outputs()`
helper in `pipeline.py`:

| Output table | Source |
| --- | --- |
| `fourrage_cultive`, `sous_projets_innovants`, `gestion_durable_des_paysages`, `activites_generatrices_de_revenus` | Downloaded and transformed independently (`surveys.LEGACY_SURVEYS`) |
| `fiche_simplifiee_infrastructures` | The full consolidated survey, as downloaded (after dropping blocked/partial submissions, §3.3) |
| `marches_a_betail`, `parcs_de_vaccination`, `points_d_eau`, `unites_veterinaires` | Split out of the consolidated survey by `CDR` (§3.2), then reconstructed to their pre-migration column shape (§4) |
| `infrastructures_hors_cdr` | Split out of the consolidated survey by `HCDR` non-empty (§3.2) — new table, no predecessor |

`push()` writes every one of those 10 tables' `.gpkg` to PostGIS (plus a `{name}_snapshots` table
from the snapshots parquet), then mirrors 9 of them (everything except
`fiche_simplifiee_infrastructures` itself) into legacy `PRAPS2_*`-named tables for Geonode.

`update_datasets()` publishes an OpenHexa dataset for 9 of the 10 tables (everything except
`fiche_simplifiee_infrastructures`, which is treated as an intermediate form — see §7).

The 3 surveys **not** handled by this pipeline at all — `indicateurs_regionaux`,
`indicateurs_pays`, `cultures_vivrieres` — are out of scope; see §8.

---

## 3. The consolidated survey (`fiche_simplifiee_infrastructures`)

Source: XLSForm `FICHE_SIMPLIFIEE_INFRASTRUCTURES_final_config_v7.xlsx` (shipped in this pipeline
folder), form version `3107261400`, default language `french`. Always cross-check against the
actual downloaded parquet before relying on a field being present — Kobo omits any column no
submission has ever answered, and the deployed form may have moved on since this file was written.

### 3.1 Field inventory (in form order)

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
  calculate     IDT_DUP_COUNT  count of points_live rows whose INFRASTRUCTURE_ID = IDT
  note          IDT_DUP        relevant: IDT != '' and IDT_DUP_COUNT > 0
  acknowledge   IDT_DUP_BLOCK  required; relevant: IDT != '' and IDT_DUP_COUNT > 0
  text          STMB01         relevant: STMB0 = 2
  text          STMB02         relevant: STMB0 = 2
  text          STMB03         relevant: STMB0 = 2
  calculate     TYPE_ACRONYM   MB|PV|PE|UV for CDR 1-4, else HCDR acronym (see §3.2)

group2-1 — LOCATION        relevant: (IDT = '' or IDT_DUP_COUNT = 0)
  select_one    LUV1           Pays        (6 choices)
  select_one    LUV2           Region      (80)
  select_one    LUV3           Departement (359)
  select_one    LUV4           Commune     (2176)
  text          LUV5
  select_one_from_file points_live.csv  LUV6_existing   relevant: STMB0 = 2
  geopoint      LUV6_manual    relevant: STMB0 = 1
  calculate     LUV6           = LUV6_existing or LUV6_manual, as a plain string (not a struct)
  calculate     NEARBY_RADIUS  constant 1500 (metres)
  calculate     NEARBY_COUNT   nearby points_live entries, same LUV3 + same TYPE_ACRONYM
  calculate     NEARBY_IDS     comma-joined INFRASTRUCTURE_IDs of those
  calculate     BASE_ID        TYPE_ACRONYM + rounded lat/lon
  calculate     DUP_COUNT
  calculate     INFRASTRUCTURE_ID   = IDT, else LUV6_existing, else BASE_ID[_DUP_COUNT]
  note          IDT2
  calculate     NEARBY_MSG     warning text built from NEARBY_COUNT / NEARBY_IDS
  note          IDT3           relevant: STMB0 = 1 and NEARBY_COUNT > 0

group3 — SUIVI DES TRAVAUX  relevant: (IDT = '' or IDT_DUP_COUNT = 0)
  select_one    STMB1          relevant: STMB0 = 2
  date          STMB3          relevant: STMB0 = 2
  integer       STMB4          relevant: STMB0 = 2
  select_one    STMB5          works state (7 choices)
  date          STMB11         relevant: STMB5 = 5
  date          STMB12         relevant: STMB5 = 6
  date          STMB13         relevant: STMB5 = 7
  calculate     STMB14
  select_one    STMB15         progress bracket (7 choices)
  note          CONFSITE
  select_one    CONFSITE1      (4 choices)
  select_one    DATTROL        relevant: IDUV2 != 4  (6 choices)
  text          CONFSITE2
  group3-1 — relevant: HCDR in (5, 9, 10)   [linear infrastructures]
    geotrace    STMB16
    calculate   STMB17
    calculate   STMB18
    select_one  note4

group3-2 — PHOTOS          relevant: (IDT = '' or IDT_DUP_COUNT = 0)
  note          LUV7
  image         LUV7a
  image         LUV7b
  image         LUV7c
  image         LUV7d
```

`LUV6` is a `calculate` field holding a plain string, **not** a real `geopoint` — unlike the old
per-survey forms, where the equivalent field genuinely was a geopoint (a `Struct` once cast).
`transform_survey()`'s `level_7` handling (aliased from `LUV6` via `GEO_COLUMNS`) accounts for
this: it only calls `.struct.json_encode()` when `level_7` is actually a `pl.Struct`, and leaves it
as-is otherwise.

### 3.2 Split logic — `CDR` / `HCDR` / `TYPE_ACRONYM`

`TYPE`: `1` = *CDR*, `2` = *HORS CDR*.

`CDR` (drives the 4 legacy-table splits):

| code | label | table |
| --- | --- | --- |
| 1 | `Marché à bétail` | `marches_a_betail` |
| 2 | `Parc de vaccination` | `parcs_de_vaccination` |
| 3 | `Point d'eau` | `points_d_eau` |
| 4 | `Unité vétérinaire` | `unites_veterinaires` |

`HCDR` (13 choices, all routing to `infrastructures_hors_cdr`; `HCDR` non-empty is the split rule
— `TYPE` is not additionally checked): `Aires d'abattage`, `Centres collecte/Mini Laiteries`,
`Etals de boucherie`, `Magasins aliment betail`, `Couloirs balisés (en Km)`, `Quai
d'embarquement/débarquement`, `Aire de quarantaine`, `Unité de tannerie`, `Couloirs de
transhumances sécuriés des zones conflictuelles`, `Réalisation des bandes des pare feux dans les
zones de hautes prairies d'herbes`, `Sécurisation des aires de stationnement`, `Construction des
cordons pierreux dans les bas fonds`, `Autre (à Préciser)`.

`openhexa.toolbox.kobo.utils.to_dataframe()` returns the choice **label**, not the raw code, for
`select_one` fields (confirmed by reading `cast_select_one()` in the installed toolbox, and by
inspecting real downloaded data). `split_consolidated()` in `surveys.py` (`CDR_INFRA_LIST`)
matches on **either** code or label regardless, so it stays correct if that ever changes.

`TYPE_ACRONYM` is a stable, accent-free, single-token cross-check embedded in `INFRASTRUCTURE_ID`
via `BASE_ID`:

```
CDR  1 → MB    2 → PV    3 → PE    4 → UV
HCDR 1 → AA    2 → CCML  3 → EB    4 → MAB   5 → CB    6 → QED   7 → AQ
     8 → UT    9 → CTSZC 10 → RBPF 11 → SAS  12 → CCP  13 → AUTRE (fallback)
```

`split_consolidated()` asserts every row of each CDR table has the matching `TYPE_ACRONYM`, and
every row of `infrastructures_hors_cdr` has an HCDR acronym, `log_warning`-ing (not failing) on any
mismatch — this also catches historically migrated rows whose `TYPE_ACRONYM` was assigned by an
older process and doesn't match this table (e.g. Mali submissions carrying `LHV`/`CU`, assigned
during the `add_infrastructure_id` migration, not by this form).

A row matching no split, or more than one, is `log_warning`'d with a sample of `_id` values rather
than silently dropped or duplicated.

### 3.3 Blocked / partial submissions

When an enumerator types an identifier at `IDT` that already exists (`IDT_DUP_COUNT > 0`), the
form skips ahead and the submission is saved with only `group1` + `group2` answered: no
`LUV1..LUV6`, no `_geolocation`, no `INFRASTRUCTURE_ID`, no `STMB*`, no photos. These are
data-entry rejects, not real infrastructures.

`drop_blocked_submissions()` removes them — rows where `infrastructure_id` is null or blank —
**before** dedup and before the split (`pipeline.py`'s `transform()` calls it on both the
with-duplicates and deduped frames right after `transform_survey()`, before
`_write_survey_outputs()` and before `split_consolidated()`). This must run before dedup: two
blocked rows both have a null `infrastructure_id`, and `.unique(subset=...)` treats null == null,
so without this filter they'd collapse into a single row and contaminate `concatenate_snapshots()`.

---

## 4. Reconstructing the 4 legacy CDR tables

`marches_a_betail`, `parcs_de_vaccination`, `points_d_eau`, `unites_veterinaires` must each expose
**exactly the same column names they had before** the consolidation — including columns the
consolidated survey no longer collects (kept as typed nulls) — plus whatever new columns the
consolidated survey adds. Extra columns are fine (Geonode ignores what it doesn't reference);
missing or renamed columns break the map.

The consolidated survey uses **one field name per concept** where the 4 old surveys each used
their own prefixed names (location is `LUV1..LUV6` in the consolidated form, but was `LMB1..LMB6`
in `marches_a_betail`, `LPE1..LPE5/LPE7` in `points_d_eau`, `LVAC1..LVAC6` in
`parcs_de_vaccination`; likewise `STMB*` vs `STPE*`/`STVAC*`/`STUV*`).

### 4.1 `legacy_schema.py`

Two literal constants, read at import time only (never re-derived at run time):

- **`LEGACY_COLUMNS`**: `{table_name: {column_name: polars_dtype}}`, in original column order.
  Derived from the last-generated reference parquet for each of the 4 tables (schema + order only,
  via `pl.scan_parquet(...).collect_schema()`) — this is the mandatory, Geonode-facing column set.
  One deliberate deviation: `infrastructure_id` is typed `pl.String` in every table, not the
  `pl.UInt32` found in the reference parquets — see §7's case-collision rule for why.
- **`COLUMN_RENAMES`**: `{table_name: {consolidated_field: legacy_field}}`. Derived from
  `all_forms_cols_mapping` in the sibling `add_infrastructure_id` pipeline's `config.py` (read-only
  input, never modified from here) — that dict maps legacy field name → consolidated field name;
  the entries here are the inverse. Identity mappings and empty (no-consolidated-equivalent)
  targets are dropped. `infrastructures_hors_cdr` has no `LEGACY_COLUMNS` entry (no predecessor
  table) and an empty `COLUMN_RENAMES` entry — it keeps plain consolidated field names throughout.

If either source needs re-deriving in the future (e.g. a 5th legacy table gets added), the method
is: read the reference parquet's schema for the mandatory column set, invert
`all_forms_cols_mapping` for the rename map, and cross-check the result against the reference
parquet — do not reconstruct either from field-name prefixes or intuition, and do not assume the
mapping is internally consistent with the parquet (one entry, for `unites_veterinaires`'s
`IDUV2A`, was found to disagree with the real reference parquet's own casing and was dropped rather
than applied).

### 4.2 `conform_to_legacy_schema(df, name)`

Three ordered steps:

1. **Rename** — apply `COLUMN_RENAMES[name]`, only for keys actually present in `df`. If a rename
   target already exists in `df` (collision), keep the existing column and `log_warning` rather
   than overwrite.
2. **Fill** — every column in `LEGACY_COLUMNS[name]` still missing after the rename becomes an
   all-null column cast to its declared dtype. `log_info`s the filled list.
3. **Order and keep** — legacy columns first, in their original reference-parquet order, then
   consolidated-only extras. Never drops a column.

For columns present in both: cast to the legacy dtype if safe; if not (e.g. a `select_multiple`
that casts to `List(String)` being asked to become legacy `String`), keep the source dtype and
`log_warning` rather than raise — in practice this rarely fires, since `transform_survey()`
already JSON-encodes struct/list columns to plain strings before the split happens. Post-condition:
`set(LEGACY_COLUMNS[name]) <= set(out.columns)`, asserted in code.

---

## 5. The 4 standalone legacy surveys

`fourrage_cultive`, `sous_projets_innovants`, `gestion_durable_des_paysages`,
`activites_generatrices_de_revenus` are unrelated to the consolidation: they were never merged into
`fiche_simplifiee_infrastructures`, are downloaded and transformed independently
(`surveys.LEGACY_SURVEYS` in `pipeline.py`), and never go through
`split_consolidated`/`conform_to_legacy_schema`/`drop_blocked_submissions` — those are
consolidated-survey-specific. `PICTURES` and `GEO_COLUMNS` in `surveys.py` have entries for all 4,
keyed by survey name, applied by the same `transform_survey()` every survey goes through.

**They have no Kobo-native infrastructure identifier** — unlike the consolidated survey, which
computes `INFRASTRUCTURE_ID` inside the form itself. For these 4 (and only these 4,
`SURVEYS_WITH_GEO_ASSIGNED_ID` in `surveys.py`), `transform_survey()` calls `identify_duplicates()`
to assign one based on geographic proximity: points within `min_distance` km of each other
(rounded-coordinate bucketing, then pairwise haversine distance within each bucket) share an ID,
via `with_row_index()` + `group_pairs()`/`reassign_ids()`. This runs before the internal
`drop_duplicates()` call, which then dedups on the ID it just produced. It is **not** applied to
`fiche_simplifiee_infrastructures` — that survey's `INFRASTRUCTURE_ID` (renamed to
`infrastructure_id`, see §7) already comes from Kobo and needs no reassignment.

`drop_duplicates()` and `concatenate_snapshots()` both degrade gracefully rather than raise when
`infrastructure_id` is absent for a survey (skip dedup / fall back to Kobo's own `_id`) — this
matters only as a defensive fallback now that these 4 always get one via `identify_duplicates()`.

---

## 6. What each task does

- **`download`**: for every entry of `SURVEYS` (`LEGACY_SURVEYS` + `CONSOLIDATED_SURVEY`),
  downloads survey data to `raw/{name}.parquet` and field metadata to
  `metadata/{name}_fields.{parquet,xlsx}`.
- **`transform`**: for each of the 4 `LEGACY_SURVEYS`, runs `transform_survey()` and writes its
  outputs directly. For the consolidated survey: runs `transform_survey()` once, drops blocked
  submissions, writes its own outputs (§7), then `split_consolidated()` + `conform_to_legacy_schema()`
  per entry of `OUTPUT_TABLES`, writing each split's outputs. All writes go through
  `_write_survey_outputs()`, which also fills `_tags`/`_notes` (§7) and guards empty/no-geometry
  splits: an empty split, or one with no valid geometry, `log_warning`s and skips its gpkg/snapshots
  write rather than failing the run or overwriting a populated table with nothing.
- **`push`**: PostGIS-pushes every one of the 10 tables' `.gpkg` (plus its `_snapshots` table where
  present), then mirrors 9 of them (all but `fiche_simplifiee_infrastructures`) into `PRAPS2_*`
  tables for Geonode, each wrapped in `try`/`except` so one bad mirror can't abort the task.
- **`update_datasets`**: publishes an OpenHexa dataset per entry of `DATASETS` (9 tables, not
  `fiche_simplifiee_infrastructures`). The 5 consolidated-derived tables share one
  `metadata/fiche_simplifiee_infrastructures_fields.xlsx`; the 4 standalone legacy surveys keep
  their own per-survey fields file, since they're still downloaded independently. An entry whose
  `dataset_uid` starts with `"TODO"` is skipped with `log_info` rather than crashing the task — the
  mechanism for wiring up a new table's dataset before its UID exists.

---

## 7. Cross-cutting rules

- **Never let two output columns differ only by case.** GDAL's GeoPackage driver rejects it
  outright (`FieldError: Error adding field '...' to layer`) — confirmed by reproducing it directly
  with a 2-column GeoDataFrame. This is why `transform_survey()` renames the consolidated survey's
  `INFRASTRUCTURE_ID` to lowercase `infrastructure_id` immediately after computing
  `LATITUDE`/`LONGITUDE` (before anything else touches it), rather than carrying both the legacy
  tables' dead `infrastructure_id` placeholder and the consolidated survey's real
  `INFRASTRUCTURE_ID` as separate columns. Any future rename touching an existing column name
  should be checked against this.
- **`fiche_simplifiee_infrastructures` gets the same file + DB-push treatment as every other
  table**, but deliberately **no OpenHexa dataset and no Geonode mirror** — it's an intermediate
  form with no historical `PRAPS2_*` layer, and publishing it as its own dataset/layer would be
  redundant with the 5 tables already split out of it.
- **`_tags`/`_notes`**: Geonode's layer config expects these two Kobo system columns on every
  table except `parcs_de_vaccination` (that table's submission history never included a tagged or
  annotated row, so the column never existed for it — not a bug). Both are Kobo system columns
  that `to_dataframe()` only emits when at least one submission in the batch has one set; when
  absent, `_write_survey_outputs()` adds them as null `pl.String` for every table except
  `parcs_de_vaccination` (`GEONODE_EXTRA_COLUMNS` / `TABLES_WITHOUT_GEONODE_EXTRA_COLUMNS`).
- **Never rename or drop a column of the 4 legacy CDR tables or `infrastructures_hors_cdr`** once
  it's in their output — additive only. Don't change output table names, the `data/kobo/`
  directory layout, the pipeline id `extract-surveys`, its 3 parameters
  (`output_dir`/`push_to_db`/`overwrite`), existing dataset UIDs, or `PRAPS2_*` mirror table names.
- **Don't add runtime database introspection or read the reference parquets / the
  `add_infrastructure_id` pipeline's `config.py` at pipeline run time** — §4.1's constants are the
  encoded result; re-derive them (by hand, reading the actual sources) only if the underlying data
  changes, never algorithmically from prefixes or labels.
- **Don't modify `add_infrastructure_id`** — it's read-only input to this pipeline (`config.py`'s
  `all_forms_cols_mapping`), owned by a sibling pipeline.

---

## 8. Out of scope

`indicateurs_regionaux`, `indicateurs_pays`, `cultures_vivrieres` are not downloaded, transformed,
pushed, or published by this pipeline. Their old UIDs remain as commented-out entries in
`pipeline.py`'s `LEGACY_SURVEYS` for reference, not for reactivation.

---

## 9. Downstream dependents

`sync_attachments` (sibling pipeline) reads `surveys/{name}.parquet` for each name in its own
`SURVEYS` list and re-uploads any photo attachments found in `_attachments` to GCS. It currently
lists `fiche_simplifiee_infrastructures` + the 4 standalone legacy surveys — not the 4 split CDR
tables or `infrastructures_hors_cdr`, since their rows are already covered by
`fiche_simplifiee_infrastructures` itself (splitting doesn't duplicate or lose any row). One gotcha
worth knowing if that pipeline is touched again: most `fiche_simplifiee_infrastructures`
submissions have zero photo attachments (unlike the old dedicated forms, where a photo was
effectively always present) — `serialize()` here represents an empty attachments list as `null`,
not `"[]"`, so any code reading that column must guard for `None` before `json.loads`-ing it.

---

## 10. Verification approach

No Kobo credentials or warehouse access in the dev sandbox — verify offline:

1. `python -m py_compile pipeline.py surveys.py legacy_schema.py`; `pyflakes` (or `ruff check` /
   `ruff format --diff` if available) on all three, and a check for any remaining reference to the
   3 out-of-scope surveys (§8) in executed code paths.
2. A synthetic fixture covering: one row per `CDR` value, non-empty-`HCDR` rows, a row matching no
   split, a duplicate `infrastructure_id` with two different `DATE`s, a null `_geolocation` row, a
   blocked/partial submission, and a variant frame with the `LUV*`/`STMB*`/`LUV7*` columns absent
   entirely (what a low-submission-count batch looks like) — run through
   `transform_survey → drop_blocked_submissions → split_consolidated → conform_to_legacy_schema`
   and assert: the blocked row is dropped and never reaches a split; every split has the expected
   row count with no row in two splits; `TYPE_ACRONYM` is consistent per split; each legacy table's
   mandatory columns are present, in original order, matching the real reference parquet (not a
   retyped copy of it); renamed fields carried real data (not just correctly-named nulls); missing
   legacy columns are all-null with the declared dtype; consolidated-only extras are present; dedup
   kept the most recent `DATE`; an empty split warns without raising and writes no gpkg; the
   missing-columns variant runs end to end.
3. Wherever feasible, re-run the same steps against the real last-downloaded raw parquet for every
   survey (further exercises real-world data shape that a synthetic fixture won't always capture —
   this is how the GDAL case-collision and the `_attachments`-is-`null` issues were actually found).
4. Keep verification scripts as throwaways under `/tmp`, not part of the repo — this pipeline has
   no test suite, and scope is `pipeline.py`/`surveys.py`/`legacy_schema.py` only; no opportunistic
   refactors or reformatting of code that didn't need to change.
