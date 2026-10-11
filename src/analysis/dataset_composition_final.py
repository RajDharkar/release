"""
READ-ONLY authoritative dataset composition report for the paper's data
table (EACL Industry Track). No training, no generation, no model calls,
no scoring-logic changes -- this script only reads existing canonical
files and computes counts/distributions with pandas (never wc -l, which
misreports row counts on these files due to embedded newlines in text
fields).

CANONICAL FILE MAP (see reports/dataset_composition_final.md for the full
per-cell rationale):
  AAVE fiction HWT   : data/interim/aave_book_hwt_clean_w280.csv (seed-overlap removed)
                       pre-exclusion: data/interim/aave_book_w280.csv
  SAE  fiction HWT   : data/interim/sae_fiction_hape_w280.csv (HAP-E chunk_2 only, re-windowed @280)
  AAVE poetry HWT    : data/processed/aave_poetry.csv, variety=='AAVE' (excludes 5 Gullah rows)
  SAE  poetry HWT    : data/processed/sae_poetry.csv (NEW dedicated PD-poetry-era-matched source;
                       supersedes the LiteraryTaste-based SAE poetry used in earlier detection
                       experiments this session -- flagged explicitly in the report)
  AAVE lyrics HWT    : data/processed/aave_lyric.csv (not redistributable, gitignored, counted in-repo only)
  SAE  oratory HWT   : data/interim/sae_oratory_w225.csv, needs_review==False
  SAE  personal_narrative HWT/MGT : data/processed/unified.csv (genre=='personal_narrative', variety=='SAE')
  Fiction/Poetry/Oratory MGT : the three confound-controlled 3-model regeneration files
  Arm C (HGT over human SAE) : data/processed/arm_c_mv_perturbation.csv
  Arm D (HGT over machine SAE): the three per-genre arm_d_*.csv files

EXCLUDED, reported but NOT counted in the benchmark:
  - unified.csv AAVE short_fiction/personal_narrative rows (311, flagged needs_review_reason)
  - unified.csv's own browndw short_fiction SAE-human pool (includes chunk_1, violates the
    chunk_1-exclusion decision -- confirmed here, an independent reason beyond supersession)
  - LiteraryTaste fiction (3,117 rows) -- token-length band mismatch (max=201 vs the
    200-388 AAVE-HWT/MGT range), structurally unusable regardless of any experiment's choices
  - w120 and w225 versions of AAVE fiction / SAE oratory -- superseded by w280 / w225-is-still-
    current-for-oratory (oratory was NOT re-windowed to 280, only fiction was; see report)
  - AAVE personal narrative -- ruled out entirely (2026-07-17 decision: spoken-vs-written
    register confound vs the SAE personal-narrative comparison side)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM = REPO_ROOT / "data/interim"
PROCESSED = REPO_ROOT / "data/processed"
REPORT_PATH = REPO_ROOT / "reports/dataset_composition_final.md"

SMALL_CELL_THRESHOLD = 200


def wc(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split().str.len()


def token_dist(series: pd.Series) -> dict:
    if len(series) == 0:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    w = wc(series)
    q = w.quantile([0, .25, .5, .75, 1.0])
    return {"min": int(q[0]), "p25": int(q[.25]), "median": int(q[.5]), "p75": int(q[.75]), "max": int(q[1.0])}


def fmt_dist(d: dict) -> str:
    if d["min"] is None:
        return "n/a"
    return f"{d['min']}/{d['p25']}/{d['median']}/{d['p75']}/{d['max']}"


def cell(label, variety, genre, raw_n, usable_n, unit, window, source_file, dist, n_docs, note=""):
    return {"label": label, "variety": variety, "genre": genre, "raw_n": raw_n, "usable_n": usable_n,
            "unit": unit, "window": window, "source_file": source_file, "dist": dist, "n_docs": n_docs, "note": note}


def main():
    cells = []
    empty_cells = []
    exclusions = []

    print("=" * 70)
    print("DATASET COMPOSITION AUDIT (read-only)")
    print("=" * 70)

    # ================= AAVE fiction HWT =================
    fic_aave_pre = pd.read_csv(INTERIM / "aave_book_w280.csv", low_memory=False)
    fic_aave = pd.read_csv(INTERIM / "aave_book_hwt_clean_w280.csv", low_memory=False)
    cells.append(cell("HWT", "AAVE", "fiction", len(fic_aave_pre), len(fic_aave),
                       "windowed passage", "target=280,min=200",
                       "data/interim/aave_book_hwt_clean_w280.csv (pre-exclusion: aave_book_w280.csv)",
                       token_dist(fic_aave["text"]), fic_aave["book_id"].nunique()))
    exclusions.append("AAVE fiction HWT, w120 (data/processed/aave_book.csv, 27,278 rows) and w225 "
                       "(data/interim/aave_book_hwt_clean_w225.csv, 10,084 rows) -- both superseded by "
                       "w280; w225 nominally overlapped fiction MGT's length band but its length-only "
                       "AUC was 1.00 (see reports/detection_v2_length_diagnostic.md), a confound w280 fixes.")

    # ================= SAE fiction HWT =================
    fic_sae = pd.read_csv(INTERIM / "sae_fiction_hape_w280.csv", low_memory=False)
    cells.append(cell("HWT", "SAE", "fiction", len(fic_sae), len(fic_sae),
                       "windowed passage", "target=280,min=200 (over HAP-E fic_chunk_2 only)",
                       "data/interim/sae_fiction_hape_w280.csv",
                       token_dist(fic_sae["text"]), fic_sae["pair_id"].nunique()))

    u = pd.read_csv(PROCESSED / "unified.csv", low_memory=False)
    lt_fic = u[(u.source_dataset == "mj-storytelling/LiteraryTaste") & (u.label == "human") & (u.genre == "short_fiction")]
    lt_fic_wc = wc(lt_fic["text"])
    exclusions.append(f"LiteraryTaste fiction (unified.csv, source_dataset='mj-storytelling/LiteraryTaste', "
                       f"genre='short_fiction', label='human'): {len(lt_fic)} rows -- EXCLUDED, token-length "
                       f"band mismatch (max={int(lt_fic_wc.max())} words vs the AAVE-HWT/MGT range of "
                       f"~200-388; a hard, unbridgeable gap, not a modeling choice -- see the SAE-fiction "
                       f"exclusion ruling, reports/detection_v2_report.md).")
    hape_c1c2 = u[(u.source_dataset == "browndw/human-ai-parallel-corpus") & (u.label == "human") &
                  (u.genre == "short_fiction")]
    exclusions.append(f"unified.csv's OWN browndw/HAP-E short_fiction human pool ({len(hape_c1c2)} rows): "
                       f"includes BOTH fic_chunk_1 (1,395) and fic_chunk_2 (1,395) -- chunk_1 is the "
                       f"seed/prompt text given to the generating model, and its inclusion here violates "
                       f"the 2026-07-17 chunk_1-exclusion decision. This is an independent, verified data-"
                       f"quality reason (not just supersession) that unified.csv must not be used for the "
                       f"SAE fiction HWT cell -- use sae_fiction_hape_w280.csv instead.")
    exclusions.append("unified.csv AAVE short_fiction/human (296 rows) and personal_narrative/AAVE/human "
                       "(15 rows) -- 311 total, explicitly flagged via needs_review_reason "
                       "(2026-08-01 decision); zero id overlap with the current 49-book pipeline.")

    # ================= AAVE poetry HWT =================
    poetry_a_full = pd.read_csv(PROCESSED / "aave_poetry.csv", low_memory=False)
    poetry_aave = poetry_a_full[poetry_a_full.variety == "AAVE"]
    gullah = poetry_a_full[poetry_a_full.variety == "Gullah"]
    cells.append(cell("HWT", "AAVE", "poetry", len(poetry_a_full), len(poetry_aave),
                       "whole document", "n/a (unwindowed)",
                       "data/processed/aave_poetry.csv, variety=='AAVE'",
                       token_dist(poetry_aave["text"]), len(poetry_aave)))
    exclusions.append(f"aave_poetry.csv also contains {len(gullah)} Gullah-variety rows -- excluded from "
                       f"the AAVE cell (Gullah is a distinct creole, not AAVE).")

    # ================= SAE poetry HWT =================
    poetry_s_full = pd.read_csv(PROCESSED / "sae_poetry.csv", low_memory=False)
    cells.append(cell("HWT", "SAE", "poetry", len(poetry_s_full), len(poetry_s_full),
                       "whole document", "n/a (unwindowed)",
                       "data/processed/sae_poetry.csv",
                       token_dist(poetry_s_full["text"]), len(poetry_s_full)))
    old_lt_poetry = u[(u.source_dataset == "mj-storytelling/LiteraryTaste") & (u.label == "human") & (u.genre == "poetry")]
    exclusions.append(f"OLD SAE poetry source (unified.csv LiteraryTaste poetry, {len(old_lt_poetry)} rows): "
                       f"SUPERSEDED by data/processed/sae_poetry.csv (era-matched PD poetry, same 4 source "
                       f"anthologies as the AAVE poetry cell: dunbar/hughes/johnson1922/henderson). "
                       f"**Note**: Experiments B/C/D earlier this session used the OLD LiteraryTaste-based "
                       f"SAE poetry pool (283 rows) as their SAE-poetry source, since sae_poetry.csv did not "
                       f"exist as a canonical source at that point in this session's history -- this "
                       f"composition report reflects the CURRENT on-disk state, which has since evolved.")

    # ================= AAVE lyrics HWT =================
    lyrics = pd.read_csv(PROCESSED / "aave_lyric.csv", low_memory=False)
    lyrics_valid = lyrics[lyrics["text"].notna()]
    cells.append(cell("HWT", "AAVE", "lyrics", len(lyrics), len(lyrics_valid),
                       "whole document (song)", "n/a (unwindowed)",
                       "data/processed/aave_lyric.csv (NOT redistributable, gitignored -- in-repo count only)",
                       token_dist(lyrics_valid["text"]), lyrics_valid["artist"].nunique()))
    empty_cells.append(("MGT", "SAE", "lyrics", "no lyrics MGT pipeline was ever built for this benchmark."))
    empty_cells.append(("HGT", "AAVE", "lyrics", "no Arm C/D perturbation was ever run over a lyrics source."))
    empty_cells.append(("HWT", "SAE", "lyrics", "no SAE-lyrics comparison corpus exists or was sourced."))

    # ================= SAE oratory HWT =================
    orat_full = pd.read_csv(INTERIM / "sae_oratory_w225.csv", low_memory=False)
    orat_elig = orat_full[orat_full.needs_review == False]  # noqa: E712
    cells.append(cell("HWT", "SAE", "oratory", len(orat_full), len(orat_elig),
                       "windowed passage", "target=225,min=150",
                       "data/interim/sae_oratory_w225.csv, needs_review==False",
                       token_dist(orat_elig["text"]), orat_elig["speech_id"].nunique()))
    exclusions.append(f"sae_oratory.csv (data/processed, 120w windows, 34,719 rows) -- superseded by "
                       f"sae_oratory_w225.csv (oratory was re-windowed to 225, NOT 280 -- only fiction "
                       f"needed the 280 target; oratory's own MGT distribution overlapped the 225 spike "
                       f"well enough, see reports/rewindow_225.md-era findings).")
    empty_cells.append(("HWT", "AAVE", "oratory", "no AAVE oratory/speech corpus exists or was sourced."))

    # ================= SAE personal_narrative HWT/MGT =================
    pn_sae_h = u[(u.genre == "personal_narrative") & (u.variety == "SAE") & (u.label == "human")]
    cells.append(cell("HWT", "SAE", "personal_narrative", len(pn_sae_h), len(pn_sae_h),
                       "whole document (2-chunk split)", "n/a (unwindowed, HAP-E's own chunking)",
                       "data/processed/unified.csv, genre=='personal_narrative' & variety=='SAE' & label=='human'",
                       token_dist(pn_sae_h["text"]), pn_sae_h["pair_id"].nunique()))
    pn_sae_llm = u[(u.genre == "personal_narrative") & (u.variety == "SAE") & (u.label == "llm")]
    cells.append(cell("MGT", "SAE", "personal_narrative", len(pn_sae_llm), len(pn_sae_llm),
                       "whole document", "n/a",
                       "data/processed/unified.csv, genre=='personal_narrative' & variety=='SAE' & label=='llm'",
                       token_dist(pn_sae_llm["text"]), pn_sae_llm["pair_id"].nunique(),
                       note="INHERITED, uncontrolled multi-generator pool (6 different upstream models: "
                            "gpt-4o-2024-08-06, gpt-4o-mini-2024-07-18, Meta-Llama-3-70B-Instruct, "
                            "Meta-Llama-3-70B, Meta-Llama-3-8B-Instruct, Meta-Llama-3-8B) -- NOT part of "
                            "the confound-controlled 3-model regeneration effort used for fiction/poetry/"
                            "oratory MGT. Do not compare directly against those cells."))
    empty_cells.append(("HWT", "AAVE", "personal_narrative", "ruled out entirely (2026-07-17 decision): "
                        "available AAVE personal-narrative material is overwhelmingly transcribed speech, "
                        "while the SAE comparison side (HAP-E blog) is written register -- using both would "
                        "silently confound genre with register. The 15 unified.csv AAVE/personal_narrative "
                        "rows that DO exist are explicitly flagged stale/superseded, not a live source."))
    empty_cells.append(("MGT", "AAVE", "personal_narrative", "no AAVE-seeded personal-narrative MGT was "
                        "ever built (consistent with there being no AAVE personal-narrative HWT to seed it)."))
    empty_cells.append(("HGT", "AAVE", "personal_narrative", "Arm C DOES include a personal_narrative slice "
                        "(1,526 rows, perturbed over the SAE HAP-E blog human pool) -- reported under Arm C "
                        "below. This is the one case where an AAVE-labeled artifact exists for this genre "
                        "despite there being no genuine AAVE personal-narrative HWT: Arm C is a RULE-"
                        "PERTURBED derivative of SAE text, never labeled HWT, so it does not contradict the "
                        "ruling above."))

    # ================= MGT: fiction/poetry/oratory, per generator =================
    mgt_specs = [
        ("fiction", INTERIM / "aave_fiction_mgt_modeb_run2.csv", None),
        ("poetry", INTERIM / "aave_poetry_mgt_modeb_run1.csv", None),
        ("oratory", INTERIM / "aave_oratory_mgt_modea_run1.csv", "topic_flagged"),
    ]
    mgt_by_generator = {}
    for genre, path, extra_flag in mgt_specs:
        df = pd.read_csv(path, low_memory=False)
        bad = df["completion"].isna() | (df["completion"].astype(str).str.strip() == "") | \
              (df["error"].fillna("").astype(str).str.strip() != "") | (df["refusal_bool"] == True)  # noqa: E712
        if extra_flag:
            bad = bad | (df[extra_flag] == True)  # noqa: E712
        valid = df[~bad]
        id_col = "source_book_id" if genre == "fiction" else ("source_poem_id" if genre == "poetry" else "source_speech_id")
        cells.append(cell("MGT", "SAE", genre, len(df), len(valid),
                           "whole document (rewrite of one AAVE source)", "n/a",
                           str(path.relative_to(REPO_ROOT)),
                           token_dist(valid["completion"]), valid[id_col].nunique()))
        mgt_by_generator[genre] = valid.groupby("model_short_name").size().to_dict()

    # ================= Arm C (HGT over human SAE) =================
    arm_c = pd.read_csv(PROCESSED / "arm_c_mv_perturbation.csv", low_memory=False)
    ARM_C_GENRE_MAP = {"short_fiction": "fiction", "oratory": "oratory",
                        "personal_narrative": "personal_narrative", "poetry": "poetry"}
    arm_c_by_genre = {}
    for raw_genre, mapped_genre in ARM_C_GENRE_MAP.items():
        sub = arm_c[arm_c.genre == raw_genre]
        arm_c_by_genre[mapped_genre] = {
            "n": len(sub), "n_distinct_pair_id": sub["pair_id"].nunique() if sub["pair_id"].notna().any() else None,
            "dist": token_dist(sub["text"]),
        }

    # ================= Arm D (HGT over machine SAE), per genre =================
    arm_d_specs = [
        ("fiction", INTERIM / "arm_d_aave_mgt_run1.csv"),
        ("poetry", INTERIM / "arm_d_poetry_aave_mgt_run1.csv"),
        ("oratory", INTERIM / "arm_d_oratory_aave_mgt_run1.csv"),
    ]
    arm_d_by_genre = {}
    for genre, path in arm_d_specs:
        df = pd.read_csv(path, low_memory=False)
        bad = df["aave_mgt_text"].isna() | df["sae_mgt_text"].isna()
        valid = df[~bad]
        arm_d_by_genre[genre] = {
            "raw": len(df), "usable": len(valid),
            "dist_aave": token_dist(valid["aave_mgt_text"]),
            "dist_sae_source": token_dist(valid["sae_mgt_text"]),
        }

    # ================= Totals =================
    total_usable = sum(c["usable_n"] for c in cells)
    hgt_total_c = sum(v["n"] for v in arm_c_by_genre.values())
    hgt_total_d = sum(v["usable"] for v in arm_d_by_genre.values())
    total_with_hgt = total_usable + hgt_total_c + hgt_total_d

    print(f"\nTotal HWT+MGT usable rows (excl. Arm C/D): {total_usable}")
    print(f"Arm C total: {hgt_total_c}, Arm D total: {hgt_total_d}")
    print(f"Grand total incl. HGT probe rows: {total_with_hgt}")

    # ============================================================
    # Build the markdown report
    # ============================================================
    lines = ["# Dataset composition -- final authoritative counts\n",
             "READ-ONLY audit for the paper's data table (EACL Industry Track). All counts "
             "computed with pandas (never `wc -l`, which misreports these files' row counts "
             "due to embedded newlines in text fields). No training, generation, or scoring "
             "changes were made this session.\n"]

    lines.append("## Full crossed table: LABEL x VARIETY x GENRE\n")
    lines.append("| label | variety | genre | raw n | usable n | unit | window | distinct source docs | "
                 "token dist (min/p25/median/p75/max) | flag |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        flag = ""
        if c["usable_n"] < SMALL_CELL_THRESHOLD:
            flag = f"**SMALL CELL (<{SMALL_CELL_THRESHOLD})**"
        lines.append(f"| {c['label']} | {c['variety']} | {c['genre']} | {c['raw_n']} | {c['usable_n']} | "
                     f"{c['unit']} | {c['window']} | {c['n_docs']} | {fmt_dist(c['dist'])} | {flag} |")
        if c["note"]:
            lines.append(f"|  |  |  | *{c['note']}* |||||||")
    lines.append("")

    lines.append("## MGT by generator (fiction / poetry / oratory)\n")
    lines.append("| genre | qwen3 | llama | deepseek | total usable |")
    lines.append("|---|---|---|---|---|")
    for genre, counts in mgt_by_generator.items():
        q, l, d = counts.get("qwen3", 0), counts.get("llama", 0), counts.get("deepseek", 0)
        lines.append(f"| {genre} | {q} | {l} | {d} | {q+l+d} |")
    lines.append("")

    lines.append("## Arm C (HGT over human SAE) -- by genre\n")
    lines.append("| genre | n rows | distinct source pair_id | token dist |")
    lines.append("|---|---|---|---|")
    for genre, v in arm_c_by_genre.items():
        flag = f" **SMALL CELL**" if v["n"] < SMALL_CELL_THRESHOLD else ""
        docs = v["n_distinct_pair_id"] if v["n_distinct_pair_id"] is not None else \
               f"{v['n']} (pair_id null for this genre -- each row is a distinct perturbed passage, no repeated-seed structure)"
        lines.append(f"| {genre} | {v['n']}{flag} | {docs} | {fmt_dist(v['dist'])} |")
    lines.append(f"\n**Arm C total: {hgt_total_c} rows.** Note: Arm C's poetry slice (283 rows) was built "
                 "over the OLD LiteraryTaste SAE-poetry source (frozen at generation time, per the "
                 "'Arm C is frozen and released as text' decision) -- it does NOT reflect the current "
                 "sae_poetry.csv (521 rows). Arm C's fiction/personal_narrative slices were built over "
                 "unified.csv's SAE-human pool, which (per above) includes the chunk_1 contamination for "
                 "fiction specifically -- Arm C is a probe artifact, frozen and never retrained on, so this "
                 "is disclosed rather than fixed.\n")

    lines.append("## Arm D (HGT over machine SAE) -- by genre\n")
    lines.append("| genre | raw n | usable n | AAVE-MGT token dist | source SAE-MGT token dist |")
    lines.append("|---|---|---|---|---|")
    for genre, v in arm_d_by_genre.items():
        flag = f" **SMALL CELL**" if v["usable"] < SMALL_CELL_THRESHOLD else ""
        lines.append(f"| {genre} | {v['raw']} | {v['usable']}{flag} | {fmt_dist(v['dist_aave'])} | {fmt_dist(v['dist_sae_source'])} |")
    lines.append(f"\n**Arm D total: {hgt_total_d} rows** (all rows usable -- 0 null aave_mgt_text/sae_mgt_text "
                 "in any of the 3 files).\n")

    lines.append("## Empty cells in the genre x variety x label grid\n")
    lines.append("| label | variety | genre | reason |")
    lines.append("|---|---|---|---|")
    for lbl, var, genre, reason in empty_cells:
        lines.append(f"| {lbl} | {var} | {genre} | {reason} |")
    lines.append("")

    lines.append("## Excluded / superseded sources (not counted above)\n")
    for e in exclusions:
        lines.append(f"- {e}")
    lines.append("")

    lines.append("## Totals\n")
    lines.append(f"- **HWT + MGT usable rows (excl. Arm C/D probe): {total_usable}**")
    lines.append(f"- Arm C (HGT, probe-only, never trained on): {hgt_total_c}")
    lines.append(f"- Arm D (HGT, probe-only, never trained on): {hgt_total_d}")
    lines.append(f"- **Grand total incl. HGT probe rows: {total_with_hgt}**")
    total_docs = sum(c["n_docs"] for c in cells if isinstance(c["n_docs"], int))
    lines.append(f"- Sum of distinct source documents across HWT/MGT cells: {total_docs} "
                 "(sum across cells -- a document can appear as one HWT row's source book and "
                 "separately as multiple MGT rows' source, so this is not a global distinct-document "
                 "count across the whole benchmark, just a per-cell sum for scale reference)\n")

    lines.append("## Small-cell flags (usable n < 200)\n")
    small = [c for c in cells if c["usable_n"] < SMALL_CELL_THRESHOLD]
    if small:
        for c in small:
            lines.append(f"- **{c['label']}/{c['variety']}/{c['genre']}**: n={c['usable_n']} -- claims "
                         "resting on this cell alone should report exact n and avoid overgeneralizing.")
    else:
        lines.append("- None of the primary HWT/MGT cells fall under 200 usable rows.")
    small_arm_c = [(g, v) for g, v in arm_c_by_genre.items() if v["n"] < SMALL_CELL_THRESHOLD]
    small_arm_d = [(g, v) for g, v in arm_d_by_genre.items() if v["usable"] < SMALL_CELL_THRESHOLD]
    for g, v in small_arm_c:
        lines.append(f"- **Arm C / {g}**: n={v['n']}")
    for g, v in small_arm_d:
        lines.append(f"- **Arm D / {g}**: n={v['usable']}")
    lines.append("")

    # ---- Compact paper-ready table ----
    lines.append("## Compact table (paper-ready)\n")
    lines.append("| Genre | Variety | HWT | MGT | HGT (Arm C) | HGT (Arm D) |")
    lines.append("|---|---|---|---|---|---|")
    compact_rows = [
        ("fiction", "AAVE", fic_aave, None, None, None),
        ("fiction", "SAE", fic_sae, mgt_by_generator.get("fiction"), arm_c_by_genre.get("fiction"), arm_d_by_genre.get("fiction")),
        ("poetry", "AAVE", poetry_aave, None, None, None),
        ("poetry", "SAE", poetry_s_full, mgt_by_generator.get("poetry"), arm_c_by_genre.get("poetry"), arm_d_by_genre.get("poetry")),
        ("lyrics", "AAVE", lyrics_valid, None, None, None),
        ("lyrics", "SAE", None, None, None, None),
        ("oratory", "AAVE", None, None, None, None),
        ("oratory", "SAE", orat_elig, mgt_by_generator.get("oratory"), arm_c_by_genre.get("oratory"), arm_d_by_genre.get("oratory")),
        ("personal_narrative", "AAVE", None, None, arm_c_by_genre.get("personal_narrative"), None),
        ("personal_narrative", "SAE", pn_sae_h, len(pn_sae_llm), arm_c_by_genre.get("personal_narrative"), None),
    ]
    seen = set()
    for genre, variety, hwt_df, mgt_counts, arm_c_v, arm_d_v in compact_rows:
        key = (genre, variety)
        if key in seen:
            continue
        seen.add(key)
        hwt_n = "-" if hwt_df is None else str(len(hwt_df))
        if mgt_counts is None:
            mgt_n = "-"
        elif isinstance(mgt_counts, dict):
            mgt_n = str(sum(mgt_counts.values()))
        else:
            mgt_n = str(mgt_counts)
        arm_c_n = "-" if not arm_c_v else str(arm_c_v["n"])
        arm_d_n = "-" if not arm_d_v else str(arm_d_v["usable"])
        lines.append(f"| {genre} | {variety} | {hwt_n} | {mgt_n} | {arm_c_n} | {arm_d_n} |")
    lines.append("")

    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report)
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
