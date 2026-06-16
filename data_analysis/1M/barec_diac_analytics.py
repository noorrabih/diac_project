"""
Diacritics Statistics for CAMeL-Lab/BAREC-Corpus-v1.0
======================================================
Optimized for large datasets (~1M+ words).
Reports diacritic counts both with and without tanween (ً ٌ ٍ).
"""

import re
import pandas as pd
from datasets import load_dataset
from camel_tools.utils.charsets import AR_DIAC_CHARSET

# ── Tanween characters (fathatan, dammatan, kasratan) ───────────────────────
TANWEEN = {'\u064b', '\u064c', '\u064d'}  # ً ٌ ٍ

# ── Build regexes once from camel-tools charset ──────────────────────────────
_DIAC_RE            = re.compile('[' + re.escape(''.join(AR_DIAC_CHARSET)) + ']')
_DIAC_NO_TANWEEN_RE = re.compile('[' + re.escape(''.join(AR_DIAC_CHARSET - TANWEEN)) + ']')

def count_diacritics(text: str) -> int:
    return len(_DIAC_RE.findall(text))

def count_diacritics_no_tanween(text: str) -> int:
    return len(_DIAC_NO_TANWEEN_RE.findall(text))

# ── Load dataset ─────────────────────────────────────────────────────────────
print("Loading dataset …")
ds = load_dataset("CAMeL-Lab/BAREC-Corpus-v1.0")

first_split = list(ds.keys())[0]
print(f"Columns : {ds[first_split].column_names}")
print(f"Sample  : {ds[first_split][0]}\n")

TEXT_COL  = "Sentence"
LEVEL_COL = "Readability_Level_19"
ID_COL    = "ID"  # ← adjust if needed after checking printed Sample above

# ── Count with datasets .map() (batched + multiprocessing) ───────────────────
print("Counting diacritics …")

def add_diac_counts(batch):
    batch["diac_count"]            = [count_diacritics(t)            for t in batch[TEXT_COL]]
    batch["diac_count_no_tanween"] = [count_diacritics_no_tanween(t) for t in batch[TEXT_COL]]
    return batch

ds = ds.map(add_diac_counts, batched=True, batch_size=1000, num_proc=4)

# ── Concatenate all splits into one DataFrame ────────────────────────────────
df = pd.concat(
    [ds[split].select_columns([ID_COL, TEXT_COL, LEVEL_COL, "diac_count", "diac_count_no_tanween"]).to_pandas()
     for split in ds.keys()],
    ignore_index=True
)

print(f"Total sentences : {len(df):,}")
print(f"Levels found    : {sorted(df[LEVEL_COL].unique())}\n")

# ── Helper: build a distribution table ───────────────────────────────────────
def make_dist(series, total):
    dist = series.value_counts().sort_index().rename_axis("diac_count").reset_index(name="num_sentences")
    dist["pct"] = (dist["num_sentences"] / total * 100).round(2)
    return dist

def print_dist_pair(dist_all, dist_no_tanw, label, total):
    combined = dist_all.merge(
        dist_no_tanw.rename(columns={"num_sentences": "num_sentences_no_tanween", "pct": "pct_no_tanween"}),
        on="diac_count", how="outer"
    ).sort_values("diac_count").fillna(0)
    print(f"\n  {label}")
    print(f"  {'Diacritics':>12}  {'#Sents(all)':>12}  {'%(all)':>7}  {'#Sents(no tanween)':>20}  {'%(no tanween)':>14}")
    print("  " + "-" * 75)
    for _, row in combined.iterrows():
        print(f"  {int(row['diac_count']):>12}  {int(row['num_sentences']):>12}  {row['pct']:>6.2f}%  "
              f"{int(row['num_sentences_no_tanween']):>20}  {row['pct_no_tanween']:>13.2f}%")
    print(f"  all  → mean={dist_all['diac_count'].repeat(dist_all['num_sentences']).mean():.2f}  "
          f"max={int(dist_all['diac_count'].max())}")
    print(f"  excl → mean={dist_no_tanw['diac_count'].repeat(dist_no_tanw['num_sentences']).mean():.2f}  "
          f"max={int(dist_no_tanw['diac_count'].max())}")
    return combined

# ══════════════════════════════════════════════════════════════════════════════
# 1. Overall distribution
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 75)
print("OVERALL  –  Diacritics per Sentence")
print("=" * 75)

overall_all     = make_dist(df["diac_count"],            len(df))
overall_no_tanw = make_dist(df["diac_count_no_tanween"], len(df))
overall_combined = print_dist_pair(overall_all, overall_no_tanw, "All sentences", len(df))

# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-level distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 75)
print(f"PER LEVEL ({LEVEL_COL})")
print("=" * 75)

per_level_rows = []
for level in sorted(df[LEVEL_COL].unique()):
    sub = df[df[LEVEL_COL] == level]
    print(f"\n── Level: {level}  ({len(sub):,} sentences) ──────────────")

    d_all     = make_dist(sub["diac_count"],            len(sub))
    d_no_tanw = make_dist(sub["diac_count_no_tanween"], len(sub))
    combined  = print_dist_pair(d_all, d_no_tanw, "", len(sub))
    combined.insert(0, "level", level)
    per_level_rows.append(combined)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Save to CSV
# ══════════════════════════════════════════════════════════════════════════════
overall_combined.to_csv("barec_diacritics_overall.csv", index=False)
pd.concat(per_level_rows).to_csv("barec_diacritics_per_level.csv", index=False)
df[[ID_COL, LEVEL_COL, TEXT_COL, "diac_count", "diac_count_no_tanween"]].to_csv(
    "barec_diacritics_per_sentence.csv", index=False
)

print("\n\nCSV files saved:")
print("  barec_diacritics_overall.csv      — diac_count | num_sentences | pct | num_sentences_no_tanween | pct_no_tanween")
print("  barec_diacritics_per_level.csv    — same columns, with leading 'level' column")
print("  barec_diacritics_per_sentence.csv — diac_count | diac_count_no_tanween per sentence")