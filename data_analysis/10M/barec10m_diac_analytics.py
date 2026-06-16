"""
Diacritics Statistics for BAREC10M local JSON dataset
======================================================
Reads JSON files from subdirectories, each with raw_sents and sents_RL fields.
sents_RL is a list-of-lists (one list per sentence, each containing per-token
readability levels). The sentence-level RL is derived as the most common value.
"""

import re
import json
import glob as glob_module
import pandas as pd
from pathlib import Path
from camel_tools.utils.charsets import AR_DIAC_CHARSET

# ── Tanween characters ───────────────────────────────────────────────────────
TANWEEN = {'\u064b', '\u064c', '\u064d'}  # ً ٌ ٍ

# ── Build regexes once ───────────────────────────────────────────────────────
_DIAC_RE            = re.compile('[' + re.escape(''.join(AR_DIAC_CHARSET)) + ']')
_DIAC_NO_TANWEEN_RE = re.compile('[' + re.escape(''.join(AR_DIAC_CHARSET - TANWEEN)) + ']')

def count_diacritics(text: str) -> int:
    return len(_DIAC_RE.findall(text))

def count_diacritics_no_tanween(text: str) -> int:
    return len(_DIAC_NO_TANWEEN_RE.findall(text))

def sent_level(rl_entry):
    """Extract a single RL value from a token-level list or scalar."""
    if isinstance(rl_entry, list):
        return max(set(rl_entry), key=rl_entry.count) if rl_entry else None
    return rl_entry  # already a scalar (or None)

# ── Load all JSON files ──────────────────────────────────────────────────────
DATA_DIR = Path("/l/users/nour.rabih/BAREC10M")

print("Loading JSON files ...")
rows = []
skipped = 0

def load_json_tolerant(path):
    """Load JSON, falling back to partial extraction if the file is truncated."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        pass
    # File is truncated — try to salvage raw_sents and sents_RL which appear early
    m_sents  = re.search(r'"raw_sents"\s*:\s*(\[.*?\])(?=\s*,)', raw, re.DOTALL)
    m_levels = re.search(r'"sents_RL"\s*:\s*(\[.*?\])(?=\s*,)', raw, re.DOTALL)
    if m_sents:
        try:
            sents  = json.loads(m_sents.group(1))
            levels = json.loads(m_levels.group(1)) if m_levels else []
            return {"raw_sents": sents, "sents_RL": levels}, True
        except Exception:
            pass
    return None, True


for json_file in map(Path, glob_module.glob(str(DATA_DIR / "**" / "*.json"), recursive=True)):
    source = json_file.parent.name  # subdirectory name, e.g. "Majed"
    try:
        data, was_truncated = load_json_tolerant(json_file)
        if data is None:
            skipped += 1
            print(f"  WARNING: skipped {json_file.name} — could not parse even partially")
            continue
        if was_truncated:
            print(f"  PARTIAL:  {json_file.name} — truncated, salvaged raw_sents+sents_RL")

        sents  = data.get("raw_sents", [])
        levels = data.get("sents_RL", [])

        # Flatten: one RL value per sentence
        sent_levels = [sent_level(rl) for rl in levels]

        # Pad if lengths differ
        if len(sent_levels) != len(sents):
            print(f"  WARNING: level/sentence count mismatch in {json_file.name} "
                  f"({len(sent_levels)} levels vs {len(sents)} sents)")
            sent_levels += [None] * (len(sents) - len(sent_levels))

        for sent, rl in zip(sents, sent_levels):
            rows.append({
                "source":   source,
                "file":     json_file.name,
                "Sentence": sent,
                "RL":       rl,
            })

    except Exception as e:
        skipped += 1
        print(f"  WARNING: skipped {json_file.name} — {e}")

print(f"Loaded {len(rows):,} sentences from {DATA_DIR} ({skipped} files skipped)\n")

# Build DataFrame and count diacritics
print("Counting diacritics ...")
df = pd.DataFrame(rows)

# Coerce RL to numeric: some files store levels as strings, others as ints
df["RL"] = pd.to_numeric(df["RL"], errors="coerce")

df["diac_count"]            = df["Sentence"].apply(count_diacritics)
df["diac_count_no_tanween"] = df["Sentence"].apply(count_diacritics_no_tanween)

print(f"Total sentences : {len(df):,}")
print(f"Sources found   : {sorted(df['source'].unique())}")
print(f"RL levels found : {sorted(df['RL'].dropna().unique())}")
print(f"RL null count   : {df['RL'].isna().sum():,} sentences have no RL\n")

# ── Helper: build a distribution table ──────────────────────────────────────
def make_dist(series, total):
    dist = (series.value_counts().sort_index()
            .rename_axis("diac_count").reset_index(name="num_sentences"))
    dist["pct"] = (dist["num_sentences"] / total * 100).round(2)
    return dist

def print_dist_pair(dist_all, dist_no_tanw, label, total):
    combined = dist_all.merge(
        dist_no_tanw.rename(columns={
            "num_sentences": "num_sentences_no_tanween",
            "pct": "pct_no_tanween"
        }),
        on="diac_count", how="outer"
    ).sort_values("diac_count").fillna(0)
    if label:
        print(f"\n  {label}")
    print(f"  {'Diacritics':>12}  {'#Sents(all)':>12}  {'%(all)':>7}  "
          f"{'#Sents(no tanween)':>20}  {'%(no tanween)':>14}")
    print("  " + "-" * 75)
    for _, row in combined.iterrows():
        print(f"  {int(row['diac_count']):>12}  {int(row['num_sentences']):>12}  "
              f"{row['pct']:>6.2f}%  {int(row['num_sentences_no_tanween']):>20}  "
              f"{row['pct_no_tanween']:>13.2f}%")
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
# 2. Per-RL distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 75)
print("PER READABILITY LEVEL (sents_RL)")
print("=" * 75)

per_level_rows = []
for level in sorted(df["RL"].dropna().unique()):
    sub = df[df["RL"] == level]
    print(f"\n── RL: {level}  ({len(sub):,} sentences) ──────────────")
    d_all     = make_dist(sub["diac_count"],            len(sub))
    d_no_tanw = make_dist(sub["diac_count_no_tanween"], len(sub))
    combined  = print_dist_pair(d_all, d_no_tanw, "", len(sub))
    combined.insert(0, "RL", level)
    per_level_rows.append(combined)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Per-source distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 75)
print("PER SOURCE (subdirectory)")
print("=" * 75)

per_source_rows = []
for source in sorted(df["source"].unique()):
    sub = df[df["source"] == source]
    print(f"\n── Source: {source}  ({len(sub):,} sentences) ──────────────")
    d_all     = make_dist(sub["diac_count"],            len(sub))
    d_no_tanw = make_dist(sub["diac_count_no_tanween"], len(sub))
    combined  = print_dist_pair(d_all, d_no_tanw, "", len(sub))
    combined.insert(0, "source", source)
    per_source_rows.append(combined)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Per-source × per-RL distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 75)
print("PER SOURCE × PER READABILITY LEVEL")
print("=" * 75)

per_source_rl_rows = []
for source in sorted(df["source"].unique()):
    src_df = df[df["source"] == source]
    for level in sorted(src_df["RL"].dropna().unique()):
        sub = src_df[src_df["RL"] == level]
        if len(sub) == 0:
            continue
        d_all     = make_dist(sub["diac_count"],            len(sub))
        d_no_tanw = make_dist(sub["diac_count_no_tanween"], len(sub))
        combined  = d_all.merge(
            d_no_tanw.rename(columns={
                "num_sentences": "num_sentences_no_tanween",
                "pct":           "pct_no_tanween"
            }),
            on="diac_count", how="outer"
        ).sort_values("diac_count").fillna(0)
        combined.insert(0, "RL",     level)
        combined.insert(0, "source", source)
        per_source_rl_rows.append(combined)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Save to CSV
# ══════════════════════════════════════════════════════════════════════════════
overall_combined.to_csv("barec10m_diacritics_overall.csv", index=False)
pd.concat(per_level_rows).to_csv("barec10m_diacritics_per_rl.csv", index=False)
pd.concat(per_source_rows).to_csv("barec10m_diacritics_per_source.csv", index=False)
pd.concat(per_source_rl_rows).to_csv("barec10m_diacritics_per_source_per_rl.csv", index=False)
df[["source", "file", "RL", "Sentence", "diac_count", "diac_count_no_tanween"]].to_csv(
    "barec10m_diacritics_per_sentence.csv", index=False
)

print("\n\nCSV files saved:")
print("  barec10m_diacritics_overall.csv             — overall distribution")
print("  barec10m_diacritics_per_rl.csv              — per readability level")
print("  barec10m_diacritics_per_source.csv          — per source subdirectory")
print("  barec10m_diacritics_per_source_per_rl.csv   — per source × per RL")
print("  barec10m_diacritics_per_sentence.csv        — per sentence with metadata")