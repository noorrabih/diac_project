"""
Diacritics Statistics for BAREC10M — BINNED by diacritization rate (sixths)
=============================================================================
Bins sentences by diacritization rate (diacritics / Arabic letters) into 6 bins:
  0        rate = 0 exactly (no diacritics)
  1/5      rate in (0,   0.2]
  2/5      rate in (0.2, 0.4]
  3/5      rate in (0.4, 0.6]
  4/5      rate in (0.6, 0.8]
  5/5      rate > 0.8  (fully or more diacritized)
- Letters:  Arabic script characters only; punctuation and spaces excluded
- Shadda (ّ U+0651) immediately followed by another diacritic counts as ONE
"""

import re
import json
import glob as glob_module
import pandas as pd
from pathlib import Path
from camel_tools.utils.charsets import AR_DIAC_CHARSET

SHADDA    = 'ّ'
TANWEEN   = {'ً', 'ٌ', 'ٍ'}
_DIAC_SET = AR_DIAC_CHARSET

# Arabic core letters (ء–غ  and  ف–ي) plus alef wasla (ٱ)
_AR_LETTER_RE = re.compile(r'[ء-غف-يٱ]')

BIN_LABELS = ["0", "1/5", "2/5", "3/5", "4/5", "5/5"]
BIN_EDGES  = [-0.001, 0.0, 0.2, 0.4, 0.6, 0.8, float('inf')]


def assign_bin(rate: float) -> str:
    if rate == 0.0:   return "0"
    if rate <= 0.2:   return "1/5"
    if rate <= 0.4:   return "2/5"
    if rate <= 0.6:   return "3/5"
    if rate <= 0.8:   return "4/5"
    return "5/5"


def count_letters(text: str) -> int:
    return len(_AR_LETTER_RE.findall(text))


def count_diac(text: str, exclude_tanween: bool = False) -> int:
    """Count diacritics; shadda immediately followed by another diacritic = 1 unit."""
    exclude = TANWEEN if exclude_tanween else set()
    count = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c in _DIAC_SET:
            if c == SHADDA and (i + 1) < len(text) and text[i + 1] in _DIAC_SET:
                # fused pair: shadda + next diacritic
                if text[i + 1] not in exclude:
                    count += 1
                i += 2
            else:
                if c not in exclude:
                    count += 1
                i += 1
        else:
            i += 1
    return count


def sent_level(rl_entry):
    if isinstance(rl_entry, list):
        return max(set(rl_entry), key=rl_entry.count) if rl_entry else None
    return rl_entry


# ── Load all JSON files ──────────────────────────────────────────────────────
DATA_DIR = Path("/l/users/nour.rabih/BAREC10M")

print("Loading JSON files ...")
rows = []
skipped = 0


def load_json_tolerant(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        pass
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
    source = json_file.parent.name
    try:
        data, was_truncated = load_json_tolerant(json_file)
        if data is None:
            skipped += 1
            print(f"  WARNING: skipped {json_file.name} — could not parse even partially")
            continue
        if was_truncated:
            print(f"  PARTIAL:  {json_file.name} — truncated, salvaged raw_sents+sents_RL")

        sents       = data.get("raw_sents", [])
        levels      = data.get("sents_RL", [])
        sent_levels = [sent_level(rl) for rl in levels]

        if len(sent_levels) != len(sents):
            print(f"  WARNING: level/sentence count mismatch in {json_file.name} "
                  f"({len(sent_levels)} levels vs {len(sents)} sents)")
            sent_levels += [None] * (len(sents) - len(sent_levels))

        for sent, rl in zip(sents, sent_levels):
            rows.append({"source": source, "file": json_file.name, "Sentence": sent, "RL": rl})

    except Exception as e:
        skipped += 1
        print(f"  WARNING: skipped {json_file.name} — {e}")

print(f"Loaded {len(rows):,} sentences ({skipped} files skipped)\n")

# ── Build DataFrame and compute rates ────────────────────────────────────────
print("Computing diacritization rates ...")
df = pd.DataFrame(rows)
df["RL"] = pd.to_numeric(df["RL"], errors="coerce")

df["letter_count"]       = df["Sentence"].apply(count_letters)
df["diac_count"]         = df["Sentence"].apply(count_diac)
df["diac_count_no_tanw"] = df["Sentence"].apply(lambda s: count_diac(s, exclude_tanween=True))

df["rate"]         = df["diac_count"]         / df["letter_count"].replace(0, pd.NA)
df["rate_no_tanw"] = df["diac_count_no_tanw"] / df["letter_count"].replace(0, pd.NA)
df[["rate", "rate_no_tanw"]] = df[["rate", "rate_no_tanw"]].fillna(0.0)

df["bin"]         = pd.cut(df["rate"],         bins=BIN_EDGES, labels=BIN_LABELS,
                            right=True, include_lowest=True)
df["bin_no_tanw"] = pd.cut(df["rate_no_tanw"], bins=BIN_EDGES, labels=BIN_LABELS,
                            right=True, include_lowest=True)

print(f"Total sentences : {len(df):,}")
print(f"Sources found   : {sorted(df['source'].unique())}")
print(f"RL levels found : {sorted(df['RL'].dropna().unique())}")
print()


# ── Helper: bin distribution table ───────────────────────────────────────────
def make_bin_dist(bin_series, rate_series, total):
    grp = pd.DataFrame({"bin": bin_series, "rate": rate_series})
    agg = (grp.groupby("bin", observed=False)["rate"]
               .agg(num_sentences="count", mean_rate="mean")
               .reset_index())
    agg["pct"]       = (agg["num_sentences"] / total * 100).round(2)
    agg["mean_rate"] = agg["mean_rate"].fillna(0.0).round(4)
    return agg


def print_dist_pair(dist_all, dist_no_tanw, label, total):
    combined = dist_all.merge(
        dist_no_tanw.rename(columns={
            "num_sentences": "num_sentences_no_tanw",
            "pct":           "pct_no_tanw",
            "mean_rate":     "mean_rate_no_tanw",
        }),
        on="bin", how="outer"
    )
    combined["bin"] = pd.Categorical(combined["bin"], categories=BIN_LABELS, ordered=True)
    combined = combined.sort_values("bin").fillna(0)

    if label:
        print(f"\n  {label}")
    print(f"  {'Bin':>10}  {'#Sents(all)':>12}  {'%(all)':>7}  {'rate(all)':>10}  "
          f"{'#Sents(no tanw)':>16}  {'%(no tanw)':>11}  {'rate(no tanw)':>14}")
    print("  " + "-" * 90)
    for _, row in combined.iterrows():
        print(f"  {str(row['bin']):>10}  {int(row['num_sentences']):>12}  {row['pct']:>6.2f}%  "
              f"{row['mean_rate']:>10.4f}  {int(row['num_sentences_no_tanw']):>16}  "
              f"{row['pct_no_tanw']:>10.2f}%  {row['mean_rate_no_tanw']:>14.4f}")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 1. Overall
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 95)
print("OVERALL  –  Diacritization Rate Bins  (diacritics / Arabic letters, fifths)")
print("=" * 95)

overall_all      = make_bin_dist(df["bin"],         df["rate"],        len(df))
overall_no_tanw  = make_bin_dist(df["bin_no_tanw"], df["rate_no_tanw"], len(df))
overall_comb     = print_dist_pair(overall_all, overall_no_tanw, "All sentences", len(df))

# ══════════════════════════════════════════════════════════════════════════════
# 2. Per RL
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 95)
print("PER READABILITY LEVEL")
print("=" * 95)

per_level_rows = []
for level in sorted(df["RL"].dropna().unique()):
    sub = df[df["RL"] == level]
    print(f"\n── RL: {level}  ({len(sub):,} sentences) ──────────────")
    d_all     = make_bin_dist(sub["bin"],         sub["rate"],        len(sub))
    d_no_tanw = make_bin_dist(sub["bin_no_tanw"], sub["rate_no_tanw"], len(sub))
    comb      = print_dist_pair(d_all, d_no_tanw, "", len(sub))
    comb.insert(0, "RL", level)
    per_level_rows.append(comb)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Per source
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 95)
print("PER SOURCE")
print("=" * 95)

per_source_rows = []
for source in sorted(df["source"].unique()):
    sub = df[df["source"] == source]
    print(f"\n── Source: {source}  ({len(sub):,} sentences) ──────────────")
    d_all     = make_bin_dist(sub["bin"],         sub["rate"],        len(sub))
    d_no_tanw = make_bin_dist(sub["bin_no_tanw"], sub["rate_no_tanw"], len(sub))
    comb      = print_dist_pair(d_all, d_no_tanw, "", len(sub))
    comb.insert(0, "source", source)
    per_source_rows.append(comb)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Per source × RL
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 95)
print("PER SOURCE × PER READABILITY LEVEL")
print("=" * 95)

per_src_rl_rows = []
for source in sorted(df["source"].unique()):
    src_df = df[df["source"] == source]
    for level in sorted(src_df["RL"].dropna().unique()):
        sub = src_df[src_df["RL"] == level]
        if len(sub) == 0:
            continue
        d_all     = make_bin_dist(sub["bin"],         sub["rate"],        len(sub))
        d_no_tanw = make_bin_dist(sub["bin_no_tanw"], sub["rate_no_tanw"], len(sub))
        comb = d_all.merge(
            d_no_tanw.rename(columns={
                "num_sentences": "num_sentences_no_tanw",
                "pct":           "pct_no_tanw",
                "mean_rate":     "mean_rate_no_tanw",
            }),
            on="bin", how="outer"
        )
        comb["bin"] = pd.Categorical(comb["bin"], categories=BIN_LABELS, ordered=True)
        comb = comb.sort_values("bin").fillna(0)
        comb.insert(0, "RL",     level)
        comb.insert(0, "source", source)
        per_src_rl_rows.append(comb)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Save CSVs
# ══════════════════════════════════════════════════════════════════════════════
overall_comb.to_csv("barec10m_bins_overall.csv", index=False)
pd.concat(per_level_rows).to_csv("barec10m_bins_per_rl.csv", index=False)
pd.concat(per_source_rows).to_csv("barec10m_bins_per_source.csv", index=False)
pd.concat(per_src_rl_rows).to_csv("barec10m_bins_per_source_per_rl.csv", index=False)
df[["source", "file", "RL", "Sentence",
    "letter_count", "diac_count", "rate", "bin",
    "diac_count_no_tanw", "rate_no_tanw", "bin_no_tanw"]].to_csv(
    "barec10m_bins_per_sentence.csv", index=False
)

print("\n\nCSV files saved:")
print("  barec10m_bins_overall.csv             — overall bin distribution")
print("  barec10m_bins_per_rl.csv              — per readability level")
print("  barec10m_bins_per_source.csv          — per source subdirectory")
print("  barec10m_bins_per_source_per_rl.csv   — per source × per RL")
print("  barec10m_bins_per_sentence.csv        — per sentence with metadata")
