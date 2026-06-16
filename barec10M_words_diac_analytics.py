"""
Word-Level Lexeme Diacritics Statistics for BAREC10M
=====================================================
For each sentence, raw_sents is split to match the tokenization of word[]/lex[]/RL[]
exactly by index, so raw_split[i] == diacritized surface of word[i].

Diac count is taken from raw_split[i] directly.
Grouping key is lex[i]. Per-token RL from RL[i].

Output:
  barec10m_lex_diacritics.csv  — lex | RL | diac_count | occurrences | words
"""

import re
import json
import glob as glob_module
import pandas as pd
from pathlib import Path
from collections import defaultdict
from camel_tools.utils.charsets import AR_DIAC_CHARSET

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/l/users/nour.rabih/BAREC10M")

# ── Diacritic helpers ─────────────────────────────────────────────────────────
TANWEEN  = {'\u064b', '\u064c', '\u064d'}  # ً ٌ ٍ
DIAC_SET = AR_DIAC_CHARSET

def strip_diac(text: str) -> str:
    return ''.join(c for c in text if c not in DIAC_SET)

def count_diacs(text: str) -> int:
    return sum(1 for c in text if c in DIAC_SET)

def count_diacs_no_tanween(text: str) -> int:
    return sum(1 for c in text if c in DIAC_SET and c not in TANWEEN)

def is_punct(word: str) -> bool:
    return not any(c.isalpha() for c in word)

def split_raw_like_word(raw_sent: str, word_tokens: list) -> list:
    """
    Split raw_sent into exactly len(word_tokens) tokens, aligned by index.
    Each output token is the substring of raw_sent (with any diacritics present)
    that corresponds to word_tokens[i] (stripped of diacritics for matching).
    Uses a left-to-right scan so repeated words are handled correctly.
    """
    raw_stripped = strip_diac(raw_sent)
    result = []
    pos = 0  # cursor in raw_stripped (non-diac chars only)

    for w in word_tokens:
        w_key = strip_diac(w)
        if not w_key.strip():
            result.append(w)
            continue

        idx = raw_stripped.find(w_key, pos)
        if idx == -1:
            result.append(w)   # fallback: no diacritics
            continue

        # Map idx and idx+len(w_key) back to positions in raw_sent
        nd = 0
        raw_start = raw_end = None
        for ci, ch in enumerate(raw_sent):
            if ch not in DIAC_SET:
                if nd == idx:
                    raw_start = ci
                if nd == idx + len(w_key):
                    raw_end = ci
                    break
                nd += 1
        if raw_end is None:
            raw_end = len(raw_sent)

        result.append(raw_sent[raw_start:raw_end])
        pos = idx + len(w_key)

    return result

# ── Main accumulators ─────────────────────────────────────────────────────────
acc    = defaultdict(lambda: {'occ': 0, 'words': set()})
acc_nt = defaultdict(int)

skipped      = 0
total_tokens = 0

json_files = list(map(Path, glob_module.glob(str(DATA_DIR / "**" / "*.json"), recursive=True)))
print(f"Processing {len(json_files):,} JSON files ...")
j = 0
for i, json_file in enumerate(json_files):
    j += 1
    if i % 500 == 0:
        print(f"  {i:,} / {len(json_files):,} files  ({total_tokens:,} tokens so far)")

    try:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {json_file.name}")
        raw_sents = data.get("raw_sents", [])
        words_all = data.get("word",      [])
        lexes_all = data.get("lex",       [])
        rls_all   = data.get("RL",        [])

        for raw_sent, words, lexes, rls in zip(raw_sents, words_all, lexes_all, rls_all):

            # split raw_sent to align 1:1 with word[]/lex[]/RL[]
            raw_split = split_raw_like_word(raw_sent, words)
            # import pdb; pdb.set_trace()  # ← DEBUG
            for raw_tok, word, lex, rl in zip(raw_split, words, lexes, rls):
                if is_punct(word) or not lex or is_punct(lex):
                    continue

                dc    = count_diacs(raw_tok)
                dc_nt = count_diacs_no_tanween(raw_tok)

                acc[(lex, rl, dc)]['occ'] += 1
                acc[(lex, rl, dc)]['words'].add(word)
                acc_nt[(lex, rl, dc_nt)] += 1
                total_tokens += 1

    except Exception as e:
        skipped += 1
        print(f"  WARNING: skipped {json_file.name} — {e}")
    if j == 1000:
        break  # ← REMOVE after testing on a few files

print(f"\nDone. {total_tokens:,} tokens | {len(acc):,} unique (lex, RL, diac_count) groups | {skipped} files skipped")

# ── Build DataFrame ───────────────────────────────────────────────────────────
print("\nBuilding DataFrame ...")

rows = [
    {
        'lex':         lex,
        'RL':          rl,
        'diac_count':  dc,
        'occurrences': val['occ'],
        'words':       '|'.join(sorted(val['words'])),
    }
    for (lex, rl, dc), val in acc.items()
]

rows_nt = [
    {'lex': lex, 'RL': rl, 'diac_count_no_tanween': dc, 'occurrences_no_tanween': cnt}
    for (lex, rl, dc), cnt in acc_nt.items()
]

df    = pd.DataFrame(rows)
df_nt = pd.DataFrame(rows_nt)

df = (df.merge(df_nt, on=['lex', 'RL'], how='left')
        .sort_values(['lex', 'RL', 'diac_count'])
        .reset_index(drop=True))

df['RL'] = pd.to_numeric(df['RL'], errors='coerce')

# ── Save ──────────────────────────────────────────────────────────────────────
out = "barec10m_lex_diacritics.csv"
df.to_csv(out, index=False)

print(f"\nSaved: {out}")
print(f"  Rows           : {len(df):,}")
print(f"  Unique lexemes : {df['lex'].nunique():,}")
print(f"  RL range       : {df['RL'].min()} – {df['RL'].max()}")
print(f"\nTop 20 lexemes by total occurrences:")
print(df.groupby('lex')['occurrences'].sum()
        .sort_values(ascending=False).head(20).to_string())