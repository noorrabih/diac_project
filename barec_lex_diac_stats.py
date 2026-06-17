"""
BAREC Corpus — Lexeme Diacritics Statistics (via CAMeL BERT Disambiguator)
===========================================================================
Pipeline:
  1. Load all three splits of CAMeL-Lab/BAREC-Corpus-v1.0 (word-level rows)
  2. Group by unique Sentence, disambiguate with BERT to get (lex, diac) per token
  3. Aggregate by (lex, diac) across the full corpus

Output CSV:  barec_lex_diac_stats.csv
  lex | diac | totalcount | count_level
  count_level — JSON {Readability_Level: count}

Run inside the camel_tools conda env:
  conda run -n camel_tools python barec_lex_diac_stats.py
"""

import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charsets import AR_DIAC_CHARSET

DIAC_SET = set(AR_DIAC_CHARSET)

def count_diacs(word: str) -> int:
    return sum(1 for c in word if c in DIAC_SET)

# ── CAMeL Tools setup ─────────────────────────────────────────────────────────
DB_PATH  = "/Users/nour.rabih/Desktop/arwi/calima-msa-s31.db"
db       = MorphologyDB(DB_PATH, "a")
analyzer = Analyzer(db, 'ADD_PROP', cache_size=100000)
bert     = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False)
bert._analyzer = analyzer

# ── Load all splits ───────────────────────────────────────────────────────────
BASE = "hf://datasets/CAMeL-Lab/BAREC-Corpus-v1.0/"
SPLITS = {
    'train': 'data/train-00000-of-00001.parquet',
    # 'dev':   'data/dev-00000-of-00001.parquet',
    # 'test':  'data/test-00000-of-00001.parquet',
}

df = pd.concat(
    [pd.read_parquet(BASE + path) for path in SPLITS.values()],
    ignore_index=True
)
print(f"Loaded {len(df):,} word-level rows")

# df = df[:100]  # ← remove for full run

# ── Deduplicate: one row per unique sentence ──────────────────────────────────
sent_df = (
    df.drop_duplicates(subset=['Sentence'])
      [['ID', 'Sentence', 'Readability_Level']]
      .reset_index(drop=True)
)
print(f"Unique sentences to process: {len(sent_df):,}\n")
# ── Helpers ───────────────────────────────────────────────────────────────────
def is_punct(word: str) -> bool:
    return not any(c.isalpha() for c in word)

OUT              = "barec_lex_diac_stats.csv"
CHECKPOINT_EVERY = 2_000   # save partial CSV every N sentences

# acc[(lex, diac)][RL] → count
acc = defaultdict(lambda: defaultdict(int))

def save_checkpoint(acc, path):
    rows = [
        {
            'lex':         lex,
            'diac':        diac,
            'diac_count':  count_diacs(diac),
            'totalcount':  sum(level_counts.values()),
            'count_level': json.dumps({str(k): v
                                       for k, v in sorted(level_counts.items())}),
        }
        for (lex, diac), level_counts in acc.items()
    ]
    (pd.DataFrame(rows)
       .sort_values(['lex', 'totalcount'], ascending=[True, False])
       .reset_index(drop=True)
       .to_csv(path, index=False))

# ── Main loop ─────────────────────────────────────────────────────────────────
skipped = 0

for idx, row in tqdm(sent_df.iterrows(), total=len(sent_df), desc="Disambiguating"):
    sentence = str(row['Sentence'])
    # print(row['ID'], sentence)
    rl       = row['Readability_Level']

    tokens = simple_word_tokenize(sentence)
    if not tokens:
        continue

    try:
        disambig = bert.tag_sentence(tokens)
    except Exception as e:
        skipped += 1
        continue

    # Debug first sentence to confirm dict structure
    if idx == 0:
        print("Sample d_word:", type(disambig[0]), disambig[0])

    for token, d_word in zip(tokens, disambig):
        # lex from BERT analysis, diac from the surface word in the sentence
        lex  = d_word.get('lex', '')
        diac = token

        if not lex or not diac:
            continue
        if is_punct(lex) or is_punct(diac):
            continue

        acc[(lex, diac)][rl] += 1

    if (idx + 1) % CHECKPOINT_EVERY == 0:
        save_checkpoint(acc, OUT)
        print(f"  [checkpoint] {idx+1:,} sentences | {len(acc):,} (lex,diac) pairs")

# ── Final save ────────────────────────────────────────────────────────────────
save_checkpoint(acc, OUT)

result = pd.read_csv(OUT)
print(f"\nSaved: {OUT}")
print(f"  Rows              : {len(result):,}")
print(f"  Unique lexemes    : {result['lex'].nunique():,}")
print(f"  Sentences skipped : {skipped:,}")

print(f"\nTop 10 most frequent diac forms:")
print(result.nlargest(10, 'totalcount')[['lex', 'diac', 'diac_count', 'totalcount', 'count_level']].to_string(index=False))


# show top ones with diac_count > 0
print(f"\nTop 10 most frequent diac forms (diac_count > 0):")
print(result[result['diac_count'] > 0].nlargest(10, 'totalcount')[['lex', 'diac', 'diac_count', 'totalcount', 'count_level']].to_string(index=False))