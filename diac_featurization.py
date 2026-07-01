"""
Generate Arabic morphological disambiguation feature CSVs.

For each input word, outputs:
1. counts_<word>.csv
   - number of analyses before diacritics
   - number left by each diacritic alone
   - number left cumulatively

2. features_<word>.csv
   - available analyzer feature values pre-diacritics
   - after each diacritic alone
   - after cumulative diacritics

Usage:
    python diac_featurization.py رُبَّ
    python diac_featurization.py رُبَّ أَمْرُ
"""

import csv
import os
import sys
from collections import defaultdict

from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.utils.charsets import AR_DIAC_CHARSET


# ── Constants ──────────────────────────────────────────────────────────────

DIACRITICS = set(AR_DIAC_CHARSET)

FATHA    = 'َ'
KASRA    = 'ِ'
DAMMA    = 'ُ'
SUKUN    = 'ْ'
SHADDA   = 'ّ'
TANWIN_D = 'ً'
TANWIN_N = 'ٌ'
TANWIN_G = 'ٍ'

DIAC_NAME = {
    FATHA:    'Fatha',
    KASRA:    'Kasra',
    DAMMA:    'Damma',
    SUKUN:    'Sukun',
    SHADDA:   'Shadda',
    TANWIN_D: 'Fathatan',
    TANWIN_N: 'Dammatan',
    TANWIN_G: 'Kasratan',
}

FEATURES = [
    'pos', 'vox', 'mod', 'gen', 'asp', 'num', 'stt', 'cas', 'rat',
    'prc0', 'prc1', 'prc2', 'prc3', 'per', 'enc0', 
]
# 'pattern', 'gloss'
DB_PATH = '/home/nour.rabih/Readability-morph/camel_morph_msa_v1.0.db'
OUTPUT_DIR = '/home/nour.rabih/diac'


# ── Arabic helpers ─────────────────────────────────────────────────────────

def normalize_shadda_order(text):
    """
    Reorder diacritics so shadda always precedes the following vowel
    at each consonant position.
    """
    result = []
    i = 0

    while i < len(text):
        ch = text[i]

        if ch not in DIACRITICS:
            result.append(ch)
            i += 1
        else:
            diacs = []

            while i < len(text) and text[i] in DIACRITICS:
                diacs.append(text[i])
                i += 1

            shadda_chars = [d for d in diacs if d == SHADDA]
            other_chars = [d for d in diacs if d != SHADDA]

            result.extend(shadda_chars + other_chars)

    return ''.join(result)


def strip_diacritics(text):
    return ''.join(c for c in text if c not in DIACRITICS)


def extract_diacritics(text):
    """
    Return list of:
        (consonant_index, diacritic_char)

    consonant_index is 0-based.
    """
    result = []
    con_idx = -1

    for ch in text:
        if ch not in DIACRITICS:
            con_idx += 1
        else:
            result.append((con_idx, ch))

    return result


def apply_n_diacritics(diacritized, n):
    """
    Return the word with only the first n diacritics kept.
    """
    result = []
    seen = 0

    for ch in diacritized:
        if ch in DIACRITICS:
            if seen < n:
                result.append(ch)
                seen += 1
        else:
            result.append(ch)

    return ''.join(result)


def apply_single_diacritic(base, consonant_position, diacritic):
    """
    Return base word with exactly one diacritic added at consonant_position.

    consonant_position is 0-based.
    """
    result = []
    con_idx = -1

    for ch in base:
        result.append(ch)

        if ch not in DIACRITICS:
            con_idx += 1
            if con_idx == consonant_position:
                result.append(diacritic)

    return ''.join(result)


def is_consistent(candidate_diac, partial):
    """
    True if candidate_diac contains all diacritics in partial
    at the same consonant positions.
    """
    if strip_diacritics(candidate_diac) != strip_diacritics(partial):
        return False

    partial_by_pos = defaultdict(set)
    cand_by_pos = defaultdict(set)

    for pos, d in extract_diacritics(partial):
        partial_by_pos[pos].add(d)

    for pos, d in extract_diacritics(candidate_diac):
        cand_by_pos[pos].add(d)

    for pos, diacs in partial_by_pos.items():
        if not diacs.issubset(cand_by_pos[pos]):
            return False

    return True


# ── Feature helpers ────────────────────────────────────────────────────────

def get_feature_values(analyses, feature):
    """
    Return sorted feature values for a given feature.
    """
    values = {str(a.get(feature, 'na')) for a in analyses}
    return sorted(values)


def format_values(values):
    """
    Store feature values as a readable string in CSV.
    """
    if not values:
        return ''

    return ' | '.join(values)


def summarize_feature_values(analyses):
    """
    Return:
        feature -> sorted list of values
    """
    return {
        feature: get_feature_values(analyses, feature)
        for feature in FEATURES
    }


def filter_analyses_by_partial(all_analyses, partial_form):
    """
    Keep only analyses whose diacritized form is consistent with partial_form.
    """
    return [
        analysis
        for analysis in all_analyses
        if is_consistent(analysis.get('diac', ''), partial_form)
    ]


# ── Main feature extraction ────────────────────────────────────────────────

def build_disambiguation_features(target, all_analyses):
    """
    Build stage-level analysis information.

    For each diacritic stage, we compute:

    1. single_analyses:
       analyses left when considering only the newly added diacritic.

    2. cumulative_analyses:
       analyses left when considering all diacritics up to this stage.

    Returns list of stage dictionaries.
    """
    target = normalize_shadda_order(target)
    base = strip_diacritics(target)
    diacs = extract_diacritics(target)

    initial_count = len(all_analyses)

    stages = []

    for i, (con_pos, diac_char) in enumerate(diacs, start=1):
        single_form = apply_single_diacritic(base, con_pos, diac_char)
        cumulative_form = apply_n_diacritics(target, i)

        single_analyses = filter_analyses_by_partial(all_analyses, single_form)
        cumulative_analyses = filter_analyses_by_partial(all_analyses, cumulative_form)

        stage = {
            'word': target,
            'base': base,
            'stage': i,
            'diacritic': diac_char,
            'diacritic_name': DIAC_NAME.get(diac_char, diac_char),
            'consonant_position_0based': con_pos,
            'consonant_position_1based': con_pos + 1,
            'single_form': single_form,
            'cumulative_form': cumulative_form,
            'initial_count': initial_count,
            'single_count': len(single_analyses),
            'cumulative_count': len(cumulative_analyses),
            'single_removed': initial_count - len(single_analyses),
            'cumulative_removed': initial_count - len(cumulative_analyses),
            'single_analyses': single_analyses,
            'cumulative_analyses': cumulative_analyses,
        }

        stages.append(stage)

    return stages


# ── CSV writers ────────────────────────────────────────────────────────────

# def write_counts_csv(target, stages, output_path):
#     """
#     Write one row per added diacritic.

#     This is the main numerical feature table.
#     """
#     fieldnames = [
#         'word',
#         'base',
#         'stage',
#         'diacritic',
#         'diacritic_name',
#         'consonant_position_0based',
#         'consonant_position_1based',
#         'single_form',
#         'cumulative_form',
#         'initial_count',
#         'single_count',
#         'cumulative_count',
#         'single_removed',
#         'cumulative_removed',
#         'single_remaining_ratio',
#         'cumulative_remaining_ratio',
#     ]

#     with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()

#         for s in stages:
#             initial = s['initial_count']

#             if initial == 0:
#                 single_ratio = 0
#                 cumulative_ratio = 0
#             else:
#                 single_ratio = s['single_count'] / initial
#                 cumulative_ratio = s['cumulative_count'] / initial

#             writer.writerow({
#                 'word': s['word'],
#                 'base': s['base'],
#                 'stage': s['stage'],
#                 'diacritic': s['diacritic'],
#                 'diacritic_name': s['diacritic_name'],
#                 'consonant_position_0based': s['consonant_position_0based'],
#                 'consonant_position_1based': s['consonant_position_1based'],
#                 'single_form': s['single_form'],
#                 'cumulative_form': s['cumulative_form'],
#                 'initial_count': s['initial_count'],
#                 'single_count': s['single_count'],
#                 'cumulative_count': s['cumulative_count'],
#                 'single_removed': s['single_removed'],
#                 'cumulative_removed': s['cumulative_removed'],
#                 'single_remaining_ratio': round(single_ratio, 6),
#                 'cumulative_remaining_ratio': round(cumulative_ratio, 6),
#             })

#     print(f"  → {output_path}")


# def write_features_csv(target, all_analyses, stages, output_path):
#     """
#     Write long-format feature table.

#     Each row = one feature at one diacritic stage.
#     """
#     fieldnames = [
#         'word',
#         'base',
#         'stage',
#         'diacritic',
#         'diacritic_name',
#         'consonant_position_1based',
#         'single_form',
#         'cumulative_form',
#         'feature',

#         'pre_num_values',
#         'single_num_values',
#         'cumulative_num_values',

#         'pre_values',
#         'single_values',
#         'cumulative_values',

#         'single_fixed',
#         'cumulative_fixed',
#         'single_narrowed',
#         'cumulative_narrowed',
#     ]

#     pre_feature_values = summarize_feature_values(all_analyses)

#     with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()

#         for s in stages:
#             single_feature_values = summarize_feature_values(s['single_analyses'])
#             cumulative_feature_values = summarize_feature_values(s['cumulative_analyses'])

#             for feature in FEATURES:
#                 pre_vals = pre_feature_values[feature]
#                 single_vals = single_feature_values[feature]
#                 cumulative_vals = cumulative_feature_values[feature]

#                 pre_num = len(pre_vals)
#                 single_num = len(single_vals)
#                 cumulative_num = len(cumulative_vals)

#                 writer.writerow({
#                     'word': s['word'],
#                     'base': s['base'],
#                     'stage': s['stage'],
#                     'diacritic': s['diacritic'],
#                     'diacritic_name': s['diacritic_name'],
#                     'consonant_position_1based': s['consonant_position_1based'],
#                     'single_form': s['single_form'],
#                     'cumulative_form': s['cumulative_form'],
#                     'feature': feature,

#                     'pre_num_values': pre_num,
#                     'single_num_values': single_num,
#                     'cumulative_num_values': cumulative_num,

#                     'pre_values': format_values(pre_vals),
#                     'single_values': format_values(single_vals),
#                     'cumulative_values': format_values(cumulative_vals),

#                     # fixed = only one possible value remains
#                     'single_fixed': int(single_num == 1),
#                     'cumulative_fixed': int(cumulative_num == 1),

#                     # narrowed = fewer possible values than before diacritics
#                     'single_narrowed': int(single_num < pre_num),
#                     'cumulative_narrowed': int(cumulative_num < pre_num),
#                 })

#     print(f"  → {output_path}")

# this one keeps the values of the features
# def write_compact_features_csv(target, all_analyses, stages, output_path):
#     """
#     Write compact feature values in one row per word.

#     For each analyzer feature, this stores:
#     - pre-diacritic values
#     - cumulative post-diacritic values after each stage
#     - number of values before/after
#     - whether the feature becomes fixed after each stage
#     """

#     row = {
#         'word': target,
#         'base': strip_diacritics(target),
#         'initial_count': len(all_analyses),
#         'num_diacritics': len(stages),
#     }

#     pre_feature_values = summarize_feature_values(all_analyses)

#     for feature in FEATURES:
#         pre_vals = pre_feature_values[feature]

#         row[f'{feature}_pre_n'] = len(pre_vals)
#         row[f'{feature}_pre_values'] = format_values(pre_vals)

#         for s in stages:
#             i = s['stage']

#             cumulative_feature_values = summarize_feature_values(
#                 s['cumulative_analyses']
#             )

#             cumulative_vals = cumulative_feature_values[feature]

#             row[f'd{i}_{feature}_post_n'] = len(cumulative_vals)
#             row[f'd{i}_{feature}_post_values'] = format_values(cumulative_vals)
#             row[f'd{i}_{feature}_fixed'] = int(len(cumulative_vals) == 1)
#             row[f'd{i}_{feature}_narrowed'] = int(len(cumulative_vals) < len(pre_vals))

#     fieldnames = list(row.keys())

#     with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerow(row)

#     print(f"  → {output_path}")

def write_compact_features_csv(target, all_analyses, stages, output_path):
    """
    Write compact numeric feature summaries in one row per word.

    Keeps only meaningful numeric columns:
    - number of possible values before diacritics
    - number of possible values after each cumulative diacritic stage
    - absolute reduction
    - reduction ratio

    Does not store:
    - actual feature values
    - fixed flags
    - narrowed flags
    """

    row = {
        'word': target,
        'base': strip_diacritics(target),
        'initial_count': len(all_analyses),
        'num_diacritics': len(stages),
    }

    pre_feature_values = summarize_feature_values(all_analyses)

    for feature in FEATURES:
        pre_vals = pre_feature_values[feature]
        pre_n = len(pre_vals)

        row[f'{feature}_pre_n'] = pre_n

        for s in stages:
            i = s['stage']

            cumulative_feature_values = summarize_feature_values(
                s['cumulative_analyses']
            )

            post_vals = cumulative_feature_values[feature]
            post_n = len(post_vals)

            reduction = pre_n - post_n
            reduction_ratio = reduction / pre_n if pre_n else 0

            row[f'd{i}_{feature}_post_n'] = post_n
            row[f'd{i}_{feature}_reduction'] = reduction
            row[f'd{i}_{feature}_reduction_ratio'] = round(reduction_ratio, 6)

    fieldnames = list(row.keys())

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"  → {output_path}")
# ── Optional: one combined row per word ─────────────────────────────────────

def write_wide_counts_csv(target, stages, output_path):
    """
    Optional wide-format count features.

    Useful if you want one row per word for ML.
    Example columns:
        d1_single_count, d1_cumulative_count, d2_single_count, ...
    """
    row = {
        'word': target,
        'base': strip_diacritics(target),
        'initial_count': stages[0]['initial_count'] if stages else 0,
        'num_diacritics': len(stages),
    }

    for s in stages:
        i = s['stage']
        row[f'd{i}_diacritic'] = s['diacritic']
        row[f'd{i}_diacritic_name'] = s['diacritic_name']
        row[f'd{i}_position'] = s['consonant_position_1based']
        row[f'd{i}_single_count'] = s['single_count']
        row[f'd{i}_cumulative_count'] = s['cumulative_count']
        row[f'd{i}_single_removed'] = s['single_removed']
        row[f'd{i}_cumulative_removed'] = s['cumulative_removed']

    fieldnames = list(row.keys())

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"  → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    words = sys.argv[1:] if len(sys.argv) > 1 else ['ذَهب']

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading morphology database from {DB_PATH} ...")
    db = MorphologyDB(DB_PATH, 'a')
    analyzer = Analyzer(db, 'NONE', cache_size=100_000)

    for word in words:
        word = normalize_shadda_order(word)
        base = strip_diacritics(word)

        # Important:
        # Analyze the undiacritized base form, then filter using diacritics.
        # This gives the full ambiguity space before diacritics.
        all_analyses = analyzer.analyze(base)

        if not all_analyses:
            print(f"  No analyses found for {word!r} / base={base!r}, skipping.")
            continue

        print(f"Processing {word!r} | base={base!r} | initial analyses={len(all_analyses)}")

        stages = build_disambiguation_features(word, all_analyses)

        # safe_name = base.replace('/', '_')

        # counts_path = os.path.join(OUTPUT_DIR, f'counts_{safe_name}.csv')
        # features_path = os.path.join(OUTPUT_DIR, f'features_{safe_name}.csv')
        # wide_counts_path = os.path.join(OUTPUT_DIR, f'wide_counts_{safe_name}.csv')
        safe_name = base.replace('/', '_')

        wide_counts_path = os.path.join(OUTPUT_DIR, f'wide_counts_{safe_name}.csv')
        compact_features_path = os.path.join(OUTPUT_DIR, f'compact_features_{safe_name}.csv')

        write_wide_counts_csv(word, stages, wide_counts_path)
        write_compact_features_csv(word, all_analyses, stages, compact_features_path)
        # write_counts_csv(word, stages, counts_path)
        # write_features_csv(word, all_analyses, stages, features_path)
        # write_wide_counts_csv(word, stages, wide_counts_path)


if __name__ == '__main__':
    main()