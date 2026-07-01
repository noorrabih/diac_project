"""
Generate Arabic morphological disambiguation CSV tables.

Usage:
    python generate_disambiguation_csv.py رُبَّ
    python generate_disambiguation_csv.py رُبَّ أَمْرُ ...
"""

import csv
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
    FATHA:    'Fatha (َ)',
    KASRA:    'Kasra (ِ)',
    DAMMA:    'Damma (ُ)',
    SUKUN:    'Sukun (ْ)',
    SHADDA:   'Shadda (ّ)',
    TANWIN_D: 'Fathatan (ً)',
    TANWIN_N: 'Dammatan (ٌ)',
    TANWIN_G: 'Kasratan (ٍ)',
}

FEATURES = [
    'pos', 'vox', 'mod', 'gen', 'asp', 'num', 'stt', 'cas', 'rat',
    'prc0', 'prc1', 'prc2', 'prc3', 'per', 'enc0', 'pattern', 'gloss'
]

STATUS_LABEL = {
    'fixed':    'fixed ✓',
    'locked':   'locked ✓',
    'narrowed': 'narrowed ↓',
    'open':     'open',
}

DB_PATH = '/Users/nour.rabih/Desktop/arwi/calima-msa-s31.db'

# ── Arabic helpers ─────────────────────────────────────────────────────────

def normalize_shadda_order(text):
    """Reorder diacritics so shadda always precedes the following vowel at each position."""
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
            other_chars  = [d for d in diacs if d != SHADDA]
            result.extend(shadda_chars + other_chars)
    return ''.join(result)


def strip_diacritics(text):
    return ''.join(c for c in text if c not in DIACRITICS)


def extract_diacritics(text):
    """Return list of (consonant_index, diacritic_char) pairs (0-based)."""
    result = []
    con_idx = -1
    for ch in text:
        if ch not in DIACRITICS:
            con_idx += 1
        else:
            result.append((con_idx, ch))
    return result


def apply_n_diacritics(diacritized, n):
    """Return the word with only the first n diacritics kept."""
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


def is_consistent(candidate_diac, partial):
    """True if candidate_diac contains all diacritics in partial at the same positions."""
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

# ── Stage analysis ─────────────────────────────────────────────────────────

def analyze_word(target, all_analyses):
    """
    Returns a list of dicts, one per diacritic stage (stage 0 = no diacritics).
    Each dict has: n, partial_form, diacritic_added, diacritic_position,
                   analyses (list), feat_vals (dict feat -> set of str values).
    """
    base = strip_diacritics(target)
    diacs = extract_diacritics(target)

    def feat_vals(analyses):
        return {f: {str(a.get(f, 'na')) for a in analyses} for f in FEATURES}

    stages = [{
        'n': 0,
        'partial_form': base,
        'diacritic_added': None,
        'diacritic_position': None,
        'analyses': all_analyses,
        'feat_vals': feat_vals(all_analyses),
    }]

    for i in range(1, len(diacs) + 1):
        partial = apply_n_diacritics(target, i)
        con_pos, diac_char = diacs[i - 1]
        surviving = [a for a in all_analyses if is_consistent(a.get('diac', ''), partial)]
        stages.append({
            'n': i,
            'partial_form': partial,
            'diacritic_added': diac_char,
            'diacritic_position': con_pos,
            'analyses': surviving,
            'feat_vals': feat_vals(surviving),
        })

    return stages

# ── Status & value formatting ──────────────────────────────────────────────

def get_status(stages, stage_idx, feat):
    current = stages[stage_idx]['feat_vals'][feat]
    initial = stages[0]['feat_vals'][feat]

    if len(initial) == 1:
        return 'fixed'
    if len(current) == 1:
        return 'locked'
    if stage_idx == 0:
        return 'open'
    prev = stages[stage_idx - 1]['feat_vals'][feat]
    if len(current) < len(prev):
        return 'narrowed'
    return 'open'


def format_values(vals, feat):
    if len(vals) == 1:
        return next(iter(vals))
    vals_sorted = sorted(vals)
    if feat == 'pattern' and len(vals) > 3:
        return f"{{{vals_sorted[0]} … {vals_sorted[-1]}}} ×{len(vals)}"
    return '{' + ', '.join(vals_sorted) + '}'

# ── CSV writer ─────────────────────────────────────────────────────────────

def write_disambiguation_csv(target, all_analyses, output_path):
    stages = analyze_word(target, all_analyses)
    n = len(stages)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Row 1: title
        writer.writerow([f'Arabic Morphological Disambiguation: {target}'] + [''] * (n * 2))

        # Row 2: stage headers (stage text spans Status+Values columns visually)
        row = ['']
        for s in stages:
            if s['n'] == 0:
                header = f"Stage 0\nNo diacritics ({s['partial_form']})\n{len(s['analyses'])} analyses"
            else:
                con = s['diacritic_position'] + 1
                dname = DIAC_NAME.get(s['diacritic_added'], s['diacritic_added'])
                header = (
                    f"Stage {s['n']}\n"
                    f"{dname} on C{con} ({s['partial_form']})\n"
                    f"{len(s['analyses'])} analyses"
                )
            row.extend([header, ''])
        writer.writerow(row)

        # Row 3: column headers
        writer.writerow(['Feature'] + ['Status', 'Values'] * n)

        # Feature rows
        for feat in FEATURES:
            row = [feat]
            for i, s in enumerate(stages):
                status = get_status(stages, i, feat)
                vals = s['feat_vals'][feat]
                row.extend([STATUS_LABEL[status], format_values(vals, feat)])
            writer.writerow(row)

        # Empty separator row
        writer.writerow([])

        # Surviving forms row — deduplicated by lex+pos (same as notebook)
        row = ['Surviving Forms']
        for s in stages:
            unique_forms = sorted({
                f"{a.get('diac', '?')} ({a.get('pos', '?')})"
                for a in s['analyses']
            })
            # Wrap at 2 forms per line for readability
            lines = [
                ', '.join(unique_forms[j:j + 2])
                for j in range(0, len(unique_forms), 2)
            ]
            row.extend(['\n'.join(lines), ''])
        writer.writerow(row)

    print(f"  → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    words = sys.argv[1:] if len(sys.argv) > 1 else ['ذَهب']

    print(f"Loading morphology database from {DB_PATH} ...")
    db = MorphologyDB(DB_PATH, 'a')
    analyzer = Analyzer(db, 'NONE', cache_size=100_000)

    output_dir = '/Users/nour.rabih/Desktop/Diac/diac_project'

    for word in words:
        word = normalize_shadda_order(word)
        base = word
        analyses = analyzer.analyze(word)
        if not analyses:
            print(f"  No analyses found for {word!r}, skipping.")
            continue
        out = f'{output_dir}/disambiguation_{base}.csv'
        print(f"Processing {word!r}  ({len(analyses)} analyses)  base={base!r}")
        write_disambiguation_csv(word, analyses, out)


if __name__ == '__main__':
    main()
