"""
Word-level and subsentence-level Arabic feature extraction.

Word-level features: every morphological feature drafted in word_feats.py
(itself a superset of disambig_features.py's extract_word_features, plus the
per-word-only checks that disambig_features.py's analyze_sentence used to
aggregate to sentence level). Extraction logic for each feature mirrors
disambig_features.py exactly; word_feats.py's syntax errors/orphaned
fragments (noun_num, noon_niswa) are fixed here.

Subsentence-level (syntactic) features: every pattern in
parser_features.py's extract_features(), generalized from "first match only,
one boolean per sentence" to "every match, emitted as its own subsentence"
(a token + the other token(s) participating in the dependency relation,
e.g. a PRT/prep token + its OBJ child for jar_majroor).

IDs:
    word id:        f"{sentence_id}_{i}"      i = 1-indexed word position
    subsentence id:  f"{sentence_id}_sub_{i}"  i = 1-indexed match counter,
                                                shared across all pattern
                                                types, in token-scan order
"""

# TODO check nominal_sentence and advanced_khabar
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.charsets import AR_DIAC_CHARSET

from camel_parser.src.classes import TextParams
from camel_parser.src.conll_output import text_tuples_to_string
from camel_parser.src.data_preparation import get_tagset, parse_text
from camel_parser.src.initialize_disambiguator.disambiguator_interface import get_disambiguator

from conllx_df.src.conllx_df import ConllxDf
from conllx_df.src.conll_utils import get_token_details, add_parent_details, add_direction

tqdm.pandas()

# --------------------------------------------------------------------------
# Shared constants (from disambig_features.py / parser_features.py)
# --------------------------------------------------------------------------

WEAK_FINAL_RE = re.compile(r'[يىا]$')
qassam_lex = ['حَقّ', 'رَبّ', 'شَمْس', 'قَمَر', 'تِين', 'اللّٰه', 'حَياة', 'مَوْت', 'دِين', 'قُرْآن',
              'كَعْبَة', 'سَماء', 'عَصْر', 'فَجْر', 'لَيْل', 'أَرْض', 'كَعْبَة', 'جَبَل', 'مَلْأَك',
              'جَبْرَئِيل', 'ميكائيل', 'مُحَمَّد', 'مُوسَى', 'عِيسَى', 'إِبْرَاهِيم', 'نُوح', 'يُونُس',
              'يُوسُف', 'يَعْقُوب', 'إِسْمَاعِيل', 'إِسْحَاق', 'إِدْرِيس', 'نَبِيّ']
lex_list = ['ثُمَّ', 'حَتَّى', 'أَوْ', 'أَمْ', 'لٰكِنَّ', 'لٰكِنْ', 'أَمّا']
tanween = ['ً', 'ٍ', 'ٌ']
vowels = ['a', 'e', 'i', 'o', 'u', 'aa', 'ee', 'ii', 'oo', 'uu']
arabic_vowels = ['ي', 'و', 'ا', 'ى', 'ؤ', 'ئ', 'إ', 'أ', 'آ', 'ة']

kana_set = ["كان", "صار", "أصبح", "زال", "ليس", "أمسى", "بات", "ظل", "أضحى", "برح", "فتئ", "دام", "انفك"]
inna_w_akhawataha = ['إن', 'أن', 'كأن', 'لكن', 'لعل', 'ليت']

# --------------------------------------------------------------------------
# Disambiguator setup (word-level), same model as disambig_features.py
# --------------------------------------------------------------------------

db = MorphologyDB("/home/nour.rabih/Readability-morph/camel_morph_msa_v1.0.db", "a")
analyzer = Analyzer(db, 'ADD_PROP', cache_size=100000)
bert = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False)
bert._analyzer = analyzer


@lru_cache(maxsize=1024)
def disambiguate_sentence(sentence_text):
    sentence = simple_word_tokenize(sentence_text)
    return bert.tag_sentence(sentence)


# --------------------------------------------------------------------------
# Parser setup (subsentence-level), same model as parser_features.py
# --------------------------------------------------------------------------

_model_path = Path("camel_parser/models")
_model_name = "CAMeLBERT-CATiB-biaffine.model"
_arclean = CharMapper.builtin_mapper("arclean")
_clitic_feats_df = pd.read_csv(
    '/home/nour.rabih/Readability-interpretability/camel_parser/data/clitic_feats.csv'
).astype(str).astype(object)
_tagset = get_tagset("catib")
_parser_disambiguator = get_disambiguator("bert", "calima-msa-s31")


def _parse_sentences_to_conll(sentences: list) -> ConllxDf:
    """Parse one or more sentences into a ConllxDf (no temp files).
    For a batch, use get_df_by_id(i) to retrieve each sentence's DataFrame."""
    cleaned = [_arclean(s) for s in sentences]
    params = TextParams(
        cleaned, _model_path / _model_name, _arclean,
        _parser_disambiguator, _clitic_feats_df, _tagset, ""
    )
    parsed = parse_text("text", params)
    lines = text_tuples_to_string(parsed, file_type="conll", sentences=cleaned)
    conll_data = '\n'.join(str(l) for l in lines[:-1])
    return ConllxDf(data=conll_data)


# --------------------------------------------------------------------------
# Word-level features
# --------------------------------------------------------------------------

def count_syllables(caphi, diac, prc0, prc2):
    syllable_count = 0
    if caphi is not None:
        caphi_parts = caphi.split('_')
        for i, part in enumerate(caphi_parts):
            if part in vowels:
                if i == len(caphi_parts) - 1:
                    if diac and diac[-1] in AR_DIAC_CHARSET:
                        continue
                syllable_count += 1
        if len(caphi_parts) > 2:
            if caphi_parts[-1] == 'n' and caphi_parts[-2] in vowels and diac[-1] in tanween:
                syllable_count -= 1
        if prc0 == 'Al_det':
            syllable_count -= 1
        if prc2 == 'wa_conj':
            syllable_count -= 1
    else:
        for ch in diac:
            if ch in arabic_vowels:
                syllable_count += 1
    return syllable_count


def extract_word_features(word_analysis, raw_word=''):
    """All word-level features drafted in word_feats.py, extracted from a
    single word's morphological analysis dict (plus the raw surface form,
    needed only for noon_niswa)."""
    pos = word_analysis.get('pos', '')
    num = word_analysis.get('num', '')
    asp = word_analysis.get('asp', '')
    form_gen = word_analysis.get('form_gen', '')
    form_num = word_analysis.get('form_num', '')
    enc0 = word_analysis.get('enc0', '')
    prc0 = word_analysis.get('prc0', '')
    prc1 = word_analysis.get('prc1', '')
    prc2 = word_analysis.get('prc2', '')
    prc3 = word_analysis.get('prc3', '')
    bw = word_analysis.get('bw', '')
    lex = word_analysis.get('lex', '')
    vox = word_analysis.get('vox', '')
    diac = word_analysis.get('diac', '')
    caphi = word_analysis.get('caphi')

    def weak_final():
        if pos not in ('noun', 'verb'):
            return False
        if enc0 and enc0 != '0':
            return False
        stem = word_analysis.get('stem', '')
        stem_nodiac = ''.join(ch for ch in stem if ch not in AR_DIAC_CHARSET)
        return bool(WEAK_FINAL_RE.search(stem_nodiac))

    def imperfective_singular():
        return num == 's' and asp == 'i' and pos == 'verb'

    def prc_Al_det():
        return prc0 == 'Al_det'

    def prc_waw():
        return prc2 == 'wa_conj'

    def suf_1s_pron():
        return enc0 in ['1s_pron', '1s_poss', '1s_dobj']

    def verb_present_plural():
        return pos == 'verb' and asp == 'i' and num == 'p'

    def prc_prep():
        return prc1 in ('bi_prep', 'li_prep', 'ka_prep')

    def suf_pron():
        prefixes = ["1p", "2ms", "2fs", "2mp", "2fp", "2p", "3ms", "3fs", "3mp", "3fp", "3p"]
        suffixes = ["dobj", "poss", "pron"]
        valid_enc0_values = [f"{prefix}_{suffix}" for prefix in prefixes for suffix in suffixes]
        return enc0 in valid_enc0_values

    def dual_noun_adj():
        return num == 'd' and pos in ['noun', 'adj', 'noun_quant', 'adj_comp']

    def plural_fem_noun_adj():
        return form_num == 'p' and form_gen == 'f'

    def verb_past_s_p():
        return pos == 'verb' and asp == 'p' and num in ['s', 'p']

    def plural_masc():
        return form_gen == 'm' and form_num == 'p' and pos in ['noun', 'adj']

    def verb_past_dual():
        return pos == 'verb' and asp == 'p' and num == 'd'

    def verb_present_dual():
        return pos == 'verb' and asp == 'i' and num == 'd'

    def verb_command():
        return pos == 'verb' and asp == 'c' and num == 's'

    def suf_dual_pron():
        prefixes = ["2d", "3d"]
        suffixes = ["dobj", "poss", "pron"]
        valid_enc0_values = [f"{prefix}_{suffix}" for prefix in prefixes for suffix in suffixes]
        return enc0 in valid_enc0_values

    def broken_plural():
        return pos in ['noun', 'adj'] and form_num == 's' and num == 'p'

    def noon_niswa():
        word_nodiac = ''.join(ch for ch in raw_word if ch not in AR_DIAC_CHARSET)
        return bool(pos in ('verb', 'noun') and num == 'p' and word_nodiac
                     and word_nodiac.endswith('ن') and '3FP' in bw)


    def waw_alqassam():
        return prc2 in ['wa_prep'] and lex in qassam_lex

    def verb_command_plural():
        return asp == 'c' and num == 'p' and pos == 'verb'

    def amma_lakin():
        return lex in lex_list

    def verb_command_dual():
        return asp == 'c' and num == 'd' and pos == 'verb'

    def interrogative_alif():
        return prc3 in ['>a_ques', 'A_ques']

    def ba_alqassam():
        return prc1 in ['bi_prep'] and lex in qassam_lex

    def passive_voice():
        return vox == 'p' and pos == 'verb'

    def proper_noun():
        return pos == 'noun_prop'

    def pronoun():
        return pos == 'pron'

    def akho_abo():
        akho = (lex == 'أَخ' and pos == 'noun' and diac == 'أَخُو')
        abo = (lex == 'أَب' and pos == 'noun' and diac == 'أَبُو')
        return akho or abo

    def verb():
        return pos == 'verb'

    def adj():
        return pos == 'adj'

    def noun_num():
        return pos == 'noun_num'

    def demonstrative_pronoun_singular():
        return pos == 'pron_dem' and num == 's'

    def preposition():
        return pos == 'prep'

    def ordinal_number():
        return num == 'adj_num'

    def demonstrative_pronoun_plural_dual():
        return pos == 'pron_dem' and num in ('p', 'd')

    def negation_particle():
        return pos == 'part_neg'

    def relative_pronoun_singular():
        return pos == 'pron_rel' and num == 's'

    def interrogative_tools():
        return pos in ('adv_interrog', 'pron_interrog')

    def hal_interrog():
        return lex == 'هَلْ' and pos == 'part_interrog'

    def other_proclitics():
        for prc in (prc0, prc1, prc2, prc3):
            if not prc:
                continue
            if prc.startswith('s_') or prc.startswith('sa_'):
                return True
            if prc.startswith('w_') and prc != 'wa_prep':
                return True
            if prc.startswith('f_'):
                return True
        return False
    # if word i punctuation, skip
    if pos == 'punc':
        return {}
    return {
        'weak_final': weak_final(),
        'imperfective_singular': imperfective_singular(),
        'prc_Al_det': prc_Al_det(),
        'prc_waw': prc_waw(),
        'suf_1s_pron': suf_1s_pron(),
        'verb_present_plural': verb_present_plural(),
        'prc_prep': prc_prep(),
        'suf_pron': suf_pron(),
        'dual_noun_adj': dual_noun_adj(),
        'plural_fem_noun_adj': plural_fem_noun_adj(),
        'verb_past_s_p': verb_past_s_p(),
        'plural_masc': plural_masc(),
        'verb_past_dual': verb_past_dual(),
        'verb_present_dual': verb_present_dual(),
        'verb_command': verb_command(),
        'suf_dual_pron': suf_dual_pron(),
        'broken_plural': broken_plural(),
        'noon_niswa': noon_niswa(),
        'waw_alqassam': waw_alqassam(),
        'verb_command_plural': verb_command_plural(),
        'amma_lakin': amma_lakin(),
        'verb_command_dual': verb_command_dual(),
        'interrogative_alif': interrogative_alif(),
        'ba_alqassam': ba_alqassam(),
        'passive_voice': passive_voice(),
        'proper_noun': proper_noun(),
        'pronoun': pronoun(),
        'akho_abo': akho_abo(),
        'verb': verb(),
        'adj': adj(),
        'noun_num': noun_num(),
        'demonstrative_pronoun_singular': demonstrative_pronoun_singular(),
        'preposition': preposition(),
        'ordinal_number': ordinal_number(),
        'demonstrative_pronoun_plural_dual': demonstrative_pronoun_plural_dual(),
        'negation_particle': negation_particle(),
        'relative_pronoun_singular': relative_pronoun_singular(),
        'interrogative_tools': interrogative_tools(),
        'hal_interrog': hal_interrog(),
        'other_proclitics': other_proclitics(),
        'syllable_count': count_syllables(caphi, diac, prc0, prc2),
        'diac_count': sum(1 for ch in diac if ch in AR_DIAC_CHARSET),
        'diac_count_no_tanween': sum(1 for ch in diac if ch in AR_DIAC_CHARSET and ch not in tanween),
        'lex': lex,
        'pos': pos,
        'diac': diac,
    }


def extract_word_feats(df: pd.DataFrame) -> pd.DataFrame:
    def process_row(row):
        diac_count = row['diac_count']
        if  diac_count > 0 :
            sentence_id = row['ID']
            sentence = row['Sentence']
            if len(sentence.split(" ")) == 1:
                sentence = '"' + sentence + '"'
            sentence_analysis = disambiguate_sentence(sentence)
            tokenized_raw = simple_word_tokenize(sentence)
            return [
                {'ID': f"{sentence_id}_{i}", 'sentence_id': sentence_id,
                'word_index': i, 'word': raw_word,
                **extract_word_features(word_analysis, raw_word)}
                for i, (word_analysis, raw_word) in enumerate(zip(sentence_analysis, tokenized_raw), start=1)
            ]
        else:
            print(f"skipping {row['ID']} no diac")
            
    results = df.progress_apply(process_row, axis=1)
    return pd.DataFrame([word for row in results if row is not None for word in row])




# --------------------------------------------------------------------------
# Subsentence-level (syntactic) features
# --------------------------------------------------------------------------

def _feats(sen_df, token_id):
    vals = sen_df.loc[sen_df['ID'] == token_id, 'FEATS'].values
    return vals[0] if len(vals) else ''


def _word_text(sen_df, token_id):
    vals = sen_df.loc[sen_df['ID'] == token_id, 'FORM'].values
    return vals[0] if len(vals) else ''


def _position(sen_df, token_id):
    idx = sen_df[sen_df['ID'] == token_id].index
    return idx[0] if len(idx) else -1


def _build_tokens(sen_df):
    tokens = {int(row.ID): get_token_details(sen_df, int(row.ID)) for row in sen_df.itertuples(index=False)}
    children_map = {token_id: [] for token_id in tokens}
    for token in tokens.values():
        add_parent_details(sen_df, token)
        add_direction(token)
        if token.head in children_map:
            children_map[token.head].append(token)
    return tokens, children_map


def _children_with_deprel(children_map, token_id, deprel):
    return [c for c in children_map.get(token_id, []) if c.deprel == deprel]


def find_subsentences(sen_df):
    """Scan a parsed sentence's dependency tree and return every match of
    every syntactic pattern in parser_features.extract_features, instead of
    just the first match per pattern. Each match is a dict listing the
    participating token ids/words and which feature it represents."""
    tokens, children_map = _build_tokens(sen_df)

    # Precompute O(1) lookups — avoids repeated O(n) DataFrame scans per token
    feats_map = dict(zip(sen_df['ID'], sen_df['FEATS']))
    form_map  = dict(zip(sen_df['ID'], sen_df['FORM']))
    pos_map   = {tid: i for i, tid in enumerate(sen_df['ID'])}

    matches = []  # list of (feature_name, [token_id, ...])

    has_tpc = any(token.deprel == "TPC" for token in tokens.values())

    # Gate for nominal_sentence / advanced_khabar, mirroring parser_features.py:
    # the whole pattern only applies if the sentence has no TPC token at all.
    nominal_pairs = []  # (PRT_token_id, SBJ_child_id)
    if not has_tpc:
        for token in tokens.values():
            if token.pos == "PRT" and token.lemma not in inna_w_akhawataha:
                for child in _children_with_deprel(children_map, token.token_id, "SBJ"):
                    if child.pos != "VRB":
                        nominal_pairs.append((token.token_id, child.token_id))

    is_nominal_sentence = len(nominal_pairs) > 0
    for prt_id, sbj_id in nominal_pairs:
        matches.append(("nominal_sentence", [prt_id, sbj_id]))

    for token in tokens.values():
        tid = token.token_id

        # 1+5. جملة فعلية بدون/مع مفعول به / تتعدى إلى مفعولين (OBJ children fetched once)
        if token.pos == "VRB":
            obj_children = _children_with_deprel(children_map, tid, "OBJ")
            if obj_children:
                matches.append(("obj", [tid] + [c.token_id for c in obj_children]))
                if len(obj_children) == 2:
                    matches.append(("verbal_sentence_with_two_objects", [tid] + [c.token_id for c in obj_children]))
            else:
                matches.append(("no_obj", [tid]))

        # 2. جار + مجرور
        if token.pos == "PRT" and "pos=prep" in feats_map.get(tid, ''):
            obj_children = _children_with_deprel(children_map, tid, "OBJ")
            if obj_children:
                matches.append(("jar_majroor", [tid] + [c.token_id for c in obj_children]))

        # 3. جمل فعلية معطوفة
        if token.pos == "VRB":
            for child in children_map[tid]:
                if child.deprel == "MOD" and child.pos == "PRT" and "pos=conj|" in feats_map.get(child.token_id, ''):
                    for grandchild in children_map[child.token_id]:
                        if grandchild.pos == "VRB" and grandchild.deprel == "OBJ":
                            matches.append(("coordinated_verbs", [tid, child.token_id, grandchild.token_id]))

        # 4. جمل فعلية (مضارعة) مع أن المصدرية
        if token.pos == "PRT" and token.lemma == "أن":
            for child in children_map[tid]:
                if child.pos == "VRB" and child.deprel == "OBJ" and "asp=i" in feats_map.get(child.token_id, ''):
                    matches.append(("verbal_present_sentence_with_an_almasdariya", [tid, child.token_id]))

        # 6. المنادى
        if token.pos == "PRT" and "pos=part_voc" in feats_map.get(tid, ''):
            obj_children = _children_with_deprel(children_map, tid, "OBJ")
            if obj_children:
                matches.append(("vocative", [tid] + [c.token_id for c in obj_children]))

        # إن وأخواتها
        if token.lemma in inna_w_akhawataha and "pos=verb_pseudo" in feats_map.get(tid, ''):
            prd_children = _children_with_deprel(children_map, tid, "PRD")
            if prd_children:
                matches.append(("inna_wa_akhawataha", [tid] + [c.token_id for c in prd_children]))

        # كان وأخواتها
        if token.pos == "VRB" and token.lemma in kana_set:
            prd_children = _children_with_deprel(children_map, tid, "PRD")
            if prd_children:
                matches.append(("kana_wa_akhawataha", [tid] + [c.token_id for c in prd_children]))

        # خبر مقدم / مبتدأ مؤخر (only applies once the sentence is nominal)
        if is_nominal_sentence:
            for child in children_map[tid]:
                if (child.deprel == "SBJ" and child.pos != "VRB"
                        and pos_map.get(tid, -1) < pos_map.get(child.token_id, -1)):
                    matches.append(("advanced_khabar", [tid, child.token_id]))

        # جملة اسمية خبرها جملة اسمية
        if token.pos != "VRB":
            tpc_children = _children_with_deprel(children_map, tid, "TPC")
            if tpc_children:
                matches.append(("nominal_with_tpc", [tid] + [c.token_id for c in tpc_children]))

        # إضافة لفظية / حقيقية
        if token.pos == "NOM":
            idf_children = _children_with_deprel(children_map, tid, "IDF")
            if idf_children:
                feature = "idafa_lafzia" if "pos=adj" in feats_map.get(tid, '') else "idafa_hakikiya"
                matches.append((feature, [tid] + [c.token_id for c in idf_children]))

    return [
        {'sub_index': i, 'feature': feature, 'token_ids': token_ids,
         'words': [form_map.get(t, '') for t in token_ids]}
        for i, (feature, token_ids) in enumerate(matches, start=1)
    ]


def extract_subsentence_feats(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    to_process = df[df['diac_count'] > 0][['ID', 'Sentence']].values.tolist()
    skipped = len(df) - len(to_process)
    if skipped:
        print(f"Skipping {skipped} sentences with no diacritics")

    all_rows = []
    for start in tqdm(range(0, len(to_process), batch_size), desc="Batches"):
        batch        = to_process[start:start + batch_size]
        sentence_ids = [r[0] for r in batch]
        sentences    = [r[1] for r in batch]

        multi = _parse_sentences_to_conll(sentences)
        for i, sentence_id in enumerate(sentence_ids):
            sen_df = multi.get_df_by_id(i)
            if sen_df is None:
                continue
            for match in find_subsentences(sen_df):
                all_rows.append({
                    'ID': f"{sentence_id}_sub_{match['sub_index']}",
                    'sentence_id': sentence_id,
                    'sub_index': match['sub_index'],
                    'feature': match['feature'],
                    'token_ids': match['token_ids'],
                    'words': match['words'],
                })

    return pd.DataFrame(all_rows)


# Example usage:
# df = pd.DataFrame({'ID': ['s1'], 'Sentence': ['ذهب الطالب إلى المدرسة']})
# word_df = extract_word_feats(df)
# print(word_df)
# sub_df = extract_subsentence_feats(df)
# print(sub_df)
from datasets import load_dataset



print("Loading BAREC dataset …")
# dataset = load_dataset("CAMeL-Lab/BAREC-Corpus-v1.0")

# INPUT_VAR = "Sentence"

# # Labels are 1-indexed in the dataset; shift to 0-indexed
# dataset_train = {
#     "text":  list(dataset["train"][INPUT_VAR]),
#     "label": [l - 1 for l in dataset["train"]["Readability_Level_19"]],
# }
# dataset_dev = {
#     "text":  list(dataset["dev"][INPUT_VAR]),
#     "label": [l - 1 for l in dataset["dev"]["Readability_Level_19"]],
# }
# dataset_test = {
#     "text":  list(dataset["test"][INPUT_VAR]),
#     "label": [l - 1 for l in dataset["test"]["Readability_Level_19"]],
# }

# print(f"Train: {len(dataset_train['text'])} | Dev: {len(dataset_dev['text'])} | Test: {len(dataset_test['text'])}")

# # run it on train
# # run it on train
# train_df = pd.DataFrame({
#     'ID': [f"train_{i}" for i in range(len(dataset_train['text']))],
#     'Sentence': dataset_train['text'],
#     'label': dataset_train['label'],
# })[:3]

# /home/nour.rabih/diac/data_analysis/1M/barec_diacritics_per_sentence.csv

train_df = pd.read_csv('/home/nour.rabih/diac/train_barec_diacritics_per_sentence_diac_only.csv')

print(f"Extracting word-level features for {len(train_df)} training sentences …")
word_feats_train = extract_word_feats(train_df)
word_feats_train.to_csv("train_word_feats_withlex.csv", index=False)
print(f"Saved train_word_feats.csv  ({len(word_feats_train)} rows)")

# print(f"Extracting subsentence-level features for {len(train_df)} training sentences …")
# sub_feats_train = extract_subsentence_feats(train_df)
# sub_feats_train.to_csv("train_subsentence_feats.csv", index=False)
# print(f"Saved train_subsentence_feats.csv  ({len(sub_feats_train)} rows)")

# # Also save the sentence-level metadata (ID + label) so it can be joined later
# train_df[['ID', 'Sentence', 'label']].to_csv("train_sentences.csv", index=False)
# print("Saved train_sentences.csv")