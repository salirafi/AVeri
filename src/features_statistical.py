from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd
from tqdm.auto import tqdm

from function_words import FUNCTION_WORDS

FUNCTION_WORD_SET = {word.lower() for word in FUNCTION_WORDS} # lowercase function words
PLACEHOLDER_RE = r"^<[^<>]+>$" # for masking placeholder



@dataclass(slots=True)
class Config:
    verbose: bool = True
    include_function_word_rate: bool = True
    exclude_placeholders_from_avg_word_length: bool = True
    phrase_role_dependency_labels: tuple[str, ...] = ("acl", "advcl", "ccomp", "pcomp", "relcl", "xcomp") # for clausal phrase signals
    pos_roles: dict[str, tuple[str, ...]] = None
    dep_roles: dict[str, tuple[str, ...]] = None

    # covering tags for en_core_web_lg spacy model
    def __post_init__(self) -> None:
        if self.pos_roles is None:
            self.pos_roles = {
                "adjective": ("ADJ",),
                "adposition": ("ADP",),
                "adverb": ("ADV",),
                "auxiliary": ("AUX",),
                "conjunction": ("CONJ",),
                "coordinating_conjunction": ("CCONJ",),
                "determiner": ("DET",),
                "interjection": ("INTJ",),
                "noun": ("NOUN",),
                "numeral": ("NUM",),
                "particle": ("PART",),
                "pronoun": ("PRON",),
                "proper_noun": ("PROPN",),
                "punctuation": ("PUNCT",),
                "subordinating_conjunction": ("SCONJ",),
                "symbol": ("SYM",),
                "verb": ("VERB",),
                "other": ("X",),
                "space": ("SPACE",),
            }
        if self.dep_roles is None:
            self.dep_roles = {
                "root": ("ROOT",),
                "adjectival_clause": ("acl",),
                "adjectival_complement": ("acomp",),
                "adverbial_clause": ("advcl",),
                "adverbial_modifier": ("advmod",),
                "agent": ("agent",),
                "adjectival_modifier": ("amod",),
                "apposition": ("appos",),
                "attribute": ("attr",),
                "auxiliary": ("aux",),
                "passive_auxiliary": ("auxpass",),
                "case_marker": ("case",),
                "coordinating_conjunction": ("cc",),
                "clausal_complement": ("ccomp",),
                "compound": ("compound",),
                "conjunct": ("conj",),
                "clausal_subject": ("csubj",),
                "passive_clausal_subject": ("csubjpass",),
                "dative": ("dative",),
                "dependency_unspecified": ("dep",),
                "determiner": ("det",),
                "direct_object": ("dobj",),
                "expletive": ("expl",),
                "indirect_object": ("iobj",),
                "interjection": ("intj",),
                "marker": ("mark",),
                "meta": ("meta",),
                "negation": ("neg",),
                "nominal_modifier": ("nmod",),
                "noun_phrase_adverbial_modifier": ("npadvmod",),
                "nominal_subject": ("nsubj",),
                "passive_nominal_subject": ("nsubjpass",),
                "numeric_modifier": ("nummod",),
                "object": ("obj",),
                "object_predicate": ("oprd",),
                "parataxis": ("parataxis",),
                "prepositional_complement": ("pcomp",),
                "object_of_preposition": ("pobj",),
                "possessive_modifier": ("poss",),
                "preconjunct": ("preconj",),
                "predeterminer": ("predet",),
                "prepositional_modifier": ("prep",),
                "particle": ("prt",),
                "punctuation": ("punct",),
                "quantifier_modifier": ("quantmod",),
                "relative_clause_modifier": ("relcl",),
                "open_clausal_complement": ("xcomp",),
            }
config = Config()



def _safe_mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)

def _safe_rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 3)

# flagging non-word tokens including both punctuations and white-space
def _word_token_indices(record: dict[str, Any]) -> list[int]:
    return [
        index for index, (is_punct, is_space) in enumerate(zip(record["token_is_punct"], record["token_is_space"], strict=False))
        if not is_punct and not is_space
    ]

# count average number of characters per word
def _avg_word_length(record: dict[str, Any], config: Config = config) -> float:
    lengths: list[int] = []
    for index, token_text in enumerate(record["tokens"]):
        if record["token_is_punct"][index] or record["token_is_space"][index]: # excluding punctuations and white spaces
            continue
        if config.exclude_placeholders_from_avg_word_length and (token_text.startswith("<") and token_text.endswith(">")): # excluding <...> placeholder from previous masking
            continue
        lengths.append(len(token_text))
    return _safe_mean(lengths)

# ========== SENTENCE STATISTICS ============

# defining a sentence
def _sentence_spans(record: dict[str, Any]) -> list[tuple[int, int]]:
    spans = record["sentence_token_spans"]
    # either compute from sentence_token_spans or number of tokens
    if spans: return spans
    if record["tokens"]: return [(0, len(record["tokens"]))]
    return [] # else no sentence

def _sentence_word_lengths(record: dict[str, Any]) -> list[int]:
    word_indices = set(_word_token_indices(record))
    sentence_lengths: list[int] = []
    for start, end in _sentence_spans(record):
        count = sum(1 for index in range(start, end) if index in word_indices)
        sentence_lengths.append(count)
    return sentence_lengths

def _sentence_function_word_counts(record: dict[str, Any]) -> list[int]:
    sentence_counts: list[int] = []
    for start, end in _sentence_spans(record):
        count = 0
        for index in range(start, end):
            if record["token_is_punct"][index] or record["token_is_space"][index]:
                continue
            if record["token_lower"][index] in FUNCTION_WORD_SET:
                count += 1
        sentence_counts.append(count)
    return sentence_counts

# ============ PHRASE STATISTICS ==============

def _phrase_role_features(record: dict[str, Any], config: Config = config) -> dict[str, float]:

    noun_phrase_count = len(record["noun_chunk_spans"]) # count noun phrases

    dependency_labels = record["token_dep"]
    prepositional_phrase_count = sum(1 for label in dependency_labels if label == "prep") # count preprositional phrases (approximated from "prep" label)

    clausal_phrase_count = sum(
        1 for label in dependency_labels if label in config.phrase_role_dependency_labels
    ) # count clausal phrases

    phrase_counts = {
        "phrase_noun_phrase_rate": noun_phrase_count,
        "phrase_prepositional_phrase_rate": prepositional_phrase_count,
        "phrase_clausal_phrase_rate": clausal_phrase_count,
    }
    total_phrase_units = sum(phrase_counts.values())

    # returning the rate for each phrase group
    return {
        feature_name: _safe_rate(count, total_phrase_units)
        for feature_name, count in phrase_counts.items()
    }

# ============ POS TAGS STATISTICS ==============

def _pos_role_features(record: dict[str, Any], config: Config = config) -> dict[str, float]:
    word_indices = _word_token_indices(record)
    pos_counts = {name: 0 for name in config.pos_roles}
    for index in word_indices:
        token_pos = record["token_pos"][index]
        for role_name, labels in config.pos_roles.items():
            if token_pos in labels:
                pos_counts[role_name] += 1
    total_pos_units = sum(pos_counts.values())
    return {
        f"pos_{role_name}_rate": _safe_rate(count, total_pos_units)
        for role_name, count in pos_counts.items()
    }

# ============ DEPENDENCY LABEL STATISTICS ==============

def _dep_role_features(record: dict[str, Any], config: Config = config) -> dict[str, float]:
    word_indices = _word_token_indices(record)
    dep_counts = {name: 0 for name in config.dep_roles}

    for index in word_indices:
        token_dep = record["token_dep"][index]
        for role_name, labels in config.dep_roles.items():
            if token_dep in labels:
                dep_counts[role_name] += 1

    total_dep_units = sum(dep_counts.values())
    return {
        f"dep_{role_name}_rate": _safe_rate(count, total_dep_units)
        for role_name, count in dep_counts.items()
    }


# ======================================================


def extract_document_statistics(record: dict[str, Any], config: Config = config) -> dict[str, float]:

    word_indices = _word_token_indices(record) # flag non-word tokens including both punctuations and white-space
    total_word_tokens = len(word_indices) # number of word tokens
    total_non_space_tokens = sum(1 for is_space in record["token_is_space"] if not is_space) # flag non-word tokens for only white-space
    total_punct_tokens = sum(1 for is_punct in record["token_is_punct"] if is_punct) # count punctuations
    total_function_words = sum(1 for index in word_indices if record["token_lower"][index] in FUNCTION_WORD_SET)

    # count sentence statistics
    sentence_lengths = _sentence_word_lengths(record)
    sentence_function_word_counts = _sentence_function_word_counts(record)

    features: dict[str, float] = {
        "avg_sentence_length_words": _safe_mean(sentence_lengths),
        "avg_function_words_per_sentence": _safe_mean(sentence_function_word_counts),
        "punctuation_rate": _safe_rate(total_punct_tokens, total_non_space_tokens),
        "avg_word_length": _avg_word_length(record, config=config),
    }

    if config.include_function_word_rate:
        features["function_word_rate"] = _safe_rate(total_function_words, total_word_tokens)

    features.update(_phrase_role_features(record, config=config)) # compute rate of phrases (from three groups)
    features.update(_pos_role_features(record, config=config)) # compute rate for POS tags
    features.update(_dep_role_features(record, config=config)) # compute rate for dependency features

    return features



def extract_split_statistics(
    df: pd.DataFrame,
    split_cache: dict[str, list[dict[str, Any]]],
    split_name: str = "",
    config: Config = config,
) -> pd.DataFrame:
    """
    Append statistical features for each configured text column in one split.
    """
    result = df.copy()

    for column in ["text1", "text2"]:

        records = split_cache[column] # linguistic_cache must contain "text1" and "text2"
        iterator = records
        iterator = tqdm(
            records,
            total=len(records),
            desc=f"Stat features [{split_name}:{column}]",
        )

        feature_rows = [extract_document_statistics(record, config=config) for record in iterator] # loop over rows
        feature_df = pd.DataFrame(feature_rows).add_prefix(f"{column}_") # make each feature as one column
        result = pd.concat([result.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    return result



def build_feature_summary(
    dict_df: dict[str, pd.DataFrame],
    config: Config = config,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary_columns = [
        "avg_sentence_length_words",
        "avg_function_words_per_sentence",
        "function_word_rate",
        "punctuation_rate",
        "avg_word_length",
    ]
    for split, df in dict_df.items():
        row: dict[str, Any] = {"split": split, "num_rows": len(df)}
        for column in ["text1", "text2"]:
            for feature_name in summary_columns:
                prefixed_name = f"{column}_{feature_name}"
                if prefixed_name in df.columns:
                    row[f"{prefixed_name}_mean"] = round(df[prefixed_name].mean(), 6)
        rows.append(row)

    return pd.DataFrame(rows)



def statistical_features_wrapper(
    dict_df: dict[str, pd.DataFrame],
    linguistic_cache: dict[str, dict[str, list[dict[str, Any]]]],
    config: Config = config,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:

    if config.verbose:
        print("======= STATISTICAL FEATURES START =======")

    statistical_dict_df: dict[str, pd.DataFrame] = {}

    for split, df in dict_df.items():

        if config.verbose:
            print(f"\nProcessing statistical features for split='{split}' ({len(df):,} rows)")

        statistical_dict_df[split] = extract_split_statistics(
            df,
            split_cache=linguistic_cache[split],
            split_name=split,
            config=config,
        )

    statistical_summary_df = build_feature_summary(statistical_dict_df, config=config)

    if config.verbose:
        print("\nStatistical feature summary:")
        print(statistical_summary_df)
        print("")
        print("======= STATISTICAL FEATURES END =======")
        print("")

    return statistical_dict_df, statistical_summary_df
