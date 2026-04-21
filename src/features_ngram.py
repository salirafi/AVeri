from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import textstat
from tqdm.auto import tqdm


@dataclass(slots=True)
class Config:
    verbose: bool = True
    char_ngram_n: int = 4
    char_tfidf_min_df: int | float = 2
    char_tfidf_max_df: int | float = 0.95
    char_tfidf_max_features: Optional[int] = 100000
    pos_ngram_range: tuple[int, int] = (3, 3)
    pos_tfidf_min_df: int | float = 2
    pos_tfidf_max_df: int | float = 0.95
    pos_tfidf_max_features: Optional[int] = 50000
    sublinear_tf: bool = True
    norm: str = "l2"
    include_readability: bool = True
    dense_output: bool = True


config = Config()


# ================== CAHARCTER 4-GRAM =======================

# bsaed on definition in https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.22954
def build_space_free_char_ngrams(text: str, n: int = 4) -> list[str]:
    spans = text.split()
    grams: list[str] = []
    for span in spans:
        if not span:
            continue
        if len(span) < n:
            grams.append(span)
            continue
        for index in range(len(span) - n + 1):
            grams.append(span[index:index + n])
    return grams

def build_char_corpus(
    dict_df: dict[str, pd.DataFrame],
    split_name: str,
    config: Config = config,
) -> dict[str, list[str]]:
    
    corpus_by_column: dict[str, list[str]] = {}
    split_df = dict_df[split_name]

    for column in ["text1", "text2"]:
        texts = split_df[column].tolist()
        iterator = tqdm(texts, total=len(texts), desc=f"Char4 prep [{split_name}:{column}]")
        corpus_by_column[column] = [
            " ".join(build_space_free_char_ngrams(text, n=config.char_ngram_n))
            for text in iterator
        ]

    return corpus_by_column

def fit_char_vectorizer(
    dict_df: dict[str, pd.DataFrame],
    config: Config = config,
) -> tuple[TfidfVectorizer, dict[str, list[str]]]:
    
    if config.verbose:
        print("\nBuilding character 4-gram training corpus from train/text1 + train/text2...")

    train_corpus_by_column = build_char_corpus(dict_df, split_name="train", config=config)
    fit_corpus: list[str] = []
    for column in ["text1", "text2"]:
        fit_corpus.extend(train_corpus_by_column[column]) # character-level features

    vectorizer = TfidfVectorizer(
        analyzer="word",
        # token_pattern=r"(?u)\b\S+\b",
        lowercase=False,
        preprocessor=None,
        tokenizer=str.split,
        ngram_range=(1, 1),
        min_df=config.char_tfidf_min_df,
        max_df=config.char_tfidf_max_df,
        max_features=config.char_tfidf_max_features,
        sublinear_tf=config.sublinear_tf,
        norm=config.norm,
    )
    vectorizer.fit(fit_corpus)

    if config.verbose:
        print(f"\nFitted character 4-gram vocabulary size: {len(vectorizer.get_feature_names_out()):,}")

    return vectorizer, train_corpus_by_column


# =================== POS n-GRAM =========================


def record_to_pos_sequence(record: dict[str, Any]) -> list[str]:
    return [
        token_pos
        for token_pos, is_punct, is_space in zip(
            record["token_pos"],
            record["token_is_punct"],
            record["token_is_space"],
            strict=False,
        ) if not is_punct and not is_space]

def build_pos_corpus(
    split_cache: dict[str, list[dict[str, Any]]],
    split_name: str,
    config: Config = config,
) -> dict[str, list[str]]:
    
    corpus_by_column: dict[str, list[str]] = {}

    for column in ["text1", "text2"]:
        records = split_cache[column]
        iterator = tqdm(records, total=len(records), desc=f"POS prep [{split_name}:{column}]")
        corpus_by_column[column] = [
            " ".join(record_to_pos_sequence(record))
            for record in iterator
        ]

    return corpus_by_column

def fit_pos_vectorizer(
    linguistic_cache: dict[str, dict[str, list[dict[str, Any]]]],
    config: Config = config,
) -> tuple[TfidfVectorizer, dict[str, list[str]]]:
    
    if config.verbose:
        print("\nBuilding POS n-gram training corpus from train/text1 + train/text2...")

    train_corpus_by_column = build_pos_corpus(linguistic_cache["train"], split_name="train", config=config)
    fit_corpus: list[str] = []
    for column in ["text1", "text2"]:
        fit_corpus.extend(train_corpus_by_column[column])

    vectorizer = TfidfVectorizer(
        analyzer="word",
        # token_pattern=r"(?u)\b\S+\b",
        lowercase=False,
        preprocessor=None,
        tokenizer=str.split,
        ngram_range=config.pos_ngram_range,
        min_df=config.pos_tfidf_min_df,
        max_df=config.pos_tfidf_max_df,
        max_features=config.pos_tfidf_max_features,
        sublinear_tf=config.sublinear_tf,
        norm=config.norm,
    )
    vectorizer.fit(fit_corpus)

    if config.verbose:
        print(f"\nFitted POS {config.pos_ngram_range}-gram vocabulary size: {len(vectorizer.get_feature_names_out()):,}")

    return vectorizer, train_corpus_by_column


# =================== MAIN =========================


def _matrix_to_feature_df(
    matrix: Any,
    column_prefix: str,
    feature_prefix: str,
    config: Config = config,
) -> pd.DataFrame:
    values = matrix.toarray() if config.dense_output else matrix
    columns = [
        f"{column_prefix}_{feature_prefix}_{index:05d}"
        for index in range(matrix.shape[1])
    ]
    return pd.DataFrame(values, columns=columns)


def build_readability_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for _, row in df.iterrows():
        feature_row: dict[str, float] = {}
        for column in ["text1", "text2"]:
            text = row[column]
            feature_row[f"{column}_readability_flesch_kincaid_grade"] = round(textstat.flesch_kincaid_grade(text), 5)
            feature_row[f"{column}_readability_gunning_fog"] = round(textstat.gunning_fog(text), 5)
            feature_row[f"{column}_readability_smog"] = round(textstat.smog_index(text), 5)
            feature_row[f"{column}_readability_coleman_liau"] = round(textstat.coleman_liau_index(text), 5)
        rows.append(feature_row)

    return pd.DataFrame(rows)


def transform_split(
    base_df: pd.DataFrame,
    dict_df: dict[str, pd.DataFrame],
    linguistic_cache: dict[str, dict[str, list[dict[str, Any]]]],
    split_name: str,
    char_vectorizer: TfidfVectorizer,
    pos_vectorizer: TfidfVectorizer,
    config: Config = config,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]], dict[str, float]]:
    
    result = base_df.copy().reset_index(drop=True)
    char_corpus_by_column = build_char_corpus(dict_df, split_name=split_name, config=config)
    pos_corpus_by_column = build_pos_corpus(linguistic_cache[split_name], split_name=split_name, config=config)
    density_stats: dict[str, float] = {}

    for column in ["text1", "text2"]:
        char_matrix = char_vectorizer.transform(char_corpus_by_column[column])
        density_stats[f"{column}_char_avg_nonzero_features"] = round(
            char_matrix.getnnz(axis=1).mean() if char_matrix.shape[0] else 0.0,
            6,
        )
        char_df = _matrix_to_feature_df(
            char_matrix,
            column_prefix=column,
            feature_prefix=f"char{config.char_ngram_n}_tfidf",
            config=config,
        )
        result = pd.concat([result, char_df.reset_index(drop=True)], axis=1)

        pos_matrix = pos_vectorizer.transform(pos_corpus_by_column[column])
        density_stats[f"{column}_pos_avg_nonzero_features"] = round(
            pos_matrix.getnnz(axis=1).mean() if pos_matrix.shape[0] else 0.0,
            6,
        )
        pos_df = _matrix_to_feature_df(
            pos_matrix,
            column_prefix=column,
            feature_prefix=f"pos{config.pos_ngram_range}_tfidf",
            config=config,
        )
        result = pd.concat([result, pos_df.reset_index(drop=True)], axis=1)

    if config.include_readability:
        readability_df = build_readability_df(dict_df[split_name])
        result = pd.concat([result, readability_df.reset_index(drop=True)], axis=1)

    return result, char_corpus_by_column, pos_corpus_by_column, density_stats


def build_summary(
    dict_df: dict[str, pd.DataFrame],
    density_stats_by_split: dict[str, dict[str, float]],
    char_vocabulary_size: int,
    pos_vocabulary_size: int,
) -> pd.DataFrame:
    
    rows: list[dict[str, Any]] = []

    for split, df in dict_df.items():
        row: dict[str, Any] = {
            "split": split,
            "num_rows": len(df),
            "char_vocabulary_size": char_vocabulary_size,
            "pos_vocabulary_size": pos_vocabulary_size,
        }
        row.update(density_stats_by_split.get(split, {}))
        rows.append(row)

    return pd.DataFrame(rows)


def ngram_features_wrapper(
    dict_df: dict[str, pd.DataFrame],
    linguistic_cache: dict[str, dict[str, list[dict[str, Any]]]],
    config: Config = config,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:

    if config.verbose:
        print("======= N-GRAM FEATURES START =======")

    char_vectorizer, train_char_corpus = fit_char_vectorizer(dict_df, config=config)
    pos_vectorizer, train_pos_corpus = fit_pos_vectorizer(linguistic_cache, config=config)

    ngram_dict_df: dict[str, pd.DataFrame] = {}
    char_corpus_by_split: dict[str, dict[str, list[str]]] = {}
    pos_corpus_by_split: dict[str, dict[str, list[str]]] = {}
    density_stats_by_split: dict[str, dict[str, float]] = {}

    for split, df in dict_df.items():

        if config.verbose:
            print(f"\nTransforming split='{split}' ({len(df):,} rows)")

        transformed_df, split_char_corpus, split_pos_corpus, density_stats = transform_split(
            base_df=df,
            dict_df=dict_df,
            linguistic_cache=linguistic_cache,
            split_name=split,
            char_vectorizer=char_vectorizer,
            pos_vectorizer=pos_vectorizer,
            config=config,
        )
        ngram_dict_df[split] = transformed_df
        char_corpus_by_split[split] = split_char_corpus
        pos_corpus_by_split[split] = split_pos_corpus
        density_stats_by_split[split] = density_stats

    ngram_summary_df = build_summary(
        ngram_dict_df,
        density_stats_by_split=density_stats_by_split,
        char_vocabulary_size=len(char_vectorizer.get_feature_names_out()),
        pos_vocabulary_size=len(pos_vectorizer.get_feature_names_out()),
    )

    ngram_artifacts = {
        "char_vectorizer": char_vectorizer,
        "char_feature_names": char_vectorizer.get_feature_names_out().tolist(),
        "train_char_corpus": train_char_corpus,
        "char_corpus_by_split": char_corpus_by_split,
        "pos_vectorizer": pos_vectorizer,
        "pos_feature_names": pos_vectorizer.get_feature_names_out().tolist(),
        "train_pos_corpus": train_pos_corpus,
        "pos_corpus_by_split": pos_corpus_by_split,
        "config": config,
    }

    if config.verbose:
        print("\nN-gram summary:")
        print(ngram_summary_df)
        print("")
        print("======= N-GRAM FEATURES END =======")
        print("")

    return ngram_dict_df, ngram_summary_df, ngram_artifacts
