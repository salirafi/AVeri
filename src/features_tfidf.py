from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from tqdm.auto import tqdm

from function_words import FUNCTION_WORDS


FUNCTION_WORD_SET = {word.lower() for word in FUNCTION_WORDS}
DEFAULT_CONTENT_POS = ("NOUN", "PROPN", "VERB", "ADJ", "ADV")


@dataclass(slots=True)
class Config:
    verbose: bool = True
    allowed_pos_tags: tuple[str, ...] = DEFAULT_CONTENT_POS
    min_token_length: int = 2
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int | float = 2
    max_df: int | float = 0.95
    max_features: Optional[int] = 5000
    sublinear_tf: bool = True
    norm: str = "l2"
    dense_output: bool = True


config = Config()




# excluding all "invalid" tokens for TF-IDF
# these include punctuations, white space, masking placeholders, no lemmatization
# function words, excluded POS tags, too short tokens (< min_token_length), stop words
def _keep_token(
    token_text: str,
    token_lemma: str,
    token_pos: str,
    is_punct: bool,
    is_space: bool,
    config: Config = config,
) -> bool:
    
    if is_punct or is_space:
        return False
    if (token_text.startswith("<") and token_text.endswith(">")):
        return False
    if not token_lemma:
        return False

    normalized = token_lemma.lower()
    if len(normalized) < config.min_token_length:
        return False
    if not any(char.isalpha() for char in normalized):
        return False
    if token_pos not in config.allowed_pos_tags:
        return False
    if normalized in FUNCTION_WORD_SET:
        return False
    if normalized in ENGLISH_STOP_WORDS:
        return False

    return True


def record_to_tfidf_text(record: dict[str, Any], config: Config = config) -> list[str]:
    
    tokens: list[str] = []

    for token_text, token_lemma, token_pos, is_punct, is_space in zip(
        record["tokens"],
        record["token_lemma"],
        record["token_pos"],
        record["token_is_punct"],
        record["token_is_space"],
        strict=False,
    ):
        if not _keep_token(
            token_text=token_text,
            token_lemma=token_lemma,
            token_pos=token_pos,
            is_punct=is_punct,
            is_space=is_space,
            config=config,
        ):
            continue

        normalized = token_lemma.lower()
        tokens.append(normalized)

    return " ".join(tokens) # build a proper text (hopefully)

# 
def build_split_corpus(
    split_cache: dict[str, list[dict[str, Any]]],
    split_name: str = "",
    config: Config = config,
) -> dict[str, list[str]]:
    
    corpus_by_column: dict[str, list[str]] = {}
    for column in ["text1", "text2"]:
        records = split_cache[column]
        iterator = tqdm(
            records,
            total=len(records),
            desc=f"TF-IDF prep [{split_name}:{column}]",
        )

        corpus_by_column[column] = [record_to_tfidf_text(record, config=config) for record in iterator] # loop over rows

    return corpus_by_column


def fit_vectorizer(
    train_cache: dict[str, list[dict[str, Any]]],
    config: Config = config,
) -> tuple[TfidfVectorizer, dict[str, list[str]]]:
    
    if config.verbose:
        print("Building TF-IDF training corpus from train/text1 + train/text2...")

    train_corpus_by_column = build_split_corpus(train_cache, split_name="train", config=config)
    fit_corpus: list[str] = []
    for column in ["text1", "text2"]:
        fit_corpus.extend(train_corpus_by_column[column]) # word-level features

    vectorizer = TfidfVectorizer(
        analyzer="word",
        # token_pattern=r"(?u)\b\w+\b",
        # lowercase=False,
        preprocessor=None,
        tokenizer=str.split,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf, # TF scaling
        norm=config.norm,
    )
    vectorizer.fit(fit_corpus) # fitting TF-IDF

    if config.verbose:
        print(f"\nFitted TF-IDF vocabulary size: {len(vectorizer.get_feature_names_out()):,}")

    return vectorizer, train_corpus_by_column



# apply the fitted TF-IDF vectorizer to one split
def transform_split(
    df: pd.DataFrame,
    split_cache: dict[str, list[dict[str, Any]]],
    vectorizer: TfidfVectorizer,
    split_name: str = "",
    config: Config = config,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, float]]:
    
    result = df.copy().reset_index(drop=True)
    feature_names = vectorizer.get_feature_names_out().tolist()
    corpus_by_column = build_split_corpus(split_cache, split_name=split_name, config=config)
    density_stats: dict[str, float] = {}

    for column in ["text1", "text2"]:
        tfidf_matrix = vectorizer.transform(corpus_by_column[column]) # ONLY use the already trained vocabulary, not adding others using fit_transform
        density_stats[f"{column}_avg_nonzero_features"] = round(tfidf_matrix.getnnz(axis=1).mean() if tfidf_matrix.shape[0] else 0.0, 5)


        # converting vector to columns
        if config.dense_output:
            values = tfidf_matrix.toarray()
        else:
            values = tfidf_matrix
        columns = [f"{column}_tfidf_{index:05d}" for index, _ in enumerate(feature_names)]
        tfidf_df = pd.DataFrame(values, columns=columns)

        result = pd.concat([result, tfidf_df.reset_index(drop=True)], axis=1)

    return result, corpus_by_column, density_stats

def build_tfidf_summary(
    dict_df: dict[str, pd.DataFrame],
    density_stats_by_split: dict[str, dict[str, float]],
    vocabulary_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, df in dict_df.items():
        row: dict[str, Any] = {
            "split": split,
            "num_rows": len(df),
            "vocabulary_size": vocabulary_size,
        }
        row.update(density_stats_by_split.get(split, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def tfidf_features_wrapper(
    dict_df: dict[str, pd.DataFrame],
    linguistic_cache: dict[str, dict[str, list[dict[str, Any]]]],
    config: Config = config,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:


    if config.verbose:
        print("======= TF-IDF FEATURES START =======")
        print("")

    vectorizer, train_corpus_by_column = fit_vectorizer(linguistic_cache["train"], config=config)

    tfidf_dict_df: dict[str, pd.DataFrame] = {}
    corpus_by_split: dict[str, dict[str, list[str]]] = {}
    density_stats_by_split: dict[str, dict[str, float]] = {}

    for split, df in dict_df.items():

        if config.verbose:
            print(f"\nTransforming split='{split}' ({len(df):,} rows)")

        transformed_df, split_corpus, density_stats = transform_split(
            df,
            split_cache=linguistic_cache[split],
            vectorizer=vectorizer,
            split_name=split,
            config=config,
        )

        tfidf_dict_df[split] = transformed_df
        corpus_by_split[split] = split_corpus
        density_stats_by_split[split] = density_stats
    # 
    tfidf_summary_df = build_tfidf_summary(
        tfidf_dict_df,
        density_stats_by_split=density_stats_by_split,
        vocabulary_size=len(vectorizer.get_feature_names_out()),
    )

    tfidf_artifacts = {
        "vectorizer": vectorizer,
        "feature_names": vectorizer.get_feature_names_out().tolist(),
        "train_corpus_by_column": train_corpus_by_column,
        "corpus_by_split": corpus_by_split,
    }

    if config.verbose:
        print("\nTF-IDF summary:")
        print(tfidf_summary_df)
        print("")
        print("======= TF-IDF FEATURES END =======")
        print("")

    return tfidf_dict_df, tfidf_summary_df, tfidf_artifacts
