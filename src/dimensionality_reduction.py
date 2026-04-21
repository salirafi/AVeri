from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


@dataclass(slots=True)
class Config:
    verbose: bool = True
    text_columns: tuple[str, ...] = ("text1", "text2")
    keep_original_columns: bool = False
    reduce_tfidf: bool = True
    reduce_char_ngrams: bool = True
    reduce_pos_ngrams: bool = True
    tfidf_components: int = 300
    char_components: int = 300
    pos_components: int = 100
    random_state: int = 42

config = Config()


# column suffixes for each features family/category
FAMILY_SPECS = {
    "tfidf": {
        "suffix_prefix": "tfidf_",
        "config_flag": "reduce_tfidf",
        "components_attr": "tfidf_components",
        "reduced_prefix": "tfidf_svd",
    },
    "char_ngrams": {
        "suffix_prefix": "char",
        "must_contain": "_tfidf_",
        "config_flag": "reduce_char_ngrams",
        "components_attr": "char_components",
        "reduced_prefix": "char_tfidf_svd",
    },
    "pos_ngrams": {
        "suffix_prefix": "pos",
        "must_contain": "_tfidf_",
        "config_flag": "reduce_pos_ngrams",
        "components_attr": "pos_components",
        "reduced_prefix": "pos_tfidf_svd",
    }}



def _match_family_suffix(suffix: str, family_name: str) -> bool:
    spec = FAMILY_SPECS[family_name]
    if family_name == "tfidf":
        return suffix.startswith(spec["suffix_prefix"])
    return suffix.startswith(spec["suffix_prefix"]) and spec["must_contain"] in suffix

# get features column suffixes
def discover_family_suffixes(df: pd.DataFrame, family_name: str) -> list[str]:
    suffixes: list[str] = []
    for column in df.columns:
        if not column.startswith("text1_"): # excluding original text columns
            continue
        suffix = column[len("text1_"):]
        if _match_family_suffix(suffix, family_name):
            suffixes.append(suffix)
    return suffixes


# ========================= Fit SVD ====================================

def _shared_train_matrix(train_df: pd.DataFrame, suffixes: list[str]) -> sparse.csr_matrix:
    text1_columns = [f"text1_{suffix}" for suffix in suffixes]
    text2_columns = [f"text2_{suffix}" for suffix in suffixes]
    train_text1 = sparse.csr_matrix(train_df[text1_columns].to_numpy(dtype=np.float32, copy=False))
    train_text2 = sparse.csr_matrix(train_df[text2_columns].to_numpy(dtype=np.float32, copy=False))
    return sparse.vstack([train_text1, train_text2], format="csr")

# truncatedSVD cannot use more components than the data matrix can support
# the safe upper bound is limited by the smaller matrix dimension.
def _effective_components(requested_components: int, train_matrix: sparse.csr_matrix) -> int:
    max_components = min(train_matrix.shape[0] - 1, train_matrix.shape[1] - 1)
    if max_components < 1: # guard against tiny matrices where upper bound would be 0
        return 1
    return min(requested_components, max_components)

def fit_family_svd(
    train_df: pd.DataFrame,
    family_name: str,
    suffixes: list[str],
    config: Config = config,
) -> tuple[TruncatedSVD, int]:
    train_matrix = _shared_train_matrix(train_df, suffixes) # building sparse matrix for fit
    requested_components = getattr(config, FAMILY_SPECS[family_name]["components_attr"]) # getting corresponding target components from config
    n_components = _effective_components(requested_components, train_matrix)

    svd = TruncatedSVD(n_components=n_components, random_state=config.random_state)
    svd.fit(train_matrix)
    return svd, n_components

# =================================================================


def transform_family_split(
    df: pd.DataFrame,
    family_name: str,
    suffixes: list[str],
    svd: TruncatedSVD,
    config: Config = config,
) -> pd.DataFrame:
    
    result = df.copy().reset_index(drop=True)
    text1_columns = [f"text1_{suffix}" for suffix in suffixes]
    text2_columns = [f"text2_{suffix}" for suffix in suffixes]

    text1_matrix = sparse.csr_matrix(result[text1_columns].to_numpy(dtype=np.float32, copy=False))
    text2_matrix = sparse.csr_matrix(result[text2_columns].to_numpy(dtype=np.float32, copy=False))

    # project the original high-dimensional features into the reduced latent space
    text1_reduced = svd.transform(text1_matrix)
    text2_reduced = svd.transform(text2_matrix)

    reduced_prefix = FAMILY_SPECS[family_name]["reduced_prefix"]
    text1_reduced_df = pd.DataFrame(text1_reduced,
        columns=[f"text1_{reduced_prefix}_{index:04d}" for index in range(text1_reduced.shape[1])])
    text2_reduced_df = pd.DataFrame(text2_reduced,
        columns=[f"text2_{reduced_prefix}_{index:04d}" for index in range(text2_reduced.shape[1])])

    if not config.keep_original_columns:
        result = result.drop(columns=text1_columns + text2_columns)

    result = pd.concat([
            result.reset_index(drop=True),
            text1_reduced_df.reset_index(drop=True),
            text2_reduced_df.reset_index(drop=True),
        ], axis=1)
    
    return result



def dimensionality_reduction_wrapper(
    dict_df: dict[str, pd.DataFrame],
    config: Config = config,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:


    if config.verbose:
        print("======= DIMENSIONALITY REDUCTION START =======")

    reduced_dict_df = {split: df.copy().reset_index(drop=True) for split, df in dict_df.items()}
    summary_rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {"svd_models": {}, "family_suffixes": {}}

    for family_name, spec in FAMILY_SPECS.items():

        if not getattr(config, spec["config_flag"]):
            continue

        suffixes = discover_family_suffixes(reduced_dict_df["train"], family_name)
        if config.verbose:
            print(f"\nFitting TruncatedSVD for family='{family_name}' with {len(suffixes):,} base features")

        svd, n_components = fit_family_svd(reduced_dict_df["train"], family_name=family_name, suffixes=suffixes, config=config)

        artifacts["svd_models"][family_name] = svd
        artifacts["family_suffixes"][family_name] = suffixes

        for split, df in list(reduced_dict_df.items()):

            if config.verbose:
                print(f"    Transforming split='{split}' for family='{family_name}'")
            reduced_dict_df[split] = transform_family_split(
                df,
                family_name=family_name,
                suffixes=suffixes,
                svd=svd,
                config=config,
            )

        explained_variance = float(svd.explained_variance_ratio_.sum())
        summary_rows.append({
                "family": family_name,
                "original_base_features": len(suffixes),
                "reduced_components": n_components,
                "explained_variance_ratio_sum": round(explained_variance, 6),
            })

        if config.verbose:
            print(
                f"    Reduced family='{family_name}' to {n_components} components "
                f"(explained variance sum={explained_variance:.4f})"
            )

    reduction_summary_df = pd.DataFrame(summary_rows)

    if config.verbose:
        print("\nDimensionality reduction summary:")
        print(reduction_summary_df)
        print("")
        print("======= DIMENSIONALITY REDUCTION END =======")
        print("")

    return reduced_dict_df, reduction_summary_df, artifacts
