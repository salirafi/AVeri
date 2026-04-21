from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
from datasets import load_from_disk


@dataclass(slots=True)
class Config():
    verbose: bool = True
    max_words: Optional[int] = None
    min_words: Optional[int] = None

config = Config()


# ================ HELPERS ================



def mask_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with invalid text fields (non-string, empty) or invalid labels (not 0 or 1).
    No need to check NaNs because the dataset does not contain NaNs.
    """
    text1_ok = df["text1"].map(lambda x: isinstance(x, str) and x.strip() != "")
    text2_ok = df["text2"].map(lambda x: isinstance(x, str) and x.strip() != "")
    label_ok = df["same"].isin([0, 1])
    valid_mask = text1_ok & text2_ok & label_ok
    return df[valid_mask] # return the valid rows



def _within_word_count_range(value, min_words: Optional[int] = None, max_words: Optional[int] = None) -> bool:
    count = len(value.split()) # count the number of words by splitting on whitespace (rough estimate)
    if min_words is not None and count < min_words: return False
    if max_words is not None and count > max_words: return False
    return True # assume no invalid rows from previous check (mask_invalid_rows)

def mask_by_word_count(df: pd.DataFrame,
                        min_words: Optional[int] = None,
                        max_words: Optional[int] = None
                        ) -> pd.DataFrame:
    """
    """
    if min_words is None and max_words is None: return df # no filtering needed

    text1_ok = df["text1"].map(lambda x: _within_word_count_range(x, min_words, max_words))
    text2_ok = df["text2"].map(lambda x: _within_word_count_range(x, min_words, max_words))
    valid_mask = text1_ok & text2_ok
    return df[valid_mask] # return the valid rows



def _symmetric_pair_id(text1: str, text2: str, same: int) -> list[str | int]:
    """
    Generate a unique identifier for each pair of texts, such that the order of the texts does not matter.
    For example, if text1="A", text2="B", same=1, then the symmetric pair ID should be the same as text1="B", text2="A", same=1.
    """
    text1, text2 = sorted([text1, text2]) # ensure the order of texts does not matter by sorting them alphabetically
    return text1, text2, same # combine the sorted texts and label into a tuple to create a unique identifier for the pair

# def _duplicate_group_keep_row(df: pd.DataFrame, split_priority: dict[str, int]) -> pd.Series:
#     ranked = df.assign(__split_priority__=df["__split__"].map(split_priority)).sort_values(by=["__split_priority__", "__row_id__"], kind="stable") # stable sort to maintain original order within each split
#     return ranked.iloc[0] # keep the first row (highest priority split, then lowest row_id) and drop the rest as duplicates


def _length_stats(series: pd.Series) -> dict[str, float]:

    word_length = series.str.split().str.len()
    char_length = series.str.len()
    return {
        "median_word_length": round(word_length.median(), 2),
        "mean_word_length": round(word_length.mean(), 2),
        "std_word_length": round(word_length.std(), 2),
        "median_char_length": round(char_length.median(), 2),
        "mean_char_length": round(char_length.mean(), 2),
        "std_char_length": round(char_length.std(), 2),
    }


# ================ MAIN FUNCTIONS ================


def load_all_splits(path: Path,
                    config: Config = config
                    ) -> pd.DataFrame:
    """
    Loads all three splits (train, validation, test) from the given path and concatenates them into a single DataFrame.
    Assumes that the splits are stored in .arrow files under the corresponding split folder.
    """
    dict_df: dict[str, pd.DataFrame] = {}
    for split in ["train", "validation", "test"]:
        dataset_dir = path / f"authorship_verification_{split}"
        ds = load_from_disk(dataset_dir)
        df = ds.to_pandas()
        df["__split__"] = split # identify which split
        # df["__row_id__"] = df.index
        dict_df[split] = df

        if config.verbose:
            print(f"Loaded {split} split: {len(df)} rows")

    return dict_df

def mask_rows(dict_df: dict[str, pd.DataFrame],
            min_words: Optional[int] = None,
            max_words: Optional[int] = None,
            config: Config = config
            ) -> pd.DataFrame:
    
    """
    Mask rows with invalid text fields, invalid labels, and optionally filter by word count. 
    Then identify and drop duplicate pairs across all splits, keeping only the first occurrence based on split priority (train > validation > test) and row_id.
    """
    if config.verbose:
        if min_words is not None or max_words is not None: 
            print(f"\nWord-count filter used: min_words={min_words}, max_words={max_words}")
        else: print("\nNo word-count filter used\n")

        rows_before_masking: dict[str, int] = {}
        for split, df in dict_df.items():
            rows_before_masking[split] = len(dict_df[split])
        print("\nStarting masking rows based on invalid text fields and labels, and word count if specified...\n")

    df_valid: list[pd.DataFrame] = [] # placeholder for valid rows from all splits
    for split, df in dict_df.items():

        if config.verbose:
            print(f"    Processing split='{split}' ({len(df):,} rows)")

        df = mask_invalid_rows(df)
        df = mask_by_word_count(df, min_words, max_words)
        df_valid.append(df) # update the filtered df back into the dict

        if config.verbose:
            print(f"    Split='{split}': remove {rows_before_masking[split] - len(df):,} rows, {len(df):,} rows remain\n")

    df_valid = pd.concat(df_valid, ignore_index=True) # combine valid rows from all splits into one DataFrame
    df_valid["__symmetric_pair_id__"] = df_valid.apply(lambda row: _symmetric_pair_id(row["text1"], row["text2"], row["same"]), axis=1) # identify symmetric pairs
    if config.verbose:
        print("Checking for duplicate pairs (same texts and label, regardless of order) before deduplication...")
        num_duplicates = df_valid.duplicated(subset="__symmetric_pair_id__").sum()
        print(f"There are {num_duplicates:,} duplicate pairs across all splits before deduplication\n")

    
    # drop duplicate pairs, keeping only the first occurrence
    # since df_valid is sorted by ["train", "validation", "test"] in load_all_splits, keep="first" will prioritize keeping the row from the train split, then validation, then test
    df_valid = df_valid.drop_duplicates(subset="__symmetric_pair_id__", keep="first", ignore_index=True)

    # for _, group in df_valid.groupby("__symmetric_pair_id__", sort=False):
    #     if len(group) < 2: continue # only interested in duplicate pairs
        
    #     split_priority = {"train": 3, "validation": 2, "test": 1} # define split priority for keeping rows
    #     keep_row = _duplicate_group_keep_row(group, split_priority) # get the row to keep based on split priority and row_id
    #     drop_rows = group.index.difference([keep_row.name]) # identify rows to drop (all except the keep_row)
    #     df_valid = df_valid.drop(index=drop_rows) # drop the duplicate rows, keeping only the one with the highest priority

    for split, df in dict_df.items():
        if config.verbose:
            print(f"    Split='{split}': {len(df_valid[df_valid['__split__'] == split]):,} valid rows after deduplication")

        dict_df[split] = df_valid[df_valid["__split__"] == split].drop(columns=["__split__", "__symmetric_pair_id__"]) # update the dict with the valid rows for each split, dropping the helper column

    return dict_df


def summary_stats(dict_df: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Generate summary statistics for each split.
    """
    summary = []
    for split, df in dict_df.items():
        stats_text1 = _length_stats(df["text1"])
        stats_text2 = _length_stats(df["text2"])
        same_distribution = df["same"].value_counts().to_dict()
        summary.append({
            "split": split,
            "num_rows": len(df),
            "same_0_count": int(same_distribution.get(0, 0)),
            "same_1_count": int(same_distribution.get(1, 0)),
            "same_0_ratio": round(df["same"].eq(0).mean(), 4),
            "same_1_ratio": round(df["same"].eq(1).mean(), 4),
            **{f"text1_{k}": v for k, v in stats_text1.items()},
            **{f"text2_{k}": v for k, v in stats_text2.items()},
        })

    summary_df = pd.DataFrame(summary)
    print("\nSummary statistics for each split:")
    print(summary_df)

    return summary_df

def audit_wrapper(path: Path, # folder path to all three splits
                  config: Config = config
                  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Wrapper around the audit structure.
    """

    if config.verbose:
        print("======= AUDIT START =======")
        print("")
    

    dict_df = load_all_splits(path)

    dict_df = mask_rows(dict_df, config.min_words, config.max_words, config)
    summary_df = summary_stats(dict_df)

    if config.verbose:
        print("")
        print("======= AUDIT END =======")
        print("")

    return dict_df, summary_df
