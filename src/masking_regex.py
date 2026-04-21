from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
import pandas as pd

# all current formatting choices can be expanded or replaced or removed based on necessities
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
DATE_RE = re.compile(
    r"(?ix)\b(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?|"
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?(?:\s+\d{4})?"
    r")\b")
TIME_RE = re.compile(r"(?i)\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\b")
CURRENCY_RE = re.compile(
    r"(?ix)\b(?:"
    r"[$€£¥]\s?\d[\d,]*(?:\.\d+)?|"
    r"\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|gbp|jpy|idr|sgd|aud|cad|dollars?|euros?|yen|rupiah)"
    r")\b")
ADDRESS_RE = re.compile(
    r"(?ix)\b\d{1,6}\s+"
    r"(?:[a-z0-9.\-]+\s+){0,5}"
    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|way|parkway|pkwy)\b\.?")
ORG_SUFFIX_RE = re.compile(r"(?i)\b(?:inc|inc\.|llc|l\.l\.c\.|ltd|ltd\.|corp|corp\.|co|co\.|company|plc|gmbh|s\.a\.|ag)\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")


# mapping for the masked entities (replaced by the chosen placeholder)
PLACEHOLDER_MAP: dict[str, str] = {
    "url": "<URL>",
    "email": "<EMAIL>",
    "date": "<DATE>",
    "time": "<TIME>",
    "currency": "<CURRENCY>",
    "address": "<ADDRESS>",
    "org_suffix": "<ORG_SUFFIX>",
    "number": "<NUMBER>",
}

@dataclass(slots=True)
class Config:
    verbose: bool = True
config = Config()




def _replace_pattern(text: str, pattern: re.Pattern[str], placeholder: str) -> tuple[str, int]:
    matches = pattern.findall(text)
    if not matches:
        return text, 0
    return pattern.sub(placeholder, text), len(matches)

# masking/replacing chosen regex entities
# can be expanded, but make sure also put in the re.compile constant and default_factory
def _mask_regex_entities(text: Any) -> tuple[str, dict[str, int]]:
    value = "" if pd.isna(text) else str(text)
    counts = {key: 0 for key in PLACEHOLDER_MAP} # for verbose=True and summary

    value, counts["url"] = _replace_pattern(value, URL_RE, PLACEHOLDER_MAP["url"])
    value, counts["email"] = _replace_pattern(value, EMAIL_RE, PLACEHOLDER_MAP["email"])
    value, counts["date"] = _replace_pattern(value, DATE_RE, PLACEHOLDER_MAP["date"])
    value, counts["time"] = _replace_pattern(value, TIME_RE, PLACEHOLDER_MAP["time"])
    value, counts["currency"] = _replace_pattern(value, CURRENCY_RE, PLACEHOLDER_MAP["currency"])
    value, counts["address"] = _replace_pattern(value, ADDRESS_RE, PLACEHOLDER_MAP["address"])
    value, counts["org_suffix"] = _replace_pattern(value, ORG_SUFFIX_RE, PLACEHOLDER_MAP["org_suffix"])
    value, counts["number"] = _replace_pattern(value, NUMBER_RE, PLACEHOLDER_MAP["number"])

    return value, counts




def mask_split(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Mask regex entities in one split DataFrame.
    """
    masked_df = df.copy()
    split_counts = {key: 0 for key in PLACEHOLDER_MAP}

    for column in ["text1", "text2"]:

        masked_texts: list[str] = []
        for value in masked_df[column].tolist():
            masked_text, counts = _mask_regex_entities(value)
            masked_texts.append(masked_text)

             # for verbose=True and summary
            for key, count in counts.items():
                split_counts[key] += count

        masked_df[column] = masked_texts

    return masked_df, split_counts

def mask_splits(
    dict_df: dict[str, pd.DataFrame],
    config: Config = config,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Mask every split using regex masking.
    """
    masked_dict_df: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for split, df in dict_df.items():
        if config.verbose:
            print(f"\nRegex masking split='{split}' ({len(df):,} rows)")

        masked_df, split_counts = mask_split(df)
        masked_dict_df[split] = masked_df

        summary_row: dict[str, Any] = {"split": split, "num_rows": len(masked_df)}
        summary_row.update({f"{key}_count": int(value) for key, value in split_counts.items()})
        summary_rows.append(summary_row)

        if config.verbose:
            print(
                "    Replacement counts: "
                f"URL={split_counts['url']:,}, EMAIL={split_counts['email']:,}, "
                f"DATE={split_counts['date']:,}, TIME={split_counts['time']:,}, "
                f"CURRENCY={split_counts['currency']:,}, ADDRESS={split_counts['address']:,}, "
                f"ORG_SUFFIX={split_counts['org_suffix']:,}, NUMBER={split_counts['number']:,}"
            )

    summary_df = pd.DataFrame(summary_rows)
    return masked_dict_df, summary_df


def masking_wrapper(
    dict_df: dict[str, pd.DataFrame],
    config: Config = config,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if config.verbose:
        print("======= REGEX MASKING START =======")

    masked_dict_df, masking_summary_df = mask_splits(dict_df, config=config)

    if config.verbose:
        print("\nRegex masking summary:")
        print(masking_summary_df)
        print("")
        print("======= REGEX MASKING END =======")
        print("")

    return masked_dict_df, masking_summary_df
