from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from ftfy import fix_text


@dataclass(slots=True)
class Config:
    verbose: bool = True
    unicode_form: str = "NFC"
config = Config()


CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]") # filter out non-printable control characters
INLINE_SPACE_RE = re.compile(r"[^\S\r\n]+") # collapse sequences inline whitespace into a single regular space
SPACES_AROUND_NEWLINE_RE = re.compile(r"[ \t]*\n[ \t]*") # match newline characters
THREE_PLUS_NEWLINES_RE = re.compile(r"\n{3,}") # match sequences of >=3 consecutive newline characters; preserving paragraph spacing to at most 3 newlines
QUOTE_DASH_TRANSLATION = str.maketrans({ # normalize similar unicode characters
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    })


# ======= DEALING WITH MATH MODE ============

# match inline/display LaTeX math spans
MATH_SPAN_RE = re.compile(
    r"(?<!\\)\$\$(.+?)(?<!\\)\$\$" # $$...$$
    r"|(?<!\\)\$(.+?)(?<!\\)\$" # $...$
    r"|\\\((.+?)\\\)" # \( ... \)
    r"|\\\[(.+?)\\\]", # \[ ... \]
    flags=re.DOTALL,
)

# greek symbol map, can be expanded
LATEX_SYMBOL_TO_TEXT = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\varepsilon": "epsilon",
    r"\zeta": "zeta",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\vartheta": "theta",
    r"\iota": "iota",
    r"\kappa": "kappa",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\nu": "nu",
    r"\xi": "xi",
    r"\pi": "pi",
    r"\varpi": "pi",
    r"\rho": "rho",
    r"\varrho": "rho",
    r"\sigma": "sigma",
    r"\varsigma": "sigma",
    r"\tau": "tau",
    r"\upsilon": "upsilon",
    r"\phi": "phi",
    r"\varphi": "phi",
    r"\chi": "chi",
    r"\psi": "psi",
    r"\omega": "omega",
    r"\Gamma": "gamma",
    r"\Delta": "delta",
    r"\Theta": "theta",
    r"\Lambda": "lambda",
    r"\Xi": "xi",
    r"\Pi": "pi",
    r"\Sigma": "sigma",
    r"\Phi": "phi",
    r"\Psi": "psi",
    r"\Omega": "omega",
}

# if math body is exactly one known symbol/constant, convert to text
# otherwise replace the whole math span with <MATH>
SIMPLE_SYMBOL_RE = re.compile(r"^\s*(\\[A-Za-z]+)\s*$")

def normalize_math_spans(value: str, math_placeholder: str = "<MATH>") -> str:
    """Convert simple math constants like '$\\alpha$' -> 'alpha'
    and replace more complex equations like '$x^2 + y^2 = z^2$' -> '<MATH>'
    """
    def _replace(match: re.Match[str]) -> str:
        # exactly one of these groups will be non-None
        math_body = next(group for group in match.groups() if group is not None)
        math_body = math_body.strip()

        simple = SIMPLE_SYMBOL_RE.fullmatch(math_body)
        if simple:
            symbol = simple.group(1)
            if symbol in LATEX_SYMBOL_TO_TEXT:
                return LATEX_SYMBOL_TO_TEXT[symbol]

        return math_placeholder

    return MATH_SPAN_RE.sub(_replace, value)

# ===============================



def _coerce_text(value: Any) -> str:
    if pd.isna(value): # the original dataset should not contain any NaNs or None
        return ""
    if isinstance(value, str):
        return value
    return str(value)

def normalize_text(text: Any, config: Config = config) -> str:
    """
    Basic normalization for one text string.
    """
    value = _coerce_text(text) # make sure text is str
    value = html.unescape(value)
    value = fix_text(value) # fix broken unicode (repair mojibake)
    value = unicodedata.normalize(config.unicode_form, value)
    value = value.replace("\r\n", "\n").replace("\r", "\n") # standardize line endings
    value = CONTROL_RE.sub("", value)
    value = value.translate(QUOTE_DASH_TRANSLATION)
    value = normalize_math_spans(value)
    value = INLINE_SPACE_RE.sub(" ", value)
    value = SPACES_AROUND_NEWLINE_RE.sub("\n", value)
    value = THREE_PLUS_NEWLINES_RE.sub("\n\n\n", value)
    value = value.strip()

    return value



def normalize_splits(
    dict_df: dict[str, pd.DataFrame],
    config: Config = config,
) -> dict[str, pd.DataFrame]:
    """
    Normalize every split produced by audit_wrapper.
    """
    normalized_dict: dict[str, pd.DataFrame] = {}

    for split, df in dict_df.items():

        if config.verbose:
            print(f"Normalizing split='{split}' ({len(df):,} rows)")
        
        normalized_df = df.copy()
        for column in ["text1", "text2"]: # assume columns to be "text1", "text2" (and "same")
            normalized_df[column] = normalized_df[column].map(lambda value: normalize_text(value, config=config))
        normalized_dict[split] = normalized_df

    return normalized_dict


def summary_stats(
    dict_df: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    for split, df in dict_df.items():
        row: dict[str, Any] = {"split": split, "num_rows": len(df)}
        for column in ["text1", "text2"]:
            word_length = df[column].str.split().str.len()
            char_length = df[column].str.len()
            row[f"{column}_mean_word_length"] = round(word_length.mean(), 2)
            row[f"{column}_mean_char_length"] = round(char_length.mean(), 2)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    print("\nNormalization summary:")
    print(summary_df)
    return summary_df


def normalization_wrapper(
    dict_df: dict[str, pd.DataFrame],
    config: Config = config,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:

    if config.verbose:
        print("\n======= NORMALIZATION START =======\n")

    normalized_dict_df = normalize_splits(dict_df, config=config)
    normalization_summary_df = summary_stats(normalized_dict_df)

    if config.verbose:
        print("\n======= NORMALIZATION END =======\n")

    return normalized_dict_df, normalization_summary_df
