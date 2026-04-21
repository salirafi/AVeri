from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import pandas as pd
import spacy
from tqdm.auto import tqdm

def _empty_record() -> dict[str, Any]:
    return {
        "tokens": [],
        "token_lower": [],
        "token_lemma": [],
        "token_pos": [],
        "token_dep": [],
        "token_is_punct": [],
        "token_is_space": [],
        "sentence_token_spans": [],
        "noun_chunk_spans": [],
    }

# addition to regex masking; cover en_core_web_lg NER labels
LABEL_TO_MASK: dict[str, str] = {
    "CARDINAL": "<CARDINAL>",
    "DATE": "<DATE>",
    "EVENT": "<EVENT>",
    "FAC": "<FACILITY>",
    "GPE": "<GPE>",
    "LANGUAGE": "<LANGUAGE>",
    "LAW": "<LAW>",
    "LOC": "<LOCATION>",
    "MONEY": "<MONEY>",
    "NORP": "<NORP>",
    "ORDINAL": "<ORDINAL>",
    "ORG": "<ORG>",
    "PERCENT": "<PERCENT>",
    "PERSON": "<PERSON>",
    "PRODUCT": "<PRODUCT>",
    "QUANTITY": "<QUANTITY>",
    "TIME": "<TIME>",
    "WORK_OF_ART": "<WORK_OF_ART>",
}



@dataclass(slots=True)
class Config:
    verbose: bool = True
    use_gpu: bool = False
    nlp_model: str = "en_core_web_sm"
    nlp_batch_size: int = 64
    nlp_n_process: int = 1 # in GPU, set to 1
    checkpoint_dir: str = "saved/masking/spacy_checkpoints"

config = Config()



def load_nlp_model(config: Config = config):

    disabled_components = [
        "entity_linker",
        "entity_ruler",
        "textcat",
        "textcat_multilabel",
        "sentencizer",
        "morphologizer",
        "trainable_lemmatizer",
    ]

    if not config.use_gpu:
        if config.verbose:
            print(f"\nLoading spaCy model '{config.nlp_model}' with NER, tagger, parser, senter, and lemmatizer enabled.")
        return spacy.load(config.nlp_model, disable=disabled_components)

    is_gpu_available = spacy.require_gpu()
    if is_gpu_available:
        print("\nGPU is available. spaCy will use the GPU for processing.")
        if config.verbose:
            print(f"Loading spaCy model '{config.nlp_model}' with NER, tagger, parser, senter, and lemmatizer enabled.")
    else: 
        raise RuntimeError("\nGPU is not available! Set use_gpu=False to continue.")

    return spacy.load(config.nlp_model, disable=disabled_components)

def _apply_ner_mask(text: str, doc: Any) -> tuple[str, dict[str, int]]:

    replacements: list[tuple[int, int, str]] = []
    counts = {label: 0 for label in LABEL_TO_MASK} # for summary

    # loop over entities in the text to tag them
    for ent in doc.ents:
        placeholder = LABEL_TO_MASK.get(ent.label_)
        if placeholder is None: # if not covered by LABEL_TO_MASK
            continue
        replacements.append((ent.start_char, ent.end_char, placeholder))
        counts[ent.label_] += 1

    if not replacements:
        return text, counts


    # apply the masking
    masked_parts: list[str] = []
    cursor = 0
    for start, end, placeholder in sorted(replacements, key=lambda item: item[0]):
        if start < cursor:
            continue
        masked_parts.append(text[cursor:start])
        masked_parts.append(placeholder) # insert the replacement token
        cursor = end
    masked_parts.append(text[cursor:])

    return "".join(masked_parts), counts



# saving spaCy assignments
def _build_linguistic_record(doc: Any) -> dict[str, Any]:
    sentence_token_spans = [(sentence.start, sentence.end) for sentence in doc.sents]
    noun_chunk_spans = [(chunk.start, chunk.end) for chunk in doc.noun_chunks]

    return {
        "tokens": [token.text for token in doc], # original token text
        "token_lower": [token.lower_ for token in doc], # lowercase form of each token
        "token_lemma": [token.lemma_ for token in doc], # dictionary form of each token
        "token_pos": [token.pos_ for token in doc], # POS tag (coarse-grained; see https://stackoverflow.com/questions/40288323/what-do-spacys-part-of-speech-and-dependency-tags-mean)
        "token_dep": [token.dep_ for token in doc], # dependency relation label for each token in the parse tree
        "token_is_punct": [token.is_punct for token in doc], # whether the token is punctuation
        "token_is_space": [token.is_space for token in doc], # whether the token is whitespace
        "sentence_token_spans": sentence_token_spans, # sentence boundaries; token index spans: (start, end)
        "noun_chunk_spans": noun_chunk_spans, # noun phrase spans; token index ranges: (start, end)
    }




def _mask_text_column(
    texts_for_nlp: list[str], # list of texts to mask
    nlp: Any,
    split_name: str,
    column: str,
    config: Config = config,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    
    final_texts: list[str] = []
    doc_records: list[dict[str, Any]] = [] # list over rows
    column_counts = {label: 0 for label in LABEL_TO_MASK}


    docs = (
        # using nlp.pipe for faster run with batching
        nlp.pipe(texts_for_nlp, batch_size=config.nlp_batch_size, n_process=config.nlp_n_process)
        if not config.use_gpu
        else nlp.pipe(texts_for_nlp, batch_size=config.nlp_batch_size, n_process=1) # make sure to set nlp_n_process=1 if use_gpu=True
    )
    docs = tqdm(docs, total=len(texts_for_nlp), desc=f"spaCy masking [{split_name}:{column}]")

    # beware: this will loop over rows again for additional masking, can be as long as regex_masking
    # also will loop over the entities for each text
    for text, doc in zip(texts_for_nlp, docs, strict=False):
        final_text, ner_counts = _apply_ner_mask(text, doc)
        final_texts.append(final_text)
        doc_records.append(_build_linguistic_record(doc))
        for key, count in ner_counts.items():
            column_counts[key] += count

    return final_texts, doc_records, column_counts


# saving progress after each loop over the columns
# this to safeguard against kernel crash mid-runtime
def _save_column_checkpoint(
    masked_dict_df: dict[str, pd.DataFrame],
    linguistic_cache: dict[str, dict[str, list[dict[str, Any]]]],
    column: str,
    config: Config = config,
) -> None:
    checkpoint_dir = Path(config.checkpoint_dir) / column
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for split, masked_df in masked_dict_df.items():
        masked_df.to_parquet(checkpoint_dir / f"{split}_spacy_masked.parquet", index=False)
    with open(checkpoint_dir / "linguistic_cache.pkl", "wb") as checkpoint_file:
        pickle.dump(linguistic_cache, checkpoint_file)



def mask_splits(
    dict_df: dict[str, pd.DataFrame],
    config: Config = config,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, dict[str, list[dict[str, Any]]]]]:
    """
    Mask every split using spaCy and build linguistic caches.
    """

    nlp = load_nlp_model(config=config)

    masked_dict_df: dict[str, pd.DataFrame] = {split: df.copy() for split, df in dict_df.items()}
    linguistic_cache: dict[str, dict[str, list[dict[str, Any]]]] = {split: {} for split in dict_df}
    split_counts_by_name: dict[str, dict[str, int]] = {split: {label: 0 for label in LABEL_TO_MASK} for split in dict_df}

    for column in ["text1", "text2"]:

        if config.verbose:
            print(f"\nProcessing spaCy masking column='{column}' across all splits")

        for split, masked_df in masked_dict_df.items():
            if config.verbose:
                print(f"  spaCy masking split='{split}' ({len(masked_df):,} rows)")

            texts_for_nlp = ["" if pd.isna(value) else str(value) for value in masked_df[column].tolist()] # loop over rows (may be slow if too many rows)
            final_texts, doc_records, column_counts = _mask_text_column( # perform spaCy masking
                texts_for_nlp,
                nlp=nlp,
                split_name=split,
                column=column,
                config=config,
            )
            masked_df[column] = final_texts
            linguistic_cache[split][column] = doc_records

            for key, count in column_counts.items():
                split_counts_by_name[split][key] += count

        _save_column_checkpoint(masked_dict_df, linguistic_cache, column=column, config=config) # saving progress to disk

        if config.verbose:
            print(f"\nSaved spaCy checkpoint for column='{column}' to '{Path(config.checkpoint_dir) / column}'.")

    summary_rows: list[dict[str, Any]] = []
    for split, masked_df in masked_dict_df.items():
        split_counts = split_counts_by_name[split]
        summary_row: dict[str, Any] = {"split": split, "num_rows": len(masked_df)}
        summary_row.update({f"{key}_count": int(value) for key, value in split_counts.items()})
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    return masked_dict_df, summary_df, linguistic_cache




def masking_wrapper(
    dict_df: dict[str, pd.DataFrame],
    config: Config = config,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, dict[str, list[dict[str, Any]]]],
]:
    if config.verbose:
        print("======= SPACY MASKING START =======")

    masked_dict_df, masking_summary_df, linguistic_cache = mask_splits(dict_df, config=config)

    if config.verbose:
        print("\nspaCy masking summary:")
        print(masking_summary_df)
        print("\n======= SPACY MASKING END =======\n")

    return masked_dict_df, masking_summary_df, linguistic_cache
