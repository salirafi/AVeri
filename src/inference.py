from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from textstat import textstat
from xgboost import XGBClassifier

from helpers import load_json, load_pickle

from masking_regex import mask_split as regex_mask_split
from masking_spacy import Config as SpacyMaskingConfig, _apply_ner_mask, _build_linguistic_record, load_nlp_model
from normalization import normalize_text, Config as NormalizationConfig
from features_statistical import extract_split_statistics, Config as StatisticalConfig
from features_tfidf import record_to_tfidf_text, Config as TFIDFConfig
from features_ngram import Config as NGramConfig, build_space_free_char_ngrams, record_to_pos_sequence
from model_training import Config as TrainingConfig


    
def _coerce_tfidf_config(payload: dict[str, Any]) -> TFIDFConfig:
    payload = dict(payload)
    if isinstance(payload.get("ngram_range"), list):
        payload["ngram_range"] = tuple(payload["ngram_range"])
    return TFIDFConfig(**payload)
def _coerce_ngram_config(payload: dict[str, Any]) -> NGramConfig:
    payload = dict(payload)
    if isinstance(payload.get("pos_ngram_range"), list):
        payload["pos_ngram_range"] = tuple(payload["pos_ngram_range"])
    return NGramConfig(**payload)
def _coerce_statistical_config(payload: dict[str, Any]) -> StatisticalConfig:
    payload = dict(payload)
    if isinstance(payload.get("phrase_role_dependency_labels"), list):
        payload["phrase_role_dependency_labels"] = tuple(payload["phrase_role_dependency_labels"])
    return StatisticalConfig(**payload)
def _coerce_training_config(payload: dict[str, Any]) -> TrainingConfig:
    payload = dict(payload)
    return TrainingConfig(**payload)

@dataclass(slots=True)
class PredictionResult:
    probability_same: float
    predicted_label: int
    threshold: float
    normalized_text1: str
    normalized_text2: str
    masked_text1: str
    masked_text2: str
    def to_dict(self) -> dict[str, Any]:
        label = "Same author" if self.predicted_label == 1 else "Different author"
        return {
            "label": label,
            "probability": self.probability_same,
            "threshold": self.threshold,
            "normalized_text1": self.normalized_text1,
            "normalized_text2": self.normalized_text2,
            "masked_text1": self.masked_text1,
            "masked_text2": self.masked_text2,
        }



# STAND-ALONE PIPELINE TO PERFORM INFERENCE USING THE TRAINED MODEL
class Inference:
    def __init__(self, project_root: str | Path | None = None) -> None:

        self.project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
        self.saved_dir = self.project_root / "saved"
        self.model_dir = self.saved_dir / "model"

        # =============================
        # the pipeline follows what is done in src/pipeline.py but adapted to do inference instead of training
        # =============================

        self.normalization_config = NormalizationConfig(**load_json(self.saved_dir / "normalization" / "normalization_config.json"))

        spacy_payload = load_json(self.saved_dir / "masking" / "spacy_config.json")
        spacy_payload["verbose"] = False
        spacy_payload["nlp_n_process"] = 1
        self.spacy_config = SpacyMaskingConfig(**spacy_payload)

        statistical_payload = load_json(self.saved_dir / "masking" / "statistical_config.json")
        statistical_payload["verbose"] = False
        self.statistical_config = _coerce_statistical_config(statistical_payload)

        tfidf_payload = load_json(self.saved_dir / "tfidf_features" / "tfidf_config.json")
        tfidf_payload["verbose"] = False
        self.tfidf_config = _coerce_tfidf_config(tfidf_payload)

        ngram_payload = load_json(self.saved_dir / "ngram_features" / "ngram_config.json")
        ngram_payload["verbose"] = False
        self.ngram_config = _coerce_ngram_config(ngram_payload)

        training_payload = load_json(self.saved_dir / "model" / "training_config.json")
        self.training_config = _coerce_training_config(training_payload)

        self.tfidf_vectorizer = load_pickle(self.saved_dir / "tfidf_features" / "vectorizer.pkl")
        self.char_vectorizer = load_pickle(self.saved_dir / "ngram_features" / "char_vectorizer.pkl")
        self.pos_vectorizer = load_pickle(self.saved_dir / "ngram_features" / "pos_vectorizer.pkl")

        self.model = None
        self.threshold = float(load_json(self.model_dir / "threshold.json")["threshold"])

        feature_spec = load_json(self.model_dir / "feature_spec.json")
        self.suffixes: list[str] = feature_spec["suffixes"]
        # self.pairwise_operations: tuple[str, ...] = tuple(feature_spec["pairwise_operations"])
        self.pairwise_column_pairs = [(f"text1_{suffix}", f"text2_{suffix}") for suffix in self.suffixes]

        self.metrics = load_json(self.model_dir / "metrics.json")
        self.nlp = None

    def _load_model(self) -> XGBClassifier:
        if self.model is None:
            model_path = self.model_dir / "model.json"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing '{model_path}'")
            model = XGBClassifier()
            model.load_model(model_path)
            self.model = model
        return self.model


    def _predict_positive_proba(self, X: np.ndarray) -> float:
        model = self._load_model()
        return float(model.predict_proba(X)[0, 1])

    def _mask_one_text(self, text: str) -> tuple[str, dict[str, Any]]:
        if self.nlp is None:
            self.nlp = load_nlp_model(config=self.spacy_config)
        doc = self.nlp(text)
        masked_text, _ = _apply_ner_mask(text, doc)
        record = _build_linguistic_record(doc)
        return masked_text, record
    
    def _build_pairwise_vector(self, feature_df: pd.DataFrame) -> np.ndarray:
        row_values = feature_df.iloc[0].to_dict()
        width = len(self.pairwise_column_pairs) * 2 # two pairwise operations: abs. diff & dot product
        X_pair = np.empty((1, width), dtype=np.float32)
        column_index = 0
        for left_col, right_col in self.pairwise_column_pairs:
            left = np.float32(row_values.get(left_col, 0.0))
            right = np.float32(row_values.get(right_col, 0.0))
            diff = left - right
            X_pair[0, column_index] = abs(diff)
            X_pair[0, column_index + 1] = left * right
            column_index += 2

        return X_pair

    def _family_suffix_groups(self) -> dict[str, list[str]]:
        return {
            "tfidf": [s for s in self.suffixes if s.startswith("tfidf_")],
            "char_ngrams": [s for s in self.suffixes if s.startswith("char") and "_tfidf_" in s],
            "pos_ngrams": [s for s in self.suffixes if s.startswith("pos") and "_tfidf_" in s],
            "scalar": [s for s in self.suffixes if not (
                s.startswith("tfidf_")
                or (s.startswith("char") and "_tfidf_" in s)
                or (s.startswith("pos") and "_tfidf_" in s)
            )],
        }
    
    def _build_global_pairwise_vector(self, feature_df: pd.DataFrame) -> np.ndarray:
        row_values = feature_df.iloc[0].to_dict()
        values: list[float] = []


        for family_suffixes in self._family_suffix_groups().values():
            if not family_suffixes:
                continue

            left = np.array([row_values.get(f"text1_{suffix}", 0.0) for suffix in family_suffixes], dtype=np.float32)
            right = np.array([row_values.get(f"text2_{suffix}", 0.0) for suffix in family_suffixes], dtype=np.float32)

            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            cosine = float(np.dot(left, right) / denominator) if denominator > 0 else 0.0
            diff = left - right
            l1 = float(np.abs(diff).sum())
            l2 = float(np.linalg.norm(diff))

            values.extend([cosine, l1, l2])

        return np.array(values, dtype=np.float32).reshape(1, -1)
    
    # predict prbability and classification of two given texts (input from the user)
    def predict(self, text1: str, text2: str, threshold: float | None = None) -> PredictionResult:

        threshold_value = self.threshold if threshold is None else float(threshold)

        pair_df = pd.DataFrame([{
                    "text1": normalize_text(text1, config=self.normalization_config),
                    "text2": normalize_text(text2, config=self.normalization_config),
                    "same": 0,
                }])

        regex_masked_df, _ = regex_mask_split(pair_df)

        # spaCy masking; not using nlp.pipe
        masked_text1, record1 = self._mask_one_text(regex_masked_df.iloc[0]["text1"])
        masked_text2, record2 = self._mask_one_text(regex_masked_df.iloc[0]["text2"])

        masked_df = regex_masked_df.copy()
        masked_df.at[0, "text1"] = masked_text1 # combining regex and spaCy masking
        masked_df.at[0, "text2"] = masked_text2 # ...

        split_cache = {"text1": [record1], "text2": [record2]} # the linguistic cache
        feature_df = pd.DataFrame() # initialize empty dataframe for the features

        # ======== statistical features ===========
        
        if self.training_config.include_statistical:
            feature_df = extract_split_statistics(
                masked_df,
                split_cache=split_cache,
                split_name="inference",
                config=self.statistical_config,
            )

        # ======== TF-IDF features ===========

        if self.training_config.include_tfidf:
            for column in ("text1", "text2"):
                docs = [record_to_tfidf_text(record, config=self.tfidf_config) for record in split_cache[column]]
                tfidf_matrix = self.tfidf_vectorizer.transform(docs).toarray()
                tfidf_cols = [f"{column}_tfidf_{index:05d}" for index in range(tfidf_matrix.shape[1])]
                tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols)
                feature_df = pd.concat([feature_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

        # ======== n-gram features ===========
        
        for column in ("text1", "text2"):

            if self.training_config.include_char_ngrams:
                char_docs = [
                    " ".join(build_space_free_char_ngrams(text, n=self.ngram_config.char_ngram_n))
                    for text in masked_df[column].tolist()]
                char_matrix = self.char_vectorizer.transform(char_docs).toarray()
                char_cols = [
                    f"{column}_char{self.ngram_config.char_ngram_n}_tfidf_{index:05d}"
                    for index in range(char_matrix.shape[1])]
                char_df = pd.DataFrame(char_matrix, columns=char_cols)
                feature_df = pd.concat([feature_df.reset_index(drop=True), char_df.reset_index(drop=True)], axis=1)

            if self.training_config.include_pos_ngrams:
                pos_docs = [" ".join(record_to_pos_sequence(record)) for record in split_cache[column]]
                pos_matrix = self.pos_vectorizer.transform(pos_docs).toarray()
                pos_cols = [
                    f"{column}_pos{self.ngram_config.pos_ngram_range}_tfidf_{index:05d}"
                    for index in range(pos_matrix.shape[1])]
                pos_df = pd.DataFrame(pos_matrix, columns=pos_cols)
                feature_df = pd.concat([feature_df.reset_index(drop=True), pos_df.reset_index(drop=True)], axis=1)

            continue

        # ======== readability features ===========

        if self.training_config.include_readability:
            readability_df = pd.DataFrame([{
                        "text1_readability_flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(masked_df.iloc[0]["text1"]), 5),
                        "text1_readability_gunning_fog": round(textstat.gunning_fog(masked_df.iloc[0]["text1"]), 5),
                        "text1_readability_smog": round(textstat.smog_index(masked_df.iloc[0]["text1"]), 5),
                        "text1_readability_coleman_liau": round(textstat.coleman_liau_index(masked_df.iloc[0]["text1"]), 5),

                        "text2_readability_flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(masked_df.iloc[0]["text2"]), 5),
                        "text2_readability_gunning_fog": round(textstat.gunning_fog(masked_df.iloc[0]["text2"]), 5),
                        "text2_readability_smog": round(textstat.smog_index(masked_df.iloc[0]["text2"]), 5),
                        "text2_readability_coleman_liau": round(textstat.coleman_liau_index(masked_df.iloc[0]["text2"]), 5)
                    }])
            feature_df = pd.concat([feature_df.reset_index(drop=True), readability_df.reset_index(drop=True)], axis=1)

        blocks: list[np.ndarray] = []
        if self.training_config.include_local_pairwise:
            blocks.append(self._build_pairwise_vector(feature_df)) # optimized
        if self.training_config.include_global_pairwise:
            blocks.append(self._build_global_pairwise_vector(feature_df))
        if not blocks:
            raise ValueError("At least one of include_local_pairwise or include_global_pairwise must be True.")

        X = np.hstack(blocks).astype(np.float32)
        probability_same = self._predict_positive_proba(X)
        predicted_label = int(probability_same >= threshold_value) # 1 if > threshold, otherwise 0

        return PredictionResult(
            probability_same=probability_same,
            predicted_label=predicted_label,
            threshold=threshold_value,
            normalized_text1=pair_df.iloc[0]["text1"],
            normalized_text2=pair_df.iloc[0]["text2"],
            masked_text1=masked_df.iloc[0]["text1"],
            masked_text2=masked_df.iloc[0]["text2"],
        )
