from __future__ import annotations

import json
from xgboost import XGBClassifier
from helpers import save_json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

@dataclass(slots=True)
class Config:
    # local pairwise
    include_statistical: bool = True
    include_tfidf: bool = True
    include_char_ngrams: bool = True
    include_pos_ngrams: bool = True
    include_readability: bool = True

    include_local_pairwise: bool = True # local features pairwise operations; this will override the local pairwise booleans config
    include_global_pairwise: bool = True # global features using cosine similarity

    threshold_metric: str = "youden_j" # reducing false positives
    threshold_grid_step: float = 0.01
    model_params: dict[str, Any] | None = None
    def __post_init__(self) -> None:
        if self.model_params is None: # hyperparameters for XGBoost

            # include local features
            self.model_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.3,
                "min_child_weight": 3,
                "reg_lambda": 5.0,
                "reg_alpha": 1.0,
                "random_state": 42,
                "n_jobs": 2,
                "tree_method": "hist",
            }

            # # only global featuers
            # self.model_params = {
            #     "objective": "binary:logistic",
            #     "eval_metric": "logloss",
            #     "n_estimators": 200,
            #     "max_depth": 2,
            #     "learning_rate": 0.03,
            #     "subsample": 0.9,
            #     "colsample_bytree": 1.0,
            #     "min_child_weight": 5,
            #     "reg_lambda": 10.0,
            #     "reg_alpha": 0.5,
            #     "random_state": 42,
            #     "n_jobs": 2,
            #     "tree_method": "hist",
            # }



def _feature_family_from_suffix(suffix: str) -> str:
    if suffix.startswith("tfidf_"):
        return "tfidf"
    if suffix.startswith("char") and "_tfidf_" in suffix:
        return "char_ngrams"
    if suffix.startswith("pos") and "_tfidf_" in suffix:
        return "pos_ngrams"
    if suffix.startswith("readability_"):
        return "readability"
    return "statistical"

def _include_family(family: str, config: Config) -> bool:
    return {
        "statistical": config.include_statistical,
        "tfidf": config.include_tfidf,
        "char_ngrams": config.include_char_ngrams,
        "pos_ngrams": config.include_pos_ngrams,
        "readability": config.include_readability,
    }[family]


# self-build cosine similarity function
def _safe_cosine_from_columns(left: sparse.csr_matrix, right: sparse.csr_matrix) -> np.ndarray:
    numerator = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    left_norm = np.sqrt(np.asarray(left.multiply(left).sum(axis=1)).ravel())
    right_norm = np.sqrt(np.asarray(right.multiply(right).sum(axis=1)).ravel())
    denominator = left_norm * right_norm
    result = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32), where=denominator > 0)
    return result.astype(np.float32)


def discover_suffixes(train_df: pd.DataFrame, config: Config) -> list[str]:
    suffixes: list[str] = []
    for column in train_df.columns:
        if not column.startswith("text1_"):
            continue
        suffix = column[len("text1_") :]
        if _include_family(_feature_family_from_suffix(suffix), config):
            suffixes.append(suffix)
    return suffixes

# summary global features
def build_global_pairwise_features(df: pd.DataFrame, suffixes: list[str]) -> tuple[Any, list[str]]:

    dtype = np.float32
    blocks = []
    feature_names = []

    families = {
        "tfidf": [s for s in suffixes if s.startswith("tfidf_")],
        "char_ngrams": [s for s in suffixes if s.startswith("char") and "_tfidf_" in s],
        "pos_ngrams": [s for s in suffixes if s.startswith("pos") and "_tfidf_" in s],
        "scalar": [s for s in suffixes if not (
            s.startswith("tfidf_")
            or (s.startswith("char") and "_tfidf_" in s)
            or (s.startswith("pos") and "_tfidf_" in s)
        )
        ]}

    for family_name, family_suffixes in families.items():
        if not family_suffixes:
            continue

        left_cols = [f"text1_{s}" for s in family_suffixes]
        right_cols = [f"text2_{s}" for s in family_suffixes]

        left = sparse.csr_matrix(df[left_cols].to_numpy(dtype=dtype))
        right = sparse.csr_matrix(df[right_cols].to_numpy(dtype=dtype))

        diff = left - right

        cosine = _safe_cosine_from_columns(left, right).reshape(-1, 1) # cosine similarity
        l1 = np.asarray(np.abs(diff).sum(axis=1)).ravel().astype(dtype).reshape(-1, 1) # l1 distance
        l2 = np.sqrt(np.asarray(diff.multiply(diff).sum(axis=1)).ravel()).astype(dtype).reshape(-1, 1) # l2 distance

        family_block = sparse.csr_matrix(np.hstack([cosine, l1, l2]), dtype=dtype)
        blocks.append(family_block)
        feature_names.extend([
            f"{family_name}_cosine_similarity",
            f"{family_name}_l1_distance",
            f"{family_name}_l2_distance",
        ])

    if not blocks:
        return sparse.csr_matrix((len(df), 0), dtype=dtype), []

    return sparse.hstack(blocks, format="csr", dtype=dtype), feature_names


# local features (pairwise operations done for each feature column)
def build_pairwise_matrix(df: pd.DataFrame, suffixes: list[str]) -> tuple[Any, np.ndarray, list[str]]:
    dtype = np.float32
    feature_names: list[str] = []

    columns: list[sparse.csr_matrix] = []
    for suffix in suffixes:

        # operate one for each column for pairwise operations
        left = sparse.csr_matrix(df[f"text1_{suffix}"].to_numpy(dtype=dtype).reshape(-1, 1))
        right = sparse.csr_matrix(df[f"text2_{suffix}"].to_numpy(dtype=dtype).reshape(-1, 1))
        diff = left - right

        # asbolute difference
        columns.append(abs(diff))
        feature_names.append(f"{suffix}_abs_diff")

        # dot product
        columns.append(left.multiply(right))
        feature_names.append(f"{suffix}_product")

    X = sparse.hstack(columns, format="csr", dtype=dtype)
    y = df["same"].to_numpy(dtype=np.int8, copy=False) # binary label
    return X, y, feature_names

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    youden_j = sensitivity + specificity - 1.0
    return {
        "threshold": round(threshold, 5),
        "accuracy": round(accuracy_score(y_true, y_pred), 5),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 5),
        "recall": round(sensitivity, 5),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 5),
        "balanced_accuracy": round(balanced_accuracy, 5),
        "specificity": round(specificity, 5),
        "youden_j": round(youden_j, 5),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 5),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# finding best threshold using grid search based on config.threshold_metric
# using different config.threshold_metric can lead to different performance for the classification (not proba)
def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    config: Config,
) -> tuple[float, dict[str, Any]]:
    
    thresholds = np.arange(0.0, 1.0+config.threshold_grid_step, config.threshold_grid_step, dtype=np.float32)
    if thresholds.size == 0:
        thresholds = np.array([0.5], dtype=np.float32)

    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_proba, threshold=best_threshold)
    best_score = float(best_metrics[config.threshold_metric])

    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_proba, threshold=float(threshold)) # compute metrics' value for each test threshold
        score = float(metrics[config.threshold_metric])
        # if current score > best_score...
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics

def train_and_save_model(save_root: str | Path | None = None, config: Config | None = None) -> dict[str, Any]:

    config = config or Config()
    project_root = Path(__file__).resolve().parents[1] # assuming this file under subfolder in project root
    saved_dir = project_root / "saved"
    ngram_dir = saved_dir / "ngram_features" / "dataframes"
    model_dir = Path(save_root) if save_root is not None else saved_dir / "model"

    ngram_dict_df = {split: pd.read_parquet(ngram_dir / f"{split}_ngram.parquet") for split in ("train", "validation", "test")}

    suffixes = discover_suffixes(ngram_dict_df["train"], config)
    X_by_split: dict[str, Any] = {}
    y_by_split: dict[str, np.ndarray] = {}
    feature_names: list[str] = []

    for split, df in ngram_dict_df.items():
        
        blocks = []
        feature_names = []

        if config.include_local_pairwise:
            X_pairwise, y, local_names = build_pairwise_matrix(df, suffixes)
            blocks.append(X_pairwise)
            feature_names.extend(local_names)
        else:
            y = df["same"].to_numpy(dtype=np.int8, copy=False)

        if config.include_global_pairwise:
            X_global, global_names = build_global_pairwise_features(df, suffixes)
            blocks.append(X_global)
            feature_names.extend(global_names)

        if not blocks: # if both local and global features are set to false
            raise ValueError("At least one of include_local_pairwise or include_global_pairwise must be True.")
        X = sparse.hstack(blocks, format="csr", dtype=np.float32)

        X_by_split[split] = X
        y_by_split[split] = y

    model = XGBClassifier(**config.model_params) # the model can be changed as desired
    model.fit(X_by_split["train"], y_by_split["train"]) # fitting

    validation_proba = model.predict_proba(X_by_split["validation"])[:, 1]
    best_threshold, validation_metrics = find_best_threshold(
        y_by_split["validation"],
        validation_proba,
        config)
    test_proba = model.predict_proba(X_by_split["test"])[:, 1]
    test_metrics = compute_metrics(y_by_split["test"], test_proba, threshold=best_threshold)


    # saving
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "model.json")
    save_json({"threshold": best_threshold}, model_dir / "threshold.json") # saving best threshold
    save_json({ # saving features used for training, including original features
            "suffixes": suffixes,
            "feature_names": feature_names,
        }, 
        model_dir / "feature_spec.json")
    save_json({ # saving model performance metrics
            "validation": validation_metrics,
            "test": test_metrics,
        }, 
        model_dir / "metrics.json")
    save_json(asdict(config), model_dir / "training_config.json") # saving model config

    return {
        "model_dir": str(model_dir),
        "model_path": str(model_dir / "model.json"),
        "threshold": best_threshold,
        "suffixes": suffixes,
        "feature_names": feature_names,
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        }}



if __name__ == "__main__":
    outputs = train_and_save_model()
    print(f"Saved model bundle to: {outputs['model_dir']}")
    print(json.dumps(outputs["metrics"], indent=2))
