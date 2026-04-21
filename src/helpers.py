from __future__ import annotations

from typing import Any
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from scipy import sparse

# default JSON format
def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)

# for config
def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
        

# for summary dataframe
def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# saving dataframe for each splot separately
def save_split_frames(split_dict: dict[str, pd.DataFrame], out_dir: Path, suffix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, df in split_dict.items():
        df.to_parquet(out_dir / f"{split}_{suffix}.parquet", index=False)



def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def save_array_dict(array_dict: dict[str, np.ndarray], out_dir: Path, suffix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in array_dict.items():
        np.save(out_dir / f"{name}_{suffix}.npy", np.asarray(arr))

def save_matrix_dict(matrix_dict: dict[str, object], out_dir: Path, suffix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, matrix in matrix_dict.items():
        if sparse.issparse(matrix):
            sparse.save_npz(out_dir / f"{split}_{suffix}.npz", matrix)
        else:
            np.save(out_dir / f"{split}_{suffix}.npy", np.asarray(matrix))



def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
    
def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)