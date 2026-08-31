from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "input"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURE_ROOT = PROJECT_ROOT / "figures"
LOG_ROOT = PROJECT_ROOT / "logs"
EXTERNAL_ROOT = PROJECT_ROOT / "data" / "external"


def locate_workbook() -> Path:
    files = list(INPUT_ROOT.rglob("*.xlsx"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one official workbook, found {len(files)}")
    return files[0]


def locate_template(name: str) -> Path:
    files = list(INPUT_ROOT.rglob(name))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one template {name}, found {len(files)}")
    return files[0]


def load_sheets(*names: str) -> dict[str, pd.DataFrame]:
    workbook = locate_workbook()
    return {name: pd.read_excel(workbook, sheet_name=name) for name in names}


def minmax(values: np.ndarray | pd.Series, lo: float | None = None, hi: float | None = None):
    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr)) if lo is None else float(lo)
    hi = float(np.nanmax(arr)) if hi is None else float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite min-max bound")
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(arr), lo, hi
    return (arr - lo) / (hi - lo), lo, hi


def clip(x, lo: float, hi: float):
    return np.minimum(hi, np.maximum(lo, x))


def write_template_csv(df: pd.DataFrame, template_name: str, output_name: str) -> Path:
    template = locate_template(template_name)
    with template.open("r", encoding="utf-8-sig", newline="") as fh:
        expected = next(csv.reader(fh))
    actual = list(df.columns)
    if actual != expected:
        raise ValueError(f"Schema mismatch for {output_name}: expected={expected}, actual={actual}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / output_name
    # Preserve the template's BOM convention while keeping RFC-compatible CSV quoting.
    has_bom = template.read_bytes().startswith(b"\xef\xbb\xbf")
    df.to_csv(output, index=False, encoding="utf-8-sig" if has_bom else "utf-8", lineterminator="\n")
    return output


def dump_json(obj: Any, name: str) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_frame(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"Column {column} contains missing or non-finite values")

