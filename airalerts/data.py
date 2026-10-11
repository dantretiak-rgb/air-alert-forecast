"""Small helpers to read the raw CSV files. Nothing here builds labels or features."""
import hashlib
import json

import pandas as pd

from . import config


def sha256_file(path, chunk_size=1 << 20):
    """SHA-256 fingerprint of a file. If one byte changes, the fingerprint changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path=config.MANIFEST_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_raw(path):
    """Read every column as plain text, so validation sees exactly what is in the file."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def parse_times(df):
    """Return a copy with started_at / finished_at as timezone-aware UTC timestamps.
    Unparseable values become NaT (the validator counts them)."""
    out = df.copy()
    for col in ("started_at", "finished_at"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce", format="ISO8601")
    if "naive" in out.columns:
        out["naive"] = out["naive"].map({"True": True, "False": False})
    return out
