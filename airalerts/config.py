"""Frozen settings. Changing anything in the 'FROZEN PROTOCOL' part after the
protocol-v1 git tag must be recorded in PROTOCOL.md under 'Deviations'."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "MANIFEST.json"

VOLUNTEER_FILE = "volunteer_data_en.csv"
OFFICIAL_FILE = "official_data_en.csv"

VOLUNTEER_COLUMNS = ["region", "started_at", "finished_at", "naive"]
OFFICIAL_COLUMNS = ["oblast", "raion", "hromada", "level", "started_at", "finished_at", "source"]
OFFICIAL_LEVELS = {"oblast", "raion", "hromada"}

# The 25 regions observed in the volunteer file at the pinned commit
# (Kyiv City is separate from Kyivska oblast; Crimea is not present).
EXPECTED_REGIONS = sorted([
    "Cherkaska oblast", "Chernihivska oblast", "Chernivetska oblast",
    "Dnipropetrovska oblast", "Donetska oblast", "Ivano-Frankivska oblast",
    "Kharkivska oblast", "Khersonska oblast", "Khmelnytska oblast",
    "Kirovohradska oblast", "Kyiv City", "Kyivska oblast", "Luhanska oblast",
    "Lvivska oblast", "Mykolaivska oblast", "Odeska oblast", "Poltavska oblast",
    "Rivnenska oblast", "Sumska oblast", "Ternopilska oblast", "Vinnytska oblast",
    "Volynska oblast", "Zakarpatska oblast", "Zaporizka oblast", "Zhytomyrska oblast",
])

# Every timestamp in both files is UTC and written like 2026-10-11 04:36:59+00:00
TIMESTAMP_REGEX = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?\+00:00"
NAIVE_ALERT_MINUTES = 30  # the dataset invents "start + 30 min" when no end message exists

# ----------------------------- FROZEN PROTOCOL -----------------------------
HORIZON_HOURS = 3  # target: >=1 new alert starts in (t, t + 3h]

# No alert after this instant is used for any label. The snapshot was committed
# upstream at 2026-10-11 04:50 UTC; cutting at 00:00 UTC leaves a safety margin.
DATA_CUTOFF = pd.Timestamp("2026-10-11 00:00", tz="UTC")

# Forecast origins t (hourly, UTC). Both ends inclusive.
SPLITS = {
    "train_for_selection": ("2025-08-16 00:00", "2026-08-15 21:00"),
    "validation":          ("2026-08-16 00:00", "2026-09-12 21:00"),
    "train_final":         ("2025-09-13 00:00", "2026-09-12 21:00"),
    "test":                ("2026-09-13 00:00", "2026-10-10 21:00"),
}


def split_bounds(name):
    """Return (first_origin, last_origin) as UTC Timestamps."""
    start, end = SPLITS[name]
    return pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
