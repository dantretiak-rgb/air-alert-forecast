"""Block 1 tests: pinned data, validation, and the frozen split arithmetic."""
import pandas as pd
import pytest

from airalerts import config
from airalerts import validate_data as vd
from airalerts.data import load_manifest, parse_times, sha256_file

VOL = config.RAW_DIR / config.VOLUNTEER_FILE
OFF = config.RAW_DIR / config.OFFICIAL_FILE
needs_official = pytest.mark.skipif(not OFF.exists(), reason="run: python -m airalerts.fetch_data")


# --------------------------- real data: everything must pass ---------------------------
def test_volunteer_file_passes_all_checks():
    results = vd.validate_volunteer(VOL, load_manifest())
    failed = [r for r in results if not r.ok]
    assert not failed, failed


@needs_official
def test_official_file_passes_all_checks():
    results = vd.validate_official(OFF, load_manifest())
    failed = [r for r in results if not r.ok]
    assert not failed, failed


@needs_official
def test_region_names_match_between_sources():
    v = pd.read_csv(VOL, dtype=str)["region"]
    o = pd.read_csv(OFF, dtype=str)["oblast"]
    assert set(v) == set(o) == set(config.EXPECTED_REGIONS)


# ---------- the checks themselves must catch bad data (otherwise a pass means nothing) ----------
def make_frame(**overrides):
    base = {
        "region": ["Kyiv City", "Kyiv City"],
        "started_at": ["2026-01-01 10:00:00+00:00", "2026-01-01 12:00:00+00:00"],
        "finished_at": ["2026-01-01 10:30:00+00:00", "2026-01-01 12:30:00+00:00"],
        "naive": ["False", "True"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_detects_wrong_columns():
    raw = make_frame().rename(columns={"region": "oblast"})
    assert not vd.check_schema(raw, config.VOLUNTEER_COLUMNS).ok


def test_detects_non_utc_timestamp():
    raw = make_frame(started_at=["2026-01-01 10:00:00+02:00", "2026-01-01 12:00:00+00:00"])
    assert not vd.check_utc_strings(raw).ok


def test_detects_timestamp_without_timezone():
    raw = make_frame(started_at=["2026-01-01 10:00:00", "2026-01-01 12:00:00+00:00"])
    assert not vd.check_utc_strings(raw).ok


def test_detects_end_before_start():
    raw = make_frame(finished_at=["2026-01-01 09:00:00+00:00", "2026-01-01 12:30:00+00:00"])
    assert not vd.check_end_after_start(parse_times(raw)).ok


def test_detects_naive_row_with_real_looking_duration():
    raw = make_frame(naive=["False", "True"], finished_at=["2026-01-01 10:30:00+00:00", "2026-01-01 14:00:00+00:00"])
    assert not vd.check_naive_rule(parse_times(raw)).ok


def test_detects_overlapping_alerts_in_one_region():
    raw = make_frame(started_at=["2026-01-01 10:00:00+00:00", "2026-01-01 10:10:00+00:00"])
    assert not vd.check_no_overlap(parse_times(raw)).ok


def test_accepts_back_to_back_alerts():
    raw = make_frame(started_at=["2026-01-01 10:00:00+00:00", "2026-01-01 10:30:00+00:00"],
                     finished_at=["2026-01-01 10:30:00+00:00", "2026-01-01 11:00:00+00:00"])
    assert vd.check_no_overlap(parse_times(raw)).ok


def test_detects_unknown_region():
    assert not vd.check_regions(["Kyiv City", "Atlantis"], config.EXPECTED_REGIONS).ok


def test_detects_tampered_file(tmp_path):
    copy = tmp_path / "copy.csv"
    copy.write_bytes(VOL.read_bytes() + b"x\n")
    expected = load_manifest()["files"][config.VOLUNTEER_FILE]["sha256"]
    assert sha256_file(copy) != expected
    assert not vd.check_sha256(copy, expected).ok


# ----------- frozen protocol: pure date arithmetic, no data and no model involved -----------
def last_label_end(split):
    return config.split_bounds(split)[1] + pd.Timedelta(hours=config.HORIZON_HOURS)


def test_no_label_window_crosses_into_next_split():
    h = pd.Timedelta(hours=config.HORIZON_HOURS)
    # last origin of a split + 3h must not go beyond the first origin of the next one
    assert last_label_end("train_for_selection") <= config.split_bounds("validation")[0]
    assert last_label_end("validation") <= config.split_bounds("test")[0]
    assert last_label_end("train_final") <= config.split_bounds("test")[0]
    # ...and the three-hour gap is exactly used up (nothing wasted, nothing shared)
    assert config.split_bounds("validation")[0] - config.split_bounds("train_for_selection")[1] == h


def test_test_labels_end_at_or_before_cutoff():
    assert last_label_end("test") <= config.DATA_CUTOFF


def test_splits_are_ordered_and_non_overlapping():
    train = config.split_bounds("train_for_selection")
    val = config.split_bounds("validation")
    test = config.split_bounds("test")
    assert train[1] < val[0] and val[1] < test[0]


def test_training_windows_are_365_days():
    for name, end_next in (("train_for_selection", "validation"), ("train_final", "test")):
        start = config.split_bounds(name)[0]
        assert config.split_bounds(end_next)[0] - start == pd.Timedelta(days=365)


def test_origins_are_whole_hours():
    for name in config.SPLITS:
        for ts in config.split_bounds(name):
            assert ts == ts.floor("h")
