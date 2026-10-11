"""Data validation. Each check returns a Result(name, ok, detail).

Run:  python -m airalerts.validate_data
Exit code 0 = every check passed, 1 = at least one failed.
"""
import re
import sys
from dataclasses import dataclass

import pandas as pd

from . import config
from .data import load_manifest, parse_times, read_raw, sha256_file


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


# ------------------------- individual checks (work on DataFrames) -------------------------
def check_sha256(path, expected):
    got = sha256_file(path)
    return Result("SHA-256 equals manifest", got == expected, f"{got[:16]}... (expected {expected[:16]}...)")


def check_schema(raw, expected_columns):
    ok = list(raw.columns) == list(expected_columns)
    return Result("columns exactly as expected", ok, f"found {list(raw.columns)}")


def check_utc_strings(raw):
    """Every timestamp text must look like 2026-10-11 04:36:59+00:00 (explicit UTC)."""
    pattern = re.compile(config.TIMESTAMP_REGEX)
    bad = 0
    for col in ("started_at", "finished_at"):
        bad += int((~raw[col].map(lambda s: bool(pattern.fullmatch(s)))).sum())
    return Result("all timestamps are explicit UTC (+00:00)", bad == 0, f"{bad} malformed/non-UTC values")


def check_no_missing_times(parsed):
    n = int(parsed["started_at"].isna().sum() + parsed["finished_at"].isna().sum())
    return Result("no unparseable/missing timestamps", n == 0, f"{n} NaT values")


def check_end_after_start(parsed):
    n = int((parsed["finished_at"] < parsed["started_at"]).sum())
    return Result("finished_at >= started_at", n == 0, f"{n} rows end before they start")


def check_duplicates(parsed, expected_count):
    n = int(parsed.duplicated().sum())
    return Result("exact-duplicate rows equal manifest", n == expected_count, f"{n} duplicates (manifest: {expected_count})")


def check_regions(names, expected):
    found = sorted(set(names))
    missing, extra = sorted(set(expected) - set(found)), sorted(set(found) - set(expected))
    return Result("25 expected regions, no others", not missing and not extra,
                  f"{len(found)} found; missing={missing}; unexpected={extra}")


def check_naive_rule(parsed):
    """naive=True records must all have the invented duration (start + 30 min)."""
    if parsed["naive"].isna().any():
        return Result("naive column is True/False only", False, "other values present")
    dur = (parsed.loc[parsed["naive"], "finished_at"] - parsed.loc[parsed["naive"], "started_at"])
    off = int((dur != pd.Timedelta(minutes=config.NAIVE_ALERT_MINUTES)).sum())
    share = parsed["naive"].mean()
    return Result("naive=True rows all last exactly 30 min", off == 0,
                  f"{int(parsed['naive'].sum())} naive rows ({share:.1%}); {off} violate the 30-min rule")


def check_no_overlap(parsed):
    """Within one region, a new alert never starts before the previous one ended."""
    d = parsed.sort_values(["region", "started_at"]).reset_index(drop=True)
    prev_end = d.groupby("region")["finished_at"].cummax().groupby(d["region"]).shift()
    n = int((d["started_at"] < prev_end).sum())
    return Result("no overlapping alerts within a region", n == 0, f"{n} rows start inside an earlier alert")


def check_manifest_stats(parsed, expected_rows, expected_last):
    last = parsed["started_at"].max()
    ok = len(parsed) == expected_rows and str(last) == expected_last
    return Result("row count and last start equal manifest", ok,
                  f"{len(parsed)} rows (manifest {expected_rows}); last start {last} (manifest {expected_last})")


def check_cutoff(parsed, cutoff):
    last = parsed["started_at"].max()
    return Result("DATA_CUTOFF is not after the last record", cutoff <= last,
                  f"cutoff {cutoff}; last record {last}; margin {last - cutoff}")


# ------------------------------- per-file validation -------------------------------
def validate_volunteer(path, manifest):
    info = manifest["files"][config.VOLUNTEER_FILE]
    results = [check_sha256(path, info["sha256"])]
    raw = read_raw(path)
    results.append(check_schema(raw, config.VOLUNTEER_COLUMNS))
    if not results[-1].ok:
        return results
    results.append(check_utc_strings(raw))
    parsed = parse_times(raw)
    results += [
        check_no_missing_times(parsed),
        check_end_after_start(parsed),
        check_duplicates(parsed, info["exact_duplicate_rows"]),
        check_regions(parsed["region"], config.EXPECTED_REGIONS),
        check_naive_rule(parsed),
        check_no_overlap(parsed),
        check_manifest_stats(parsed, info["rows"], info["last_started_at_utc"]),
        check_cutoff(parsed, config.DATA_CUTOFF),
    ]
    return results


def validate_official(path, manifest):
    info = manifest["files"][config.OFFICIAL_FILE]
    results = [check_sha256(path, info["sha256"])]
    raw = read_raw(path)
    results.append(check_schema(raw, config.OFFICIAL_COLUMNS))
    if not results[-1].ok:
        return results
    results.append(check_utc_strings(raw))
    parsed = parse_times(raw)
    levels = set(parsed["level"])
    results += [
        check_no_missing_times(parsed),
        check_end_after_start(parsed),
        check_duplicates(parsed, info["exact_duplicate_rows"]),
        check_regions(parsed["oblast"], config.EXPECTED_REGIONS),
        Result("level is only oblast/raion/hromada", levels <= config.OFFICIAL_LEVELS, f"found {sorted(levels)}"),
        check_manifest_stats(parsed, info["rows"], info["last_started_at_utc"]),
    ]
    return results


def print_results(title, results):
    print(f"\n=== {title} ===")
    for r in results:
        print(f"[{'PASS' if r.ok else 'FAIL'}] {r.name} -- {r.detail}")
    return all(r.ok for r in results)


def main():
    manifest = load_manifest()
    print(f"Pinned upstream commit: {manifest['upstream_commit']} ({manifest['upstream_commit_time_utc']})")
    all_ok = True
    vol = config.RAW_DIR / config.VOLUNTEER_FILE
    off = config.RAW_DIR / config.OFFICIAL_FILE
    all_ok &= print_results(f"{config.VOLUNTEER_FILE} (primary)", validate_volunteer(vol, manifest))
    if off.exists():
        all_ok &= print_results(f"{config.OFFICIAL_FILE} (secondary, comparison only)", validate_official(off, manifest))
    else:
        print(f"\n[SKIP] {config.OFFICIAL_FILE} not found. Run: python -m airalerts.fetch_data")
    print("\nRESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
