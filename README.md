# Air-raid alert onset forecasting (retrospective experiment)

Hourly probability, per Ukrainian region, that **at least one new air-raid alert starts in the next 3 hours**.
This is a **retrospective event-time experiment** on a historical snapshot. It is **not** a proven real-time
system, and alert counts are not attack counts. See `PROTOCOL.md` for the frozen protocol and its limits.

## Status

| Block | Content | State |
|---|---|---|
| 1 | Repository, pinned snapshot, data validation, SHA-256 | done |
| 2 | Hourly grid, target, leakage tests | not started |
| 3 | Baseline + one scikit-learn model, chronological evaluation | not started |
| 4 | Optional: comparison with the official dataset | not started |

## Data

- Primary: `volunteer_data_en.csv` (eTryvoga volunteer channel), secondary (comparison only): `official_data_en.csv`.
- Source: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset, **pinned to commit `bf0c323c78e3e4508721458797dcd38c95df13f3`** (MIT license, see `data/UPSTREAM_LICENSE.txt`).
- `data/MANIFEST.json` stores the expected SHA-256, row count and last timestamp of each file.
- The volunteer file is committed here; the official file (31 MB) is downloaded and hash-verified on demand.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m airalerts.fetch_data     # downloads/verifies the pinned files
python -m airalerts.validate_data  # data checks (exit code 0 = all passed)
python -m pytest -v                # tests
```

## Known limitations (so far)

- The snapshot's completeness and later revisions are unknown; the volunteer feed is not an official record.
- 5.0% of volunteer rows are `naive=True`: their end time is invented (start + 30 min) and is never used.
- The official file is only available up to 2026-09-07 and switched from oblast to mostly raion-level records,
  so it cannot cover the test period.
