# Experiment protocol (frozen before any model was fitted)

Frozen on: 2026-10-11 13:12 UTC, git tag `protocol-v1`.
Anything changed after that tag must be listed under "Deviations" at the bottom, with date and reason.

## 1. What this experiment is, and what it is not

This is a **retrospective, event-time forecasting experiment** on a historical file.
It is **not** a real-time operational system and makes **no claim** of operational usefulness.

- `started_at` is the time of an event as recorded by a volunteer Telegram channel (eTryvoga).
  The moment the information became *publicly available* may differ (message delay, editing, later
  corrections, back-filling). We do **not** model that delay. A forecast "made at time t" means:
  "uses only recorded events whose event time is <= t". In a real system the same information might
  have arrived later than t.
- The file is a **snapshot** (upstream commit `bf0c323`, committed 2026-10-11 04:50 UTC; last record
  04:36:59 UTC). We do not know whether it is complete, or whether older rows were revised after the
  fact. Completeness is **not** claimed. All results describe *this snapshot*.
- An alert is an official/volunteer **siren notification**, not an attack. Alert counts are not attack counts.

## 2. Target

For each region r (25 regions) and each whole hour t (UTC):
y(r, t) = 1 if at least one alert in r has t < started_at <= t + 3 hours, otherwise 0.
Alerts already in progress at t stay in the sample; in this file alerts within one region never overlap
(checked), so y=1 then requires the current alert to end and a new one to start within 3 hours.
Records with naive=True count as real alert *starts*; their `finished_at` (an invented start+30 min) is never used.

## 3. Information available at forecast time t (primary model)

Only alert **start** times <= t. No `finished_at`, no `naive` flag (it is only known after the fact),
no durations, no statistics computed over the full dataset. Calendar features use Europe/Kyiv local time.

## 4. Splits (forecast origins t, UTC, both ends inclusive)

| Split | First origin | Last origin |
|---|---|---|
| train_for_selection (365 d) | 2025-08-16 00:00 | 2026-08-15 21:00 |
| validation (28 d) | 2026-08-16 00:00 | 2026-09-12 21:00 |
| train_final (365 d) | 2025-09-13 00:00 | 2026-09-12 21:00 |
| test (28 d) | 2026-09-13 00:00 | 2026-10-10 21:00 |

Because a label looks 3 hours ahead, the last origin of each split ends 3 hours before the next
split starts, so no label window crosses a boundary. No label uses alerts after 2026-10-11 00:00 UTC
(the data cutoff, about 4.6 h before the last record in the snapshot). Features at later origins may
look back into earlier splits: that is past information, not leakage.

## 5. Models and selection

- B0: per-region alert-start frequency over the last 28 days of the training window.
- B1: per-region x Kyiv-hour frequency over the last 90 days of the training window, shrunk towards B0.
- One scikit-learn model: `HistGradientBoostingClassifier` (region as categorical, `early_stopping=False`,
  fixed `random_state`). Grid (8 combinations): max_depth {3, 6} x learning_rate {0.05, 0.1} x max_iter {100, 300}.
  Choose by **validation Brier score**. Then refit on train_final with the chosen settings.

## 6. Evaluation

Primary metric: Brier score on the test split. Also log-loss, PR-AUC, calibration curve, and skill score
vs B0 and B1; 95% interval by day-block bootstrap. Results are reported whatever they are; beating a
baseline is not a success criterion.

## 7. Rules

1. The test split is evaluated **once**, after the model configuration is fixed on validation data only.
2. No decision about features, hyper-parameters, or boundaries may use test-split labels.
3. Optional analyses (official-vs-volunteer comparison, extra variants) come after the core pipeline
   and its leakage tests have been run and shown.

## 8. Disclosure about what was looked at before freezing

Before this protocol was frozen, a one-off exploratory script computed, per region, the share of hours in
2026-09-13 .. 2026-10-11 containing at least one alert start (descriptive only; no model was fitted).
The start of the test window (2026-09-13, i.e. 28 days before the snapshot's last day) was written into that script
before its output was seen; the test end was later set to 2026-10-10 21:00 only because of the data cutoff (section 4),
not because of label values.
One consequence is noted: two regions (Luhanska, Zakarpatska) had no alert starts in that window, so
rank-based metrics (PR-AUC) are undefined for them. That fact is known from the exploratory look.

## Deviations (after protocol-v1)

None yet.
