# What was wrong, and what was changed

| # | File | Problem in the old code | Fix |
|---|------|------------------------|-----|
| 1 | `main.py`, `stability.py` | **Stale hold timer.** `update()` was only called while a hand was visible. Hold NEXT, drop your hand, return 10 s later showing NEXT, and it fired on the first frame back (old timer still running). | Added `GestureFilter.reset()`; `main.py` calls it when no hand is detected and when a frame errors. Verified by simulation. |
| 2 | `stability.py` | **Auto-repeat.** Holding a gesture fired it again every ~1.5 s (slides kept advancing). | Added a *latch*: a gesture fires once per hold and must be released (NEUTRAL or another gesture) before it can fire again. |
| 3 | `stability.py` | **No smoothing.** One flickering frame reset the hold timer, so holds felt unreliable. | Majority vote over the last 0.4 s (>=70% of frames must agree). |
| 4 | `stability.py` | ARM hold was 0.3 s (easy to toggle by accident); a slide could fire right after arming. | ARM hold = 0.8 s; slide lockout after every arm/disarm; `time.monotonic()` instead of `time.time()`. |
| 5 | `collect.py` | **No person/session column**, so leave-one-person-out evaluation was impossible. | `--person` argument (required); writes `person_id` and `session`; refuses to append to an old-format file. |
| 6 | `collect.py` | CSV reopened on every frame; thousands of near-identical rows; transition frames recorded. | File opened once, `--save-every 2`, 2 s countdown before recording starts, `finally` closes the file. |
| 7 | `collect.py` | Comments said "Shaka"; `chr(key)` ran when no key was pressed. | Comments say "L sign"; skips key code 255. |
| 8 | `train.py` | **Random split gave 99.6% on near-duplicate frames** (leakage). | Leave-one-person-out CV (needs `person_id`); falls back to contiguous-block CV with a warning if there is only one person. |
| 9 | `train.py` | Only Random Forest, no comparison; slow `iterrows` with mock objects. | Compares RandomForest, SVM, kNN, MLP; picks best by macro-F1; vectorised loading; saves comparison table + confusion matrix (CSV/PNG) in `results/`. |
| 10 | `train.py`, `main.py` | Model pickle had no version info. | Saved as a dict with feature version and scikit-learn version; `main.py` refuses a model trained on different features. |
| 11 | `features.py` | Train/live paths duplicated logic; thumb/index "L" shape not captured explicitly. | One function `features_from_array()` used everywhere; added 8 features (thumb-index angle, thumb distances, fingertip-wrist distances). **Old models must be retrained.** |
| 12 | `rules.py` | ARM was still the Shaka (scored 0/759 on L data); thumb test (`x4 < x3`) failed for 62% of open-palm frames. | ARM = L (index up, others curled, thumb far from index base, relative to hand size); PREVIOUS = four fingers up, no thumb test. |
| 13 | `test_baseline.py` | Different CSV path/mock classes than training. | Shares `load_csv`; adds per-person accuracy. |
| 14 | `main.py` | Errors printed every frame; stale window title; no hold feedback. | Each new error printed once; hold-progress bar, confidence shown; `--dry-run` mode that counts triggers (use it for the false-trigger test). |
| 15 | `actions.py` | No dry-run option. | `trigger_action(..., dry_run=True)`. |
| 16 | repo | Stray empty file `git`; no `matplotlib`; Python version unspecified. | Not included; `matplotlib` added; requirements note Python 3.10/3.11. |

## Run order
1. `python collect.py --person alice` (repeat for each friend; keys 0-3, `s` stop, `q` quit)
2. `python test_baseline.py`
3. `python train.py`
4. `python main.py --dry-run` (count accidental triggers for 5 minutes), then `python main.py`

To test the pipeline before you have new data: `python train.py --data data/landmarks_old_format.csv`
(single-person data -> block CV with a warning).

## What was tested (and what was not)
Tested: feature extraction (single == vectorised), `train.py` (block-CV and leave-one-person-out code paths),
`test_baseline.py`, and the stability filter (stale timer, auto-repeat, flicker, cooldowns, arming lockout) with simulated frames.
NOT run: `collect.py` and `main.py` were compile-checked only (no webcam here). Test them with a webcam first.
The `0.40` thumb threshold in `rules.py` was picked on the old single-person data; retune on the new data.
