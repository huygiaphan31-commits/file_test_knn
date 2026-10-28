# KNN k-selection benchmark (validation only)

- Dataset: EMNIST ByClass official train IDX; test IDX not read.
- Input images from `data/my_samples/` were not used.
- Selected 500/class pool: 31,000; scoring train: 30,380; validation: 620 (10/class).
- Same deterministic split, same normalized vectors and same macro-F1 implementation were used for every k.
- KNN algorithm, Euclidean distance, vote and tie-break are unchanged.

| k | Validation macro-F1 | Evaluation time (s) | Scoring train samples | Validation samples |
|---:|---:|---:|---:|---:|
| 9 | 0.591435475986 | 45.932 | 30,380 | 620 |
| 25 | 0.574870089110 | 47.941 | 30,380 | 620 |
| 50 | 0.543152699549 | 46.332 | 30,380 | 620 |
| 100 | 0.517974172619 | 45.402 | 30,380 | 620 |
| 200 | 0.497809576016 | 45.732 | 30,380 | 620 |
| 500 | 0.453477268665 | 45.904 | 30,380 | 620 |
| 1000 | 0.398248683684 | 48.273 | 30,380 | 620 |

Best measured configuration by macro-F1 (tie: smaller k): **k=9**, macro-F1 `0.591435475986`.

## Comparison with previous artifacts

- The 2026-10-24 k=1000 paired report records 500/class, 31,000 selected samples, 30,380 scoring train samples, 620 validation samples, seed 42, shared preprocessing, and scores 0.591435475986 (k=9) / 0.398248683684 (k=1000).
- The historical report does not store validation sample IDs or a split fingerprint. The current run stores hashes, but exact row-by-row split identity with that older run cannot be proven. Treat numerical agreement (if observed) as a reproduction check, not a cryptographic proof of identical samples.
- The 2026-10-22 train-size experiment used one fixed 620-sample set selected from a 510/class shuffled pool, then nested training subsets. That data-selection protocol differs from the current 500/class reservoir-then-split protocol. Do not compare its k=9 score 0.614766637049 directly to this run as if conditions were identical.

## Limitations

- Validation is from official EMNIST training data, not the official test set or external handwriting.
- Evaluating seven candidates on the same validation set can bias the winning validation score upward.
- Per-k time includes a full neighbor search for every validation row; load and preprocessing are reported separately and excluded from per-k time.
- This experiment does not diagnose or evaluate segmentation.

Files: `results.csv` (per-k scores/times/sample counts) and `summary.json` (configuration, hashes and comparison notes).
