# Train-size and k validation experiment

- Data: EMNIST ByClass train IDX + mapping only; official test IDX was not opened.
- Validation: fixed, stratified 10 examples/class = 620, shared across all caps.
- Train pools: nested subsets from a single deterministic 500/class pool.
- Preprocessing: shared `dataset_images_to_vectors()` pipeline; metric: `macro_f1()`.

| Train cap/class | Train samples | k | Validation macro-F1 | Validation scoring time (all k jointly) |
|---:|---:|---:|---:|---:|
| 200 | 12400 | 1 | 0.560575 | 16.50 s |
| 200 | 12400 | 3 | 0.561025 | 16.50 s |
| 200 | 12400 | 5 | 0.566742 | 16.50 s |
| 200 | 12400 | 7 | 0.568428 | 16.50 s |
| 200 | 12400 | 9 | 0.562829 | 16.50 s |
| 300 | 18600 | 1 | 0.565490 | 23.41 s |
| 300 | 18600 | 3 | 0.589690 | 23.41 s |
| 300 | 18600 | 5 | 0.588010 | 23.41 s |
| 300 | 18600 | 7 | 0.589821 | 23.41 s |
| 300 | 18600 | 9 | 0.598409 | 23.41 s |
| 500 | 31000 | 1 | 0.599386 | 35.09 s |
| 500 | 31000 | 3 | 0.597920 | 35.09 s |
| 500 | 31000 | 5 | 0.605764 | 35.09 s |
| 500 | 31000 | 7 | 0.610552 | 35.09 s |
| 500 | 31000 | 9 | 0.614767 | 35.09 s |

Best measured configuration in this experiment: **500 samples/class, k=9, macro-F1=0.614767**.

The scoring time is for one validation pass that computes neighbors up to max k once and scores all candidate k values; it is not an isolated runtime per k.
This validation protocol is intentionally fixed across caps, and differs from the historical reported baseline procedure. Validation results are not test accuracy.

## Prediction-only sample outputs

| Train cap/class | k selected on validation | Image | Prediction | Regions | Ground truth |
|---:|---:|---|---|---:|---|
| 200 | 7 | `data\my_samples\sample-01.png` | `Ke1 iJC` | 6 | Ket qua (user-confirmed) |
| 200 | 7 | `data\my_samples\sample-02.png` | `h Cji0il mILiLI I` | 14 | not provided |
| 300 | 9 | `data\my_samples\sample-01.png` | `Ke1 iIU` | 6 | Ket qua (user-confirmed) |
| 300 | 9 | `data\my_samples\sample-02.png` | `I Cji0jI mIbiLI i` | 14 | not provided |
| 500 | 9 | `data\my_samples\sample-01.png` | `kcl iI0` | 6 | Ket qua (user-confirmed) |
| 500 | 9 | `data\my_samples\sample-02.png` | `I Cji0jI m1L1LI i` | 14 | not provided |
