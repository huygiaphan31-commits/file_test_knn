# Handwriting KNN diagnosis - Preset B

- Input folder: `data/my_samples/`; all seven requested files were readable.
- Model: Preset B, 500 images/class, 31,000 training samples, fixed `k=1000`.
- Validation macro-F1: `0.398249` on 620 validation samples; this is not test accuracy.
- Only EMNIST train IDX and mapping were read. Official test IDX was not opened.
- Sample images were used only for prediction and diagnostics, not training or k selection. No code/config/data changes were made.

## Results by image

| Image | Confirmed ground truth | Regions | Prediction | Region comparison |
|---|---|---:|---|---:|
| `aB3k9C5dA7.png` | `aB3K9C5dA7` | 10 | `jl jk jjj j1j` | 0/10 |
| `eight-characters.png` | `aAbBcCdD` | 9 | `jlh3cj0l0` | Not scored: segmentation count differs |
| `sample-01.png` | `Ketqua` | 6 | `Lcl ll0` | 0/6 |
| `sample-02.png` | `Not confirmed` | 16 | `l jjhIi0ji mIblLl I` | Not scored: no confirmed ground truth |
| `test_aAbB.png` | `aAbB` | 4 | `uhh3` | 0/4 |
| `test_cCdD.png` | `cCdD` | 4 | `CjjL` | 0/4 |
| `test_abcD.png` | `abcD` | 4 | `cbc D` | 3/4 |

Ground-truth sources: prior user-confirmed values recorded in project reports/manifest. `sample-02.png` remains unlabeled.

## Segmentation and preprocessing evidence

- `eight-characters.png`: confirmed truth has 8 characters but the current segmenter returned 9 boxes. The numbered image shows the lowercase `d` area split into separate narrow components before final `D`; exact per-region scoring is withheld because alignment is unsafe.
- `sample-02.png`: 16 boxes include very narrow edge/fragment boxes, including widths 2, 3, 5, 6 and 7 pixels and components at the image borders. This is direct evidence of over-segmentation/noise components; with no confirmed text, character correctness is not assessed.
- `aB3k9C5dA7.png`, `test_aAbB.png`, `test_cCdD.png`, and `sample-01.png` have the expected number of boxes. Their crops retain recognizable character shapes in raw and normalized views, but KNN predictions are mostly wrong; strongest evidence is classification error rather than a demonstrated crop-count error.
- `test_abcD.png` has four boxes and predicts `cbcD`; compared positionally with confirmed `abcD`, positions 2-4 match and the first `a` is predicted as `c`.
- The shared normalizer uses the same glyph function for EMNIST and external crops: polarity/background correction, foreground bounds, aspect-ratio resize into 20x20, centered 28x28 canvas and vector scaling. Visual artifacts do not show consistent cut-off of valid strokes; they do show that handwritten external strokes are faint/variable compared with the training glyphs. This is a distribution difference, not proof of a specific preprocessing defect.

## Nearest neighbors

Full top-10 labels and Euclidean distances for every crop are in `neighbors_top10.csv`; `regions.csv` repeats them per region. Examples:

- `test_aAbB.png` region 2 (truth `A`, prediction `h`): A@3.839, A@4.212, h@4.324, 4@4.488, t@4.669, h@4.702, h@4.756, d@4.762, I@4.765, h@4.830.
- `test_cCdD.png` region 2 (truth `C`, prediction `j`): C@4.011, C@4.051, C@4.075, C@4.093, C@4.111, C@4.200, c@4.323, L@4.350, C@4.601, 0@4.617.
- `test_abcD.png` region 1 (truth `a`, prediction `c`): a@4.509, a@4.683, a@4.689, Q@4.984, a@5.019, a@5.043, c@5.202, c@5.228, C@5.333, c@5.384.
- `sample-01.png` region 1 (truth `K`, prediction `L`): k@4.455, k@4.664, h@4.725, K@4.841, k@4.884, L@4.940, L@4.954, Y@5.019, L@5.020, L@5.041.

For example, the closest ten for `test_cCdD.png` region 2 are mostly `C`, while the fixed-k output is `j`. This is consistent with a large neighborhood vote overriding locally similar examples. The CSV contains exact nearest-neighbor distances; nearest-10 evidence alone does not show the full 1000-vote distribution.

## Mapping and code tests

- Actual EMNIST mapping and one loaded sample per class confirmed relevant IDs: `0->0`, `9->9`, `10->A`, `11->B`, `12->C`, `13->D`, `35->Z`, `36->a`, `37->b`, `38->c`, `61->z`. No mapping mismatch was found.
- `python -B -m unittest discover -s tests -t .`: **62 tests passed, 0 failed**.
- Passing unit tests verifies code behavior, not recognition quality on user images.

## Minimal next steps (no code changed)

1. Fix the specific split-lowercase-d case in `line_recognition.py` only after adding a synthetic regression test and checking detached-mark behavior.
2. Investigate filtering border/fragment components in `line_recognition.py` for `sample-02.png`; do not score it until a user confirms its text.
3. For classification errors, review fixed `k=1000` against existing validation evidence (`0.3982` at `k=1000` versus `0.5914` at `k=9` on the reported common split). Do not change preset or choose k from these user images; request authorization before code changes.
4. Do not change label mapping. Preprocessing changes are not currently justified by demonstrated crop damage; compare crops and training glyphs before proposing one.

Artifacts: `summary.csv`, `regions.csv`, `neighbors_top10.csv`, `summary.json`, and per-image masks, numbered boxes, raw crops and normalized 28x28 crops.
