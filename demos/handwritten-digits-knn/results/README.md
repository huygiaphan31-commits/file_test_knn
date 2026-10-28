# Results

Run `main.py` to create results. No experiment outputs are included in this
directory before a run.

The program writes:

- `metrics.json`: actual run configuration, sampled counts, selected `k`,
  validation scores, accuracy, macro-F1, confusion matrix, and selected
  examples.
- `test_predictions.csv`: true and predicted character for each sampled
  official test image.
- `confusion_matrix.csv` and `confusion_matrix.png`.
- `test_examples.png`: 5–10 actual test examples, including correct/wrong
  predictions when those cases occur.
- `query_prediction.png`: only when a user supplies `--predict-image`.

Outputs are overwritten when another run uses the same results directory. Use
`--results-dir` to keep separate run folders.
