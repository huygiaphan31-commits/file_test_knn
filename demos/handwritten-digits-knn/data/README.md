# EMNIST ByClass data

This local project directory contains the five EMNIST ByClass files listed
below. The dataset is not intended to be redistributed with the source code.
For a fresh setup, obtain EMNIST from the dataset page maintained by NIST:

- <https://www.nist.gov/itl/products-and-services/emnist-dataset>

Read the current dataset terms and any redistribution restrictions before
downloading, using, or sharing the data. This demo does not redistribute the
dataset and does not assert a license that has not been verified.

Keep the ByClass IDX gzip files and mapping file under `data/EMNIST/` or its
`gzip/` subdirectory:

```text
EMNIST/gzip/
  emnist-byclass-train-images-idx3-ubyte.gz
  emnist-byclass-train-labels-idx1-ubyte.gz
  emnist-byclass-test-images-idx3-ubyte.gz
  emnist-byclass-test-labels-idx1-ubyte.gz
  emnist-byclass-mapping.txt
```

Prediction-only mode needs only the train image file, train label file and
mapping. The two test files are needed only when running test evaluation. Other
EMNIST subsets and the downloaded `gzip.zip` archive are not needed by this
ByClass demo after extraction. The loader also checks `EMNIST/` and
`EMNIST/emnist-byclass/`. The mapping file must contain one
`class_id ASCII_code` pair per line for IDs 0–61. The loader reads the mapping
rather than assuming a class order. IDX headers and the 28x28 image shape are
validated before use.

Only ByClass's official train and test splits are used. Train and test are
sampled separately, deterministically, up to the configured per-class caps.
The validation subset is then selected from the sampled train data; the
official test split is not used to select `k`.
