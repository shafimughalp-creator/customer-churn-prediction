# Data

The Telco Customer Churn file is **not included** in this repository.

## File needed

| File | Rows | Columns |
|---|---|---|
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Kaggle name) or `Telco-Customer-Churn.csv` | 7,043 | 21 |

Place it in this `data/` folder:

```
data/
├── README.md
└── WA_Fn-UseC_-Telco-Customer-Churn.csv     (not committed)
```

The loading code accepts either filename.

## How to get it

1. **Official source (recommended):** the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset on Kaggle (published by BlastChar). A free Kaggle account is required.
2. **Or just run the notebooks.** If the CSV is missing from `data/`, each notebook downloads a public copy from IBM's GitHub repository:
   `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

## Licensing and redistribution

This is a public sample dataset published by IBM for demonstration purposes. It is not committed here because its terms belong to its publisher, not to this project's MIT licence. If you re-share the file, check the source repository's terms first.

Note that the copy used by the notebooks is downloaded at run time, so the numbers in this repository can be reproduced without redistributing the data.

## Expected schema

See [`../docs/data_dictionary.md`](../docs/data_dictionary.md).
