# Data dictionary — Telco Customer Churn

Source: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (sample dataset published by IBM).
The file is not committed to this repository; see [`../data/README.md`](../data/README.md).

**Shape:** 7,043 rows × 21 columns. No duplicate rows, no duplicate `customerID`s.

## Columns

| # | Column | Type | Values / meaning | Notes |
|---|---|---|---|---|
| 0 | `customerID` | str | Unique customer identifier | **Dropped** — identifier, not a feature |
| 1 | `gender` | str | `Female`, `Male` | Binary-encoded |
| 2 | `SeniorCitizen` | int | `0`, `1` | Already binary |
| 3 | `Partner` | str | `Yes`, `No` | Binary-encoded |
| 4 | `Dependents` | str | `Yes`, `No` | Binary-encoded |
| 5 | `tenure` | int | Months the customer has stayed | Numeric |
| 6 | `PhoneService` | str | `Yes`, `No` | Binary-encoded |
| 7 | `MultipleLines` | str | `Yes`, `No`, `No phone service` | Categorical (one-hot) |
| 8 | `InternetService` | str | `DSL`, `Fiber optic`, `No` | Categorical (one-hot) |
| 9 | `OnlineSecurity` | str | `Yes`, `No`, `No internet service` | Binary-encoded after mapping `No internet service` → `No` |
| 10 | `OnlineBackup` | str | `Yes`, `No`, `No internet service` | Same treatment |
| 11 | `DeviceProtection` | str | `Yes`, `No`, `No internet service` | Same treatment |
| 12 | `TechSupport` | str | `Yes`, `No`, `No internet service` | Same treatment |
| 13 | `StreamingTV` | str | `Yes`, `No`, `No internet service` | Same treatment |
| 14 | `StreamingMovies` | str | `Yes`, `No`, `No internet service` | Same treatment |
| 15 | `Contract` | str | `Month-to-month`, `One year`, `Two year` | Categorical (one-hot) |
| 16 | `PaperlessBilling` | str | `Yes`, `No` | Binary-encoded |
| 17 | `PaymentMethod` | str | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Categorical (one-hot) |
| 18 | `MonthlyCharges` | float | Current monthly bill | Numeric |
| 19 | `TotalCharges` | str → float | Total billed to date | **Stored as text with 11 blank values**, all customers with `tenure == 0`. Converted with `pd.to_numeric(errors="coerce")` |
| 20 | `Churn` | str | `Yes`, `No` | **Target**. 1,869 Yes (26.5%) vs 5,174 No (73.5%) |

## Known traps in this dataset

1. **`TotalCharges` dtype** — it is a string column because 11 rows contain a blank space. Any arithmetic on it fails silently or raises.
2. **`"No internet service"`** — six service columns contain a third value besides Yes/No. A naive `map({'Yes': 1, 'No': 0})` turns 1,526 values into `NaN` per column.
3. **Confounded pricing** — fiber customers pay more *and* churn more, so a raw comparison of `MonthlyCharges` by churn suggests price causes churn. Within each internet service type, higher charges go with lower churn.

## Derived features (tested, not shipped)

These were evaluated in notebook 02 and **did not improve** cross-validated ROC-AUC, so the final pipeline does not use them:

| Feature | Definition |
|---|---|
| `n_addons` | Count of the six add-on services set to `Yes` |
| `autopay` | 1 if `PaymentMethod` contains "automatic" |
| `avg_monthly_spend` | `TotalCharges / tenure` (NaN when `tenure == 0`) |

## Model features (24 after encoding)

| Group | Count | Members |
|---|---|---|
| Numeric | 3 | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Binary | 12 | `SeniorCitizen`, `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, and the six add-on services |
| One-hot (with `drop="first"`) | 9 | `MultipleLines` (2), `InternetService` (2), `Contract` (2), `PaymentMethod` (3) |
