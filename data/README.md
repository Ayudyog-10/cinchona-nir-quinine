# Data

- `example_scans.csv` – 48 scans from 3 samples (demo input format for the app/CLI). These samples were part of the training set, so their predictions are not an independent test.
- Place the full workbook `whole_Cinchona_Combine_Data_wih_Reference.xlsx` here to retrain or benchmark. It is excluded from Git by `.gitignore`.

Workbook layout (row positions define samples; no ID column in the file):

| Campaign | Excel rows | Samples × scans | Collected |
|---|---|---|---|
| A | 2–73 | 6 × 12 | 2022–23 |
| B | 74–1849 | 111 × 16 | 2022–23 |
| C | 1850–4265 | 151 × 16 | Sept 2024 |

Column 1 = HPLC quinine reference; remaining 256 columns = absorbance at 892–1710 nm.
