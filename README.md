# Climate Change — Earth Surface Temperature Data

This project explores historical Earth surface temperature records and applies **machine learning** and **deep learning** to analyze trends and model temperature anomalies across multiple geographic scales.

## Repository contents

```text
.
├── archive/
│   ├── GlobalTemperatures.csv
│   ├── GlobalLandTemperaturesByCountry.csv
│   ├── GlobalLandTemperaturesByState.csv
│   ├── GlobalLandTemperaturesByMajorCity.csv
│   └── GlobalLandTemperaturesByCity.csv
├── ML DL.ipynb
├── Deep_Learning.ipynb
├── ml_dl_shared.py
├── Requirement.txt
└── .tmp_dl_images/                # images extracted from notebook outputs (optional)
```

### Notebooks

- **`ML DL.ipynb`**
  - Exploratory data analysis and classical ML exercises on the multi-scale dataset.
  - Uses **Plotly** for interactive / animated visualizations.
  - **Important:** you must set `DATA_DIR` (see Quickstart).

- **`Deep_Learning.ipynb`**
  - Trains a **sequence model (LSTM)** to forecast **seasonal city temperature anomalies**.
  - Uses `ml_dl_shared.py` for consistent preprocessing and dataset construction.
  - Runs on CPU by default; will use CUDA if PyTorch detects a GPU.

## Dataset

All dataset files are already included in this repo under `archive/`.

Notes:
- `GlobalLandTemperaturesByCity.csv` is very large; loading it can require several GB of RAM.
- The raw data includes missing temperatures and uncertainty fields; the preprocessing code drops rows that cannot be used for modeling.

## Quickstart

### 1) Create an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies

This repo includes a minimal `Requirement.txt`:

```bash
pip install -r Requirement.txt
```

The notebooks also use additional packages that are not listed there (notably **PyTorch** and **Plotly**). Install them if needed:

```bash
pip install torch plotly
```

### 3) Start Jupyter

```bash
jupyter lab
# or: jupyter notebook
```

Open either notebook:
- `ML DL.ipynb`
- `Deep_Learning.ipynb`

## Running the notebooks

### `ML DL.ipynb`: set `DATA_DIR`

At the top of the notebook, define `DATA_DIR` to point to the `archive/` folder.

If you run the notebook from the repository root, this usually works:

```python
DATA_DIR = "archive"
```

If you run the notebook from another working directory, set an absolute path instead.

### `Deep_Learning.ipynb`: preprocessing is automatic

`Deep_Learning.ipynb` imports preprocessed tables directly:
- `city_seasonal_clean` (seasonal city aggregates + anomalies)
- `dl_df` (model-ready table)

These are built by `ml_dl_shared.py`, which loads the CSVs from `archive/` automatically.

## How preprocessing works (ml_dl_shared.py)

The shared preprocessing module builds three main tables:

- `city_clean` — monthly city-level records with numeric latitude/longitude
- `city_seasonal_clean` — seasonal aggregates per city/country/year plus:
  - `TempAnomaly`: deviation from a baseline climatology (1951–1980)
  - `sigma_dev`: anomaly magnitude scaled by uncertainty
- `dl_df` — a modeling dataset with:
  - seasonal identifiers (`season_id`, `season_sin`, `season_cos`)
  - normalized time (`year_norm`, `t_index`)
  - target column: `TARGET_COL = "temp_anomaly"`

To keep the DL dataset computationally manageable, the module caps the number of city series per country (`PER_COUNTRY_CAP = 6`).

## Troubleshooting

- **`NameError: DATA_DIR is not defined` (ML notebook)**
  - Set `DATA_DIR = "archive"` near the top of `ML DL.ipynb`.

- **`ModuleNotFoundError: No module named 'torch'`**
  - Install PyTorch: `pip install torch` (or follow the official PyTorch install command for your platform).

- **`ModuleNotFoundError: No module named 'plotly'`**
  - Install Plotly: `pip install plotly`.

- **Slow / memory-heavy loads**
  - `GlobalLandTemperaturesByCity.csv` is extremely large. Consider filtering to a subset of cities/countries/years before plotting or modeling.

## Credits

Notebook authors (as listed in `ML DL.ipynb`): **Dalia, Jeancy, Sayqin**.

Dataset: Berkeley Earth surface temperature records, commonly distributed via Kaggle as “Climate Change: Earth Surface Temperature Data”.
