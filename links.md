# Dataset source links

Canonical sources for the 15 datasets registered in [`config/datasets.yaml`](config/datasets.yaml).
Local files (when used) live under `SYNTH_BENCHMARK/SYNTH/Datasets/` (see `REPO_ROOT` in `config_utils.py`).

## Phase 1 — Classification

| # | Key                | Display name   | Source in config                            | Primary link                                                                                                                | Notes                                                                                            |
| - | ------------------ | -------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1 | `cancer`         | Cancer         | local CSV`Datasets/Cancer.csv`            | [Breast Cancer Wisconsin (Diagnostic) — UCI #17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | DOI:[10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Local copy of WDBC (`diagnosis` = M/B). |
| 2 | `alzheimers`     | Alzhimers      | local Excel`Datasets/Alzhimers.xlsx`      | [OASIS Longitudinal MRI (Kaggle mirror)](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers)                         | Also:[OASIS brains](https://www.oasis-brains.org/). Longitudinal MRI demographics / CDR groups.   |
| 3 | `adult`          | Adult          | local CSV`Datasets/adult_census.csv`      | [Adult — UCI #2](https://archive.ics.uci.edu/dataset/2/adult)                                                               | DOI:[10.24432/C5XW20](https://doi.org/10.24432/C5XW20). Census income prediction.                 |
| 4 | `forest_cover`   | Forest Cover   | UCI`uci_id: 31`                           | [Covertype — UCI #31](https://archive.ics.uci.edu/dataset/31/covertype)                                                     | DOI:[10.24432/C50K5N](https://doi.org/10.24432/C50K5N).                                           |
| 5 | `bank_marketing` | Bank Marketing | local CSV`Datasets/bank-full.csv`         | [Bank Marketing — UCI #222](https://archive.ics.uci.edu/dataset/222/bank+marketing)                                         | DOI:[10.24432/C5K306](https://doi.org/10.24432/C5K306). Semicolon-separated `bank-full.csv`.    |
| 6 | `wine_quality`   | Wine Quality   | local CSV`Datasets/winequality-white.csv` | [Wine Quality — UCI #186](https://archive.ics.uci.edu/dataset/186/wine+quality)                                             | DOI:[10.24432/C56S3T](https://doi.org/10.24432/C56S3T). White wine subset used here.              |
| 7 | `cdc_diabetes`   | CDC Diabetes   | UCI`uci_id: 891`                          | [CDC Diabetes Health Indicators — UCI #891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)         | DOI:[10.24432/C53919](https://doi.org/10.24432/C53919).                                           |
| 8 | `mushroom`       | Mushroom       | UCI`uci_id: 73`                           | [Mushroom — UCI #73](https://archive.ics.uci.edu/dataset/73/mushroom)                                                       | DOI:[10.24432/C5959T](https://doi.org/10.24432/C5959T).                                           |
| 9 | `magic_gamma`    | MAGIC Gamma    | UCI`uci_id: 159`                          | [MAGIC Gamma Telescope — UCI #159](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)                           | DOI:[10.24432/C52C8B](https://doi.org/10.24432/C52C8B).                                           |

## Phase 2 — Regression

| #  | Key                   | Display name      | Source in config                 | Primary link                                                                                                                    | Notes                                                                                                            |
| -- | --------------------- | ----------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 10 | `metro_interstate`  | Metro Interstate  | UCI`uci_id: 492`               | [Metro Interstate Traffic Volume — UCI #492](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)           | DOI:[10.24432/C5X60B](https://doi.org/10.24432/C5X60B). HPO drops `date_time`.                                  |
| 11 | `online_shopping`   | Online Shopping   | local CSV`Datasets/e-shop.csv` | [Clickstream Data for Online Shopping — UCI #553](https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping) | DOI:[10.24432/C5QK7X](https://doi.org/10.24432/C5QK7X). HPO drops `session ID` and `page 2 (clothing model)`. |
| 12 | `air_quality`       | Air Quality       | UCI`uci_id: 360`               | [Air Quality — UCI #360](https://archive.ics.uci.edu/dataset/360/air+quality)                                                   | DOI:[10.24432/C59K5F](https://doi.org/10.24432/C59K5F). HPO drops `Date`, `Time`.                                  |
| 13 | `concrete`          | Concrete          | UCI`uci_id: 165`               | [Concrete Compressive Strength — UCI #165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)               | DOI:[10.24432/C5PK67](https://doi.org/10.24432/C5PK67).                                                           |
| 14 | `energy_efficiency` | Energy Efficiency | UCI`uci_id: 242`               | [Energy Efficiency — UCI #242](https://archive.ics.uci.edu/dataset/242/energy+efficiency)                                       | DOI:[10.24432/C51307](https://doi.org/10.24432/C51307). Target used: `Y1`.                                      |
| 15 | `real_estate`       | Real Estate       | UCI`uci_id: 477`               | [Real Estate Valuation — UCI #477](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)                      | DOI:[10.24432/C5J30W](https://doi.org/10.24432/C5J30W). HPO drops `X1 transaction date`. Target: `Y house price of unit area`. |

## Quick copy-paste URLs

```text
# Classification
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
https://archive.ics.uci.edu/dataset/2/adult
https://archive.ics.uci.edu/dataset/31/covertype
https://archive.ics.uci.edu/dataset/222/bank+marketing
https://archive.ics.uci.edu/dataset/186/wine+quality
https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
https://archive.ics.uci.edu/dataset/73/mushroom
https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

# Regression
https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping
https://archive.ics.uci.edu/dataset/360/air+quality
https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
https://archive.ics.uci.edu/dataset/242/energy+efficiency
https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set
```

## Loading notes

- **UCI entries** (`source: uci`): loaded via [`ucimlrepo`](https://github.com/uci-ml-repo/ucimlrepo) with the `uci_id` in YAML.
- **Local CSV/Excel entries**: paths are relative to `REPO_ROOT` (`/home/gopi.battineni/SYNTH_BENCHMARK/SYNTH`). Prefer the UCI/Kaggle links above when re-downloading.
- UCI hub: https://archive.ics.uci.edu/
