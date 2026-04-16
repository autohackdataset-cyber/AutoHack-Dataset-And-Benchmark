# AutoHack-Dataset-And-Benchmark

This repository contains the preprocessing pipelines and baseline Intrusion Detection System (IDS) evaluation scripts for the AutoHack2025 CAN-IDS dataset. The scripts are fully configured to be run out-of-the-box by cloning the repository and running the provided commands.

## 📂 Project Structure

```text
.
├── dataset/
│   └── AutoHack2025_Dataset/    # Place the raw dataset here
│       ├── Interface/
│       │   ├── train/                           # Raw train CSVs
│       │   └── test/                            # Raw test CSVs
│       └── ...
├── preprocess/
│   ├── preprocess.py                            # Basic chronological/frequency feature extractor
│   ├── preprocessing38f.py                      # 38-feature complex extractor (stats, rolling, entropy)
│   ├── README_preprocess.md                     # Details about basic preprocess
│   └── README_preprocessing38f.md               # Details about 38f preprocess
├── ids_code/
│   ├── observation1.py                          # Obs 1: Per-attack performance of baseline RF
│   ├── observation2.py                          # Obs 2: Non-periodic UDS false-positive analysis
│   └── observation3.py                          # Obs 3: 38-feature RF/XGBoost evaluation per-interface
├── requirements.txt                             # Python dependencies
└── README.md                                    # This file
```

*(Note: The `Result/` and `preprocess/source/` output folders will be automatically generated upon executing the scripts.)*

---

## 🛠️ Prerequisites & Installation

This project requires **Python 3.8+**. It is highly recommended to use a virtual environment (e.g., `venv` or `conda`).

1. **Clone or download** this repository.
2. **Download the dataset** following the instructions provided in [dataset/AutoHack2025_Dataset.md](dataset/AutoHack2025_Dataset.md) and ensure it is properly placed as shown in the structure above.
3. **Install the required packages** using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start / Usage Guide

To reproduce the methodology and evaluation entirely, simply follow the sequence below sequentially from the root directory of this repository.

### Step 1: Data Preprocessing
The model scripts rely on pre-processed data structures. First, extract the baseline sequences and the 38-feature sets:

```bash
# 1. Run basic feature processing
python preprocess/preprocess.py

# 2. Run 38-feature processing
python preprocess/preprocessing38f.py
```
*These steps will parse the raw dataset and extract time-windows, intervals, and payload statistics, saving intermediate datasets into `preprocess/source/`.*

### Step 2: Training & Observations
Run the observation scripts that evaluate the models based on the preprocessed results.

```bash
# Baseline IDS (Random Forest) Full Evaluation
python ids_code/observation1.py

# UDS False-Positive Analysis
python ids_code/observation2.py

# Comprehensive Per-Interface 38-feature Evaluation (RF & XGBoost)
python ids_code/observation3.py
```

---

## 📊 Outputs & Results

After successfully running the execution steps, check the generated results in the `Result/` directory:
- **`Result/observation1/`**: Contains `f1_auc_result.txt` (metrics) and confusion matrix plots.
- **`Result/observation2/`**: Contains metrics and plots regarding the UDS false-positive analysis.
- **`Result/observation3/`**: Contains detailed text summaries and visual confusion matrices for RF and XGBoost combinations per CAN interface (B-CAN, C-CAN, P-CAN).
