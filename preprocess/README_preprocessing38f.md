# Preprocessing 38-feature Script (`preprocessing38f.py`)

## Overview
This script is responsible for extracting 38 distinct features from the raw CAN-IDS dataset. It reads the raw data, applies feature extraction logic, and saves the processed data into CSV and pickle files. The output files are then used by the subsequent training scripts.

## Features Extracted (Total 38 Features)
1. **Basic CAN Features (10):**
   - `CAN_ID`, `DLC`, `DATA_0` to `DATA_7`
2. **Per-row Payload Statistics (15):**
   - `MEAN`, `STD`, `MIN`, `MAX`, `MEDIAN`, `SKEW`, `KURT`, `P25`, `P75`, `P90`, `MAD`, `RMS`, `ZERO_COUNT`, `SUM`, `PRODUCT`
3. **Temporal & Frequency Features Formatted as Preprocessing-Style (6):**
   - `Prev_Interval`, `ID_Prev_Interval`, `Data_Prev_Interval`, `ID_Frequency`, `Data_Frequency`, `Frequency_diff`
4. **Per-ID Rolling Payload Statistics (within a 10-second window, 6):**
   - `IAT_MEAN`, `IAT_STD`, `WINDOW_MEAN`, `WINDOW_STD`, `WINDOW_MIN`, `WINDOW_MAX`
5. **Entropy (1):**
   - `PAYLOAD_ENTROPY`

## Inputs
The script expects the raw AutoHack dataset located at:
`../dataset/AutoHack_Dataset/Interface/`
- Training Data: `train/autohack_train_data_interface.csv`, `train/autohack_train_label_interface.csv`
- Testing Data: `test/autohack_test_data_interface.csv`, `test/autohack_test_label_interface.csv`

## Outputs
The processed artifacts are saved into `preprocess/source/AutoHack_38f/`:
- `train_proc_38f.csv` / `.pkl` (Processed Train Data + Label + Interface)
- `test_proc_38f.csv` / `.pkl` (Processed Test Data + Label + Interface)
- Similar output structures separated per Interface type (B-CAN, C-CAN, P-CAN).
- `feature_columns.txt` (List of output feature columns used for training).

## Execution Guide
Simply run the script using Python:
```bash
python preprocessing38f.py
```
After executing, the preprocessing is complete and you can proceed to the model training step.
