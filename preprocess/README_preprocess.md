# Preprocessing Script (`preprocess.py`)

## Overview
This script performs foundational data preprocessing on the raw `.csv` CAN-IDS datasets. It focuses on mapping string labels to numerical formats, calculating intervals between CAN IDs or Data bytes, and generating specific tracking characteristics such as frequency counts within a specific time window (`10s`). 

## Key Processing Steps
1. **Label Mapping**: Converts `Interface` values (B-CAN, C-CAN, P-CAN) to integers. Maps sub-labels like Normal, DoS, Spoofing, Replay, Fuzzing into classification integers.
2. **Data & Arbitration ID Conversion**: Cleans up and parses hexadecimal payload vectors (`Data`) and addresses (`Arbitration_ID`) into numerical formats.
3. **Timestamp Alignment**: Modifies basic integer timestamps with an offset to act strictly and smoothly over consecutive logs and converts them into Pandas datetime indices for rolling windows.
4. **Interval Calculations**: Measures time differences since the last occurrence overall, across IDs (`ID_Prev_Interval`), and precise payload occurrences (`Data_Prev_Interval`).
5. **Rolling Frequencies**: Uses the specified `--window_size` (e.g., `10s`) to calculate how frequently distinct combinations (`Arbitration_ID` / `Data`) occur inside the rolling window.

## Workflow Integration
- **Input path**: Reads CSV files inside the dataset specifically using `Interface` paths -> `dataset/.../Interface/train` and `.../test`.
- **Output path**: Writes aggregated and cleaned datasets as `<subset>_proc.csv` to `preprocess/source/AutoHack`.

## Execution Guide
Simply run the script within your chosen environment:
```bash
python preprocess.py
```
*(The scripts use relative directory structuring internally pointing to your root `/dataset/` folder, ensuring seamless replication paths).*
