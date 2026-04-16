import os
import pandas as pd
from tqdm import tqdm

time_size = 10
window_size = f'{time_size}s'   # Time-based rolling window
over_window = time_size + 1

def hex_to_int(x, default=0):
    try:
        x = str(x).replace(" ", "").strip()
        if x == "" or x.lower() == "nan":
            return default
        return int(x, 16)
    except Exception:
        return default

def processing(df):
    steps = 8
    pbar = tqdm(total=steps, desc="  ▶ Processing steps", unit="step")

    pbar.set_description("Step 1: Mapping labels")
    df['Bus'] = df['Interface'].map({'B-CAN': 0, 'C-CAN': 1, 'P-CAN': 2})
    df.drop('Interface', axis=1, inplace=True)

    df['Label'] = df['Label'].astype(str)
    df['Class'] = df['Label'].map({
        'Normal': 0, 'DoS': 1, 'Spoofing': 1, 'Replay': 1, 'Fuzzing': 1
    }).fillna(1).astype('int32')

    df['Label'] = df['Label'].map({
        'Normal': 0, 'DoS': 1, 'Spoofing': 2, 'Replay': 3, 'Fuzzing': 4
    }).fillna(5).astype('int32')

    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    offset = df.groupby("Timestamp").cumcount().astype("float64")
    df["Timestamp"] = df["Timestamp"] + offset * 0.000003
    pbar.update(1)

    pbar.set_description("Step 2: Converting Data")
    df['Data'] = df['Data'].fillna('00').apply(hex_to_int)
    pbar.update(1)

    pbar.set_description("Step 3: Converting Arbitration_ID")
    df['Arbitration_ID'] = df['Arbitration_ID'].apply(hex_to_int)
    pbar.update(1)

    pbar.set_description("Step 4: Timestamp to datetime")
    df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('DateTime').sort_index()
    pbar.update(1)

    pbar.set_description("Step 5: Calculating intervals")
    ts_us = (df['Timestamp']*1_000_000).round()
    df["Prev_Interval"] = ts_us.diff().fillna(over_window)/1_000_000
    df["ID_Prev_Interval"] = ts_us.groupby(df["Arbitration_ID"]).diff().fillna(over_window * 1_000_000) / 1_000_000
    df["Data_Prev_Interval"] = ts_us.groupby([df["Arbitration_ID"], df["Data"]]).diff().fillna(over_window * 1_000_000) / 1_000_000
    pbar.update(1)

    pbar.set_description("Step 6: Rolling frequencies")
    df['ID_Frequency'] = (
        df.groupby('Arbitration_ID')['Arbitration_ID']
          .rolling(window_size)
          .count()
          .reset_index(level=0, drop=True)
    )
    df['Data_Frequency'] = (
        df.groupby(['Arbitration_ID', 'Data'])['Data']
          .rolling(window_size)
          .count()
          .reset_index(level=[0, 1], drop=True)
    )
    pbar.update(1)

    pbar.set_description("Step 7: Frequency diff & cleanup")
    df['Frequency_diff'] = df['ID_Frequency'] - df['Data_Frequency']
    df = df.reset_index(drop=True)
    df = df.drop(columns=['Data'], errors='ignore')
    pbar.update(1)

    pbar.set_description("Step 8: Done")
    pbar.update(1)
    pbar.close()
    return df

def main():
    program_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(program_path)
    
    path_list = [
        os.path.join(base_path, "dataset", "AutoHack2025_Dataset", "Interface", "train"),
        os.path.join(base_path, "dataset", "AutoHack2025_Dataset", "Interface", "test")
    ]

    source_path = os.path.join(program_path, "source", "AutoHack2025")
    os.makedirs(source_path, exist_ok=True)

    for path in path_list:
        file_data = {}
        csv_list = [file for file in os.listdir(path) if file.endswith("labels.csv")]

        file_name = os.path.basename(os.path.normpath(path))
        print(f"\n📂 Start processing `{file_name}` — Rolling window: {window_size}")

        with tqdm(total=len(csv_list), desc="Processing files", unit="file") as pbar:
            for file in csv_list:
                pbar.set_description(f"📄 {file}")
                file_path = os.path.join(path, file)
                df = pd.read_csv(file_path)
                processed_file = processing(df)
                file_data[file] = processed_file
                pbar.update(1)

        all_data = pd.concat(list(file_data.values()), ignore_index=True)
        output_file = os.path.join(source_path, f'{file_name}_proc.csv')
        all_data.to_csv(output_file, index=False)
        print(f"✅ Saved processed data to: {output_file}")

if __name__ == "__main__":
    main()
