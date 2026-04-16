# AutoHack Dataset Download Guide

The original dataset is too large to be hosted directly on GitHub. Please download the dataset from the Google Drive link below:

📥 **[Download AutoHack Dataset (Google Drive)](https://drive.google.com/drive/folders/1BIFxQCkWdfiiM5DpMYH6AKzPVcBSTcqL?usp=drive_link)**

## 📂 Installation Instructions

1. Click the link above and download the dataset files.
2. Extract the downloaded files (if compressed).
3. Place the `AutoHack_Dataset` folder directly into this `dataset/` directory.

When done correctly, your folder structure should look exactly like this:
```text
dataset/
    └── AutoHack_Dataset/
        ├── Interface/
        │   ├── train/
        │   │   ├── autohack_train_data_interface_with_labels.csv
        │   │   └── ...
        │   └── test/
        │       ├── autohack_test_data_interface_with_labels.csv
        │       └── ...
        └── ...
```

After placing the data here, you can return to the root directory and follow the main `README.md` to run the preprocessing and observation scripts.
