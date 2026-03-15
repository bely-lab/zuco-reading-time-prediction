import h5py
from pathlib import Path

# CHANGE THIS to your actual file path
file_path = Path("data/resultsYAC_NR.mat")

with h5py.File(file_path, "r") as f:
    print("Top-level keys:")
    for key in f.keys():
        print(" -", key)