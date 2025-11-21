import glob
import os
import pathlib
import shutil
from sklearn.model_selection import train_test_split

all_files = sorted(glob.glob(r"data\BraTS2023_slices\all\*.npz"))

train_files, temp = train_test_split(all_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp, test_size=0.5, random_state=42)

for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
    split_dir = pathlib.Path(rf"data\BraTS2023_slices\{split_name}")
    split_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, split_dir / os.path.basename(f))

print("Train:", len(train_files), "Val:", len(val_files), "Test:", len(test_files))
