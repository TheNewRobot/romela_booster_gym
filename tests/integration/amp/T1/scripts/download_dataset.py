"""Download Booster T1 soccer motions from HuggingFace into data/raw/."""
import os
import subprocess
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_T1_DIR = os.path.dirname(_SCRIPT_DIR)
_CONFIG_PATH = os.path.join(_T1_DIR, "config", "config.yaml")
_OUTPUT_DIR = os.path.join(_T1_DIR, "data", "raw")

with open(_CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

base_url = cfg["dataset"]["base_url"]
files = cfg["dataset"]["files"]

os.makedirs(_OUTPUT_DIR, exist_ok=True)
print(f"Downloading Booster T1 dataset to {_OUTPUT_DIR}...")

for fname in files:
    dest = os.path.join(_OUTPUT_DIR, fname)
    if os.path.exists(dest):
        print(f"  [SKIP] {fname} (already exists)")
    else:
        print(f"  [GET]  {fname}")
        url = f"{base_url}/{fname}?download=true"
        subprocess.run(["curl", "-L", "-o", dest, url], check=True)

print(f"\nDone. Files in {_OUTPUT_DIR}:")
for fname in sorted(os.listdir(_OUTPUT_DIR)):
    if fname.endswith(".npz"):
        size_mb = os.path.getsize(os.path.join(_OUTPUT_DIR, fname)) / (1024 * 1024)
        print(f"  {fname:30s} {size_mb:.1f} MB")
