"""Create a synthetic H&M-schema dataset so the pipeline runs without the Kaggle download.

    python scripts/make_sample.py            # writes to data/sample/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rewear.synthetic import make_synthetic

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "sample"
    make_synthetic(out)
    print(f"wrote synthetic dataset to {out}")
