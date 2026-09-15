import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewear.data import load_raw, temporal_split
from rewear.synthetic import make_synthetic


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("hm")
    make_synthetic(d, n_customers=200, n_articles=120, n_days=60)
    return load_raw(d)


@pytest.fixture(scope="session")
def split(dataset):
    return temporal_split(dataset.transactions, val_days=7)
