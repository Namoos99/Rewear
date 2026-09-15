import pandas as pd

from rewear.metrics import apk, catalog_coverage, long_tail_exposure, mapk


def test_apk_perfect_and_empty():
    assert apk(["a", "b"], ["a", "b", "c"], k=3) == 1.0
    assert apk([], ["a"], k=3) == 0.0
    assert apk(["z"], ["a", "b"], k=3) == 0.0


def test_apk_matches_kaggle_reference():
    # Known value: hit at rank 1 and rank 3 of 3 actual -> (1/1 + 2/3)/3
    assert abs(apk(["a", "b", "c"], ["a", "x", "b"], k=12) - (1 + 2 / 3) / 3) < 1e-9


def test_mapk_penalises_missing_users():
    truth = {"u1": ["a"], "u2": ["b"]}
    assert mapk(truth, {"u1": ["a"]}) == 0.5


def test_coverage_and_long_tail():
    preds = {"u1": ["a", "b"], "u2": ["a", "c"]}
    assert catalog_coverage(preds, n_catalog=10, k=12) == 0.3
    train = pd.DataFrame({"article_id": ["a"] * 10 + ["b"] * 2 + ["c"] + ["d"] * 5 + ["e"]})
    # head_fraction=0.2 of 5 items -> 1 head item ("a"); 2 of 4 slots are "a"
    assert long_tail_exposure(preds, train, head_fraction=0.2, k=12) == 0.5
