"""
Testing for the tree module.
Adopted from scikit-learn.
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/tests/test_tree.py
"""

from itertools import product

import pytest
import numpy as np
from honest_forests import HonestTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets


# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, 2])
def test_iris(criterion, max_features):
    # Check consistency on dataset iris.
    clf = HonestTreeClassifier(
        criterion=criterion, random_state=0, max_features=max_features
    )
    clf.fit(iris.data, iris.target)
    score = accuracy_score(clf.predict(iris.data), iris.target)
    assert (
        score > 0.5 and score < 1.0
    ), "Failed with {0}, criterion = {1} and score = {2}".format(
        "HonestTree", criterion, score
    )

    score = accuracy_score(
        clf.predict(iris.data), clf.predict_proba(iris.data).argmax(1)
    )
    assert score == 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
        "HonestTree", criterion, score
    )


def test_toy_accuracy():
    clf = HonestTreeClassifier()
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10
    clf = clf.fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)


@pytest.mark.parametrize(
    "honest_prior, val",
    [
        ("uniform", 0.5),
        ("empirical", 0.75),
        ("ignore", np.nan),
    ],
)
def test_impute_posteriors(honest_prior, val):
    np.random.seed(0)
    X = np.random.normal(0, 1, (100, 2))
    y = [0] * 75 + [1] * 25
    clf = HonestTreeClassifier(
        honest_fraction=0.02, random_state=0, honest_prior=honest_prior
    )
    clf = clf.fit(X, y)

    y_proba = clf.predict_proba(X)
    if np.isnan(val):
        assert (
            len(np.where(np.isnan(y_proba[:, 0]))[0]) > 50
        ), f"Failed with {honest_prior}"
    else:
        assert (
            len(np.where(y_proba[:, 0] == val)[0]) > 50
        ), f"Failed with {honest_prior}"


def test_increasing_leaves():
    np.random.seed(0)
    X = np.random.normal(0, 1, (200, 2))
    y = [0] * 100 + [1] * 100

    n_leaves = []
    for hf in [0.9, 0.7, 0.4, 0.2]:
        clf = HonestTreeClassifier(honest_fraction=hf, random_state=0)
        clf = clf.fit(X, y)
        n_leaves.append(clf.get_n_leaves())

    assert np.all(np.diff(n_leaves) > 0)


def test_impute_classes():
    np.random.seed(0)
    X = np.random.normal(0, 1, (101, 2))
    y = [2] * 50 + [1] * 50 + [0]
    clf = HonestTreeClassifier(honest_fraction=0.02, random_state=0)
    clf = clf.fit(X, y)

    y_proba = clf.predict_proba(X)

    assert y_proba.shape[1] == 3
