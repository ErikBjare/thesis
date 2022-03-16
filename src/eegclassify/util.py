import logging
from typing import List, Tuple, TypeVar, Generator, Iterable

import numpy as np

logger = logging.getLogger(__name__)


def unison_shuffled_copies(a, *bs):
    for b in bs:
        assert len(a) == len(b), "inputs need to be of same length"
    p = np.random.permutation(len(a))
    return a[p], *[b[p] for b in bs]


def test_unison_shuffled_copies():
    a = np.array(range(10))
    b = np.array(range(1, 11))
    c = np.array(range(2, 12))
    sa, sb, sc = unison_shuffled_copies(a, b, c)
    assert all(v1 == v2 - 1 == v3 - 2 for v1, v2, v3 in zip(sa, sb, sc))
    assert all(a == np.array(range(10))), "input array was mutated"
    assert all(c == np.array(range(2, 12))), "input array was mutated"


def powspace(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def test_powspace():
    assert np.isclose(
        powspace(0, 1, 2, 9),
        [
            0.0,
            0.016,
            0.0625,
            0.141,
            0.25,
            0.391,
            0.562,
            0.766,
            1.0,
        ],
        atol=1e-03,
    ).all()


T = TypeVar("T")


def take_until_next(ls: Iterable[T]) -> Generator[Tuple[int, int, T], None, None]:
    """
    Given an iterable with duplicate entries, chunk them together and return
    each chunk with its start and stop index.
    """
    last_v = None
    last_i = 0
    for i, v in enumerate(ls):
        if v == last_v:
            continue
        elif last_v is not None:
            yield last_i, i - 1, last_v
        last_v = v
        last_i = i
    if i != last_i:
        yield last_i, i, v


def test_take_until_next():
    ls = [1, 1, 1, 2, 3, 3]
    assert [(0, 2, 1), (3, 3, 2), (4, 5, 3)] == list(take_until_next(ls))


def aggregate_windows_to_epochs(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    subjs: np.ndarray,
    imgs: np.ndarray,
    test: np.ndarray,
    majority_vote=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate the window classification into epoch classification by taking the mean of
    the prediction probabilities, or a majority vote.

    Takes a fitted classifier, and needed data.
    """
    map_cls = {"code": 0, "prose": 1, 0: 0, 1: 1}

    if majority_vote is False and not hasattr(clf, "predict_proba"):
        logger.warning("Classifier does not support predict_proba, using majority vote")
        majority_vote = True

    if majority_vote:
        predicted = clf.predict(X)
    else:
        predicted_proba = clf.predict_proba(X)

    votes = []
    ys_epoch = []
    for i_start, i_stop, _ in take_until_next(list(zip(subjs[test], imgs[test]))):
        y_epoch = np.array([map_cls[v] for v in y[test][i_start : i_stop + 1]])

        # Check that we're not mixing classes
        assert 0 == np.std(y_epoch), np.std(y_epoch)
        ys_epoch.append(y_epoch[0])

        if majority_vote:
            # Use the majority vote
            y_preds = list(
                map(lambda v: map_cls[v], predicted[test][i_start : i_stop + 1])
            )
            vote = sum(y_preds) / len(y_preds)
        else:
            # Use the mean probability
            vote = np.mean(predicted_proba[test][i_start : i_stop + 1, 1])
        votes.append(vote > 0.5)
    return np.array(ys_epoch), np.array(votes)
