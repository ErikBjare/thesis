import numpy as np
from typing import List, Tuple, TypeVar, Generator, Iterable


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
    assert all([v1 == v2 - 1 == v3 - 2 for v1, v2, v3 in zip(sa, sb, sc)])
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
