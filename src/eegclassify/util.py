import numpy as np


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
