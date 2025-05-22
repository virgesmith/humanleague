import numpy as np
import pytest

import humanleague as hl
from _humanleague import _unittest as hl_unittest  # type: ignore[import]


def test_version() -> None:
    assert hl.__version__


def test_unittest() -> None:
    res = hl_unittest()
    print("unit test fails/tests: ", res["nFails"], "/", res["nTests"])
    print(res["errors"])
    assert res["nFails"] == 0


def test_SobolSequence() -> None:
    s = hl.SobolSequence(3)
    a = next(s)
    assert a.shape == (3,)
    assert np.array_equal(a, [0.5, 0.5, 0.5])
    assert np.array_equal(next(s), [0.75, 0.25, 0.75])
    assert np.array_equal(next(s), [0.25, 0.75, 0.25])

    # invalid args
    with pytest.raises(ValueError):
        hl.SobolSequence(0)
    with pytest.raises(ValueError):
        hl.SobolSequence(100000)
    with pytest.raises(TypeError):
        hl.SobolSequence(1, -10)

    # skips
    s = hl.SobolSequence(1)
    s0 = hl.SobolSequence(1, 0)
    assert all(next(s0) == next(s) for _ in range(10))

    s = hl.SobolSequence(1, 1)
    assert all(
        next(s)[0] == x
        for x in [
            0.75,
            0.25,
            0.375,
            0.875,
            0.625,
            0.125,
            0.1875,
            0.6875,
            0.9375,
            0.4375,
        ]
    )

    length = 10  # -> 8 skips
    for d in range(2, 10):
        s = hl.SobolSequence(d, length)
        s0 = hl.SobolSequence(d)
        # skip s0 forward
        for _ in range(8):
            next(s0)
        for i in range(length):
            assert (next(s) == next(s0)).all()


def test_integerise() -> None:
    # pop not valid
    with pytest.raises(ValueError):
        hl.integerise(np.array([0.4, 0.3, 0.2, 0.1]), -1)

    # zero pop
    r, stats = hl.integerise(np.array([0.4, 0.3, 0.2, 0.1]), 0)
    assert stats["rmse"] == 0.0
    assert np.array_equal(r, np.array([0, 0, 0, 0]))

    # exact
    r, stats = hl.integerise(np.array([0.4, 0.3, 0.2, 0.1]), 10)
    assert stats["rmse"] < 1e-15
    assert np.array_equal(r, np.array([4, 3, 2, 1]))

    # inexact
    r, stats = hl.integerise(np.array([0.4, 0.3, 0.2, 0.1]), 17)
    assert stats["rmse"] == pytest.approx(0.273861278752583, abs=1e-6)

    assert np.array_equal(r, np.array([7, 5, 3, 2]))

    # 1-d case
    r, stats = hl.integerise(np.array([2.0, 1.5, 1.0, 0.5]))
    assert r.sum() == 5.0
    assert stats["conv"]

    # multidim integerisation
    # invalid population
    s = np.array([[1.1, 1.0], [1.0, 1.0]])
    with pytest.raises(RuntimeError):
        hl.integerise(s)
    # invalid marginals
    s = np.array([[1.1, 1.0], [0.9, 1.0]])
    with pytest.raises(RuntimeError):
        hl.integerise(s)

    # use IPF to generate a valid fractional population
    m0 = np.array([111, 112, 113, 114, 110], dtype=float)
    m1 = np.array([136, 142, 143, 139], dtype=float)
    s = np.ones([len(m0), len(m1), len(m0)])

    fpop, _ = hl.ipf(s, [[0], [1], [2]], [m0, m1, m0])

    result, stats = hl.integerise(fpop)
    assert stats["conv"]
    assert np.sum(result) == sum(m0)
    assert stats["rmse"] < 1.05717


def test_IPF() -> None:
    m0 = np.array([52.0, 48.0])
    m1 = np.array([87.0, 13.0])
    m2 = np.array([55.0, 45.0])
    i = [0, 1]

    s = np.ones([len(m0), len(m1)])
    p, stats = hl.ipf(s, i, [m0, m1])
    assert stats["conv"]
    assert stats["pop"] == 100.0
    assert np.array_equal(p, np.array([[45.24, 6.76], [41.76, 6.24]]))

    s[0, 0] = 0.7
    p, stats = hl.ipf(s, i, [m0, m1])
    assert stats["conv"]
    # check overall population and marginals correct
    assert np.sum(p) == stats["pop"]
    assert np.allclose(np.sum(p, 0), m1)
    assert np.allclose(np.sum(p, 1), m0)

    # mix list and tuple
    im = ((0,), (1,), [2])
    s = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    p, stats = hl.ipf(s, im, (m0, m1, m2))
    assert stats["conv"]
    # check overall population and marginals correct
    assert np.sum(p) == pytest.approx(stats["pop"], 1e-8)
    assert np.allclose(np.sum(p, (0, 1)), m2)
    assert np.allclose(np.sum(p, (1, 2)), m0)
    assert np.allclose(np.sum(p, (2, 0)), m1)

    # 12D
    s = np.ones([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    i = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]
    m = np.array([2048.0, 2048.0])
    p, stats = hl.ipf(s, i, [m, m, m, m, m, m, m, m, m, m, m, m])
    assert stats["pop"] == 4096

    m0 = np.array([52.0, 48.0])
    m1 = np.array([87.0, 13.0])
    m2 = np.array([55.0, 45.0])

    seed = np.ones([len(m0), len(m1)])
    p, stats = hl.ipf(seed, [[0], [1]], [m0, m1])
    assert np.allclose(np.sum(p, (0)), m1)
    assert np.allclose(np.sum(p, (1)), m0)
    assert stats["conv"]
    assert stats["iterations"] == 1
    assert stats["maxError"] == 0.0
    assert stats["pop"] == 100.0
    assert np.array_equal(p, np.array([[45.24, 6.76], [41.76, 6.24]]))

    seed[0, 1] = 0.7
    p, stats = hl.ipf(seed, [[0], [1]], [m0, m1])
    assert np.allclose(np.sum(p, (0)), m1)
    assert np.allclose(np.sum(p, (1)), m0)
    assert stats["conv"]
    assert stats["iterations"] < 6
    assert stats["maxError"] < 5e-10
    assert stats["pop"] == 100.0

    s = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    p, stats = hl.ipf(s, [[0], [1], [2]], [m0, m1, m2])
    assert stats["conv"]
    # check overall population and marginals correct
    assert np.sum(p) == pytest.approx(stats["pop"], 1e-8)
    assert np.allclose(np.sum(p, (0, 1)), m2)
    assert np.allclose(np.sum(p, (1, 2)), m0)
    assert np.allclose(np.sum(p, (2, 0)), m1)

    # 12D
    s = np.ones([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    m = np.array([2048.0, 2048.0])
    _, stats = hl.ipf(s, i, [m, m, m, m, m, m, m, m, m, m, m, m])
    assert stats["conv"]
    assert stats["pop"] == 4096

    # IPF with fractional values (e.g. probabilities)
    seed = np.full((2,2,2), 1/8)
    indices = [(0, 1), (0, 2), (1, 2)]
    marginals = [
        np.array([[0.92625, 0.02375], [0.02375, 0.02625]]),
        np.array([[0.92625, 0.02375], [0.02375, 0.02625]]),
        np.array([[0.92625, 0.02375], [0.02375, 0.02625]])
    ]

    pop, stats = hl.ipf(seed, indices, marginals)
    assert pop.sum() == 1.0
    assert stats["conv"]


@pytest.mark.filterwarnings("ignore:humanleague.flatten is deprecated:UserWarning")
def test_QIS() -> None:
    m0 = np.array([52, 48])
    m1 = np.array([10, 77, 13])

    p, stats = hl.qis([0, 1], [m0, m1])
    assert stats["conv"]
    assert stats["chiSq"] < 0.04
    assert stats["pValue"] > 0.9
    assert stats["pop"] == 100.0
    assert np.allclose(np.sum(p, 0), m1)
    assert np.allclose(np.sum(p, 1), m0)

    m0 = np.array([52, 40, 4, 4])
    m1 = np.array([87, 10, 3])
    m2 = np.array([55, 15, 6, 12, 12])

    # tuples, scalar indices
    p, stats = hl.qis((0, 1, 2), (m0, m1, m2))
    assert stats["conv"]
    assert stats["chiSq"] < 73.0  # TODO seems a bit high (probably )
    assert stats["pValue"] > 0.0  # TODO this is suspect
    assert stats["pop"] == 100.0
    assert np.allclose(np.sum(p, (0, 1)), m2)
    assert np.allclose(np.sum(p, (1, 2)), m0)
    assert np.allclose(np.sum(p, (2, 0)), m1)

    # Test flatten functionality
    table = hl.flatten(p)

    # length is no of dims
    assert len(table) == 3
    # length of element is pop
    assert len(table[0]) == stats["pop"]
    # check consistent with marginals
    for i, mi in enumerate(m0):
        assert table[0].count(i) == mi
    for i, mi in enumerate(m1):
        assert table[1].count(i) == mi
    for i, mi in enumerate(m2):
        assert table[2].count(i) == mi

    m0 = np.array([52, 48])
    m1 = np.array([87, 13])
    m2 = np.array([67, 33])
    m3 = np.array([55, 45])
    idx = [[0], [1], [2], [3]]

    p, stats = hl.qis(idx, [m0, m1, m2, m3])
    assert stats["conv"]
    assert stats["chiSq"] < 10
    assert stats["pValue"] > 0.001
    assert stats["pop"] == 100
    assert np.allclose(np.sum(p, (0, 1, 2)), m3)
    assert np.allclose(np.sum(p, (1, 2, 3)), m0)
    assert np.allclose(np.sum(p, (2, 3, 0)), m1)
    assert np.allclose(np.sum(p, (3, 0, 1)), m2)

    m = np.array([[10, 20, 10], [10, 10, 20], [20, 10, 10]])
    idx = [[0, 1], [1, 2]]
    p, stats = hl.qis(idx, [m, m])
    assert stats["conv"]
    assert stats["chiSq"] < 10
    assert stats["pValue"] > 0.27
    assert stats["pop"] == 120
    assert np.allclose(np.sum(p, 2), m)
    assert np.allclose(np.sum(p, 0), m)


def test_QIS_dim_indexing() -> None:
    # tricky array indexing - 1st dimension of d0 already sampled, remaining dimension
    # indices on slice of d0 need to be remapped

    m0 = np.ones([4, 6, 4, 4], dtype=int)
    m1 = np.ones([4, 4, 4], dtype=int) * 6

    _, ms = hl.qis([[0, 1, 2, 3], [0, 4, 5]], [m0, m1])
    assert ms["conv"]

    _, ms = hl.qis([[0, 4, 5], [0, 1, 2, 3]], [m1, m0])
    assert ms["conv"]

    _, ms = hl.qis([[0, 1, 2], [0, 3, 4, 5]], [m1, m0])
    assert ms["conv"]


def test_QISI() -> None:
    m0 = np.array([52, 48])
    m1 = np.array([10, 77, 13])
    idx = [0, 1]
    s = np.ones([len(m0), len(m1)])

    p, stats = hl.qisi(s, idx, [m0, m1])
    assert stats["conv"]
    assert stats["chiSq"] < 0.04
    assert stats["pValue"] > 0.9
    assert stats["pop"] == 100.0
    assert np.allclose(np.sum(p, 0), m1)
    assert np.allclose(np.sum(p, 1), m0)

    m0 = np.array([52, 40, 4, 4])
    m1 = np.array([87, 10, 3])
    m2 = np.array([55, 15, 6, 12, 12])
    idx_m = ((0,), [1], (2,))
    s = np.ones((len(m0), len(m1), len(m2)))

    p, stats = hl.qisi(s, idx_m, (m0, m1, m2))
    assert stats["conv"]
    assert stats["chiSq"] < 70  # seems a bit high
    assert stats["pValue"] > 0.0  # seems a bit low
    assert stats["pop"] == 100.0
    assert np.allclose(np.sum(p, (0, 1)), m2)
    assert np.allclose(np.sum(p, (1, 2)), m0)
    assert np.allclose(np.sum(p, (2, 0)), m1)

    m0 = np.array([52, 48])
    m1 = np.array([87, 13])
    m2 = np.array([67, 33])
    m3 = np.array([55, 45])
    idx = [[0], [1], [2], [3]]
    s = np.ones([len(m0), len(m1), len(m2), len(m3)])

    p, stats = hl.qisi(s, idx, [m0, m1, m2, m3])
    assert stats["conv"]
    assert stats["chiSq"] < 5.5
    assert stats["pValue"] > 0.02
    assert stats["pop"] == 100.0
    assert np.allclose(np.sum(p, (0, 1, 2)), m3)
    assert np.allclose(np.sum(p, (1, 2, 3)), m0)
    assert np.allclose(np.sum(p, (2, 3, 0)), m1)
    assert np.allclose(np.sum(p, (3, 0, 1)), m2)

    # check dimension consistency check works
    s = np.ones([2, 3, 7, 5])
    m1 = np.ones([2, 3], dtype=int) * 5 * 7
    m2 = np.ones([3, 5], dtype=int) * 7 * 2
    m3 = np.ones([5, 7], dtype=int) * 2 * 3
    with pytest.raises(RuntimeError):
        hl.qisi(s, [[0, 1], [1, 2], [2, 3]], [m1, m2, m3])
    with pytest.raises(RuntimeError):
        hl.ipf(
            s,
            [[0, 1], [1, 2], [2, 3]],
            [m1.astype(float), m2.astype(float), m3.astype(float)],
        )

    s = np.ones((2, 3, 5))
    with pytest.raises(RuntimeError):
        hl.qisi(s, [[0, 1], [1, 2], [2, 3]], [m1, m2, m3])
    with pytest.raises(RuntimeError):
        hl.ipf(
            s,
            [[0, 1], [1, 2], [2, 3]],
            [m1.astype(float), m2.astype(float), m3.astype(float)],
        )
