import numpy as np
import pandas as pd
import pytest
import humanleague as hl


def test_tabulate_counts_basic() -> None:
    population = np.array([[0, 2], [3, 5]])
    names = ["Row", "Column"]
    result = hl.tabulate_counts(population, names)

    assert len(result) == 4
    assert result.sum() == 10
    assert result.loc[0, 0] == 0
    assert result.loc[0, 1] == 2
    assert result.loc[1, 0] == 3
    assert result.loc[1, 1] == 5


def test_tabulate_counts_no_names() -> None:
    population = np.array([[5, 6], [7, 8]])
    result = hl.tabulate_counts(population)

    expected_index = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (1, 0), (1, 1)], names=None
    )
    expected_data = [5, 6, 7, 8]
    expected = pd.Series(data=expected_data, index=expected_index)

    pd.testing.assert_series_equal(result, expected)


def test_tabulate_counts_3d_array() -> None:
    population = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    names = ["Dim1", "Dim2", "Dim3"]
    result = hl.tabulate_counts(population, names)

    expected_index = pd.MultiIndex.from_tuples(
        [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ],
        names=["Dim1", "Dim2", "Dim3"],
    )
    expected_data = [1, 2, 3, 4, 5, 6, 7, 8]
    expected = pd.Series(data=expected_data, index=expected_index)

    pd.testing.assert_series_equal(result, expected)


def test_tabulate_counts_empty_array() -> None:
    population = np.array([])
    names = ["Index"]
    with pytest.raises(ValueError):
        hl.tabulate_counts(population, names)


def test_tabulate_counts_invalid_population() -> None:
    with pytest.raises(ValueError):
        hl.tabulate_counts(np.array([["a", "b"], ["c", "d"]]))


def test_tabulate_individuals_basic() -> None:
    population = np.array([[2, 1], [0, 3]])
    names = ["Category1", "Category2"]
    result = hl.tabulate_individuals(population, names)

    expected_data = [
        [0, 0],
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 1],
        [1, 1],
    ]
    expected = pd.DataFrame(data=expected_data, columns=names)

    pd.testing.assert_frame_equal(result, expected)


def test_tabulate_individuals_no_names() -> None:
    population = np.array([[1, 0], [2, 1]])
    result = hl.tabulate_individuals(population)

    expected_data = [
        [0, 0],
        [1, 0],
        [1, 0],
        [1, 1],
    ]
    expected = pd.DataFrame(data=expected_data, columns=[0, 1])

    pd.testing.assert_frame_equal(result, expected)


def test_tabulate_individuals_3d_array() -> None:
    population = np.array([[[1, 0], [0, 2]], [[0, 1], [1, 0]]])
    names = ["Dim1", "Dim2", "Dim3"]
    result = hl.tabulate_individuals(population, names)

    expected_data = [
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
    expected = pd.DataFrame(data=expected_data, columns=names)

    pd.testing.assert_frame_equal(result, expected)


def test_tabulate_individuals_empty_array() -> None:
    population = np.array([])
    names = ["Index"]
    with pytest.raises(ValueError):
        hl.tabulate_individuals(population, names)


def test_tabulate_individuals_invalid_population() -> None:
    with pytest.raises(ValueError):
        hl.tabulate_individuals(np.array([["a", "b"], ["c", "d"]]))
