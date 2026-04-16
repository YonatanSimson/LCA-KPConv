"""Tests for empty / no-neighbor queries: padding fallback and label -1 (ignore)."""

from unittest import mock

import numpy as np
import pytest
from datasets import common


def test_degenerate_neighbor_mask_all_padding():
    n_support = 5
    neighbors = np.full((4, 3), n_support, dtype=np.int32)
    m = common.degenerate_neighbor_mask(neighbors, n_support)
    assert m.shape == (4,)
    assert m.all()


def test_degenerate_neighbor_mask_mixed():
    n_support = 5
    neighbors = np.array(
        [
            [0, n_support, n_support],
            [n_support, n_support, n_support],
        ],
        dtype=np.int32,
    )
    m = common.degenerate_neighbor_mask(neighbors, n_support)
    assert not m[0]
    assert m[1]


def test_degenerate_neighbor_mask_empty_second_dim():
    neighbors = np.zeros((3, 0), dtype=np.int32)
    m = common.degenerate_neighbor_mask(neighbors, 5)
    assert m.shape == (3,)
    assert not m.any()


def test_mark_degenerate_query_labels_inplace():
    labels = np.array([1, 2, 3, 4], dtype=np.int32)
    n_support = 2
    neighbors = np.array(
        [
            [0, 1],
            [n_support, n_support],
            [n_support, n_support],
            [1, 0],
        ],
        dtype=np.int32,
    )
    common.mark_degenerate_query_labels(labels, neighbors, n_support)
    np.testing.assert_array_equal(labels, np.array([1, -1, -1, 4], dtype=np.int32))


def test_batch_neighbors_no_in_radius_neighbors_returns_padding_sentinel():
    """C++ returns one column of Ns when every query has zero in-radius neighbors (rebuild cpp_neighbors after wrapper change)."""
    q = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
    s = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    bq = np.array([1], dtype=np.int32)
    bs = np.array([1], dtype=np.int32)
    out = common.batch_neighbors(q, s, bq, bs, radius=0.01)
    assert out.shape == (1, 1)
    np.testing.assert_array_equal(out, np.array([[1]], dtype=np.int32))


def test_batch_neighbors_zero_queries_no_cpp_call():
    with mock.patch.object(common, "cpp_neighbors") as mock_cpp:
        out = common.batch_neighbors(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.array([1], dtype=np.int32),
            radius=0.1,
        )
        mock_cpp.batch_query.assert_not_called()
    assert out.shape == (0, 1)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_mark_degenerate_respects_minus_one_dtype(dtype):
    labels = np.array([0, 1, 2], dtype=dtype)
    neighbors = np.full((3, 1), 3, dtype=np.int32)
    common.mark_degenerate_query_labels(labels, neighbors, n_support=3)
    assert labels.dtype == dtype
    assert (labels == -1).all()


def test_mark_degenerate_query_labels_rejects_non_ndarray():
    with pytest.raises(TypeError, match="numpy.ndarray"):
        common.mark_degenerate_query_labels(
            [0, 1, 2],
            np.zeros((3, 1), dtype=np.int32),
            n_support=1,
        )


def test_mark_degenerate_query_labels_rejects_readonly():
    labels = np.array([0, 1, 2], dtype=np.int32)
    labels.setflags(write=False)
    neighbors = np.full((3, 1), 3, dtype=np.int32)
    with pytest.raises(ValueError, match="writeable"):
        common.mark_degenerate_query_labels(labels, neighbors, n_support=3)
