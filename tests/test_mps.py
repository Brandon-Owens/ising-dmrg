"""Checks for MPS construction and reconstruction."""

import numpy as np
import pytest

from ising_dmrg.mps import mps_to_state, state_to_left_mps


def test_reconstruct_known_product_state():
    """Check basis order and complex coefficients independently of SVD."""
    a = np.array([1, 1j]) / np.sqrt(2)
    b = np.array([0, 1])
    c = np.array([1, 0])

    tensors = [
        a.reshape(1, 2, 1),
        b.reshape(1, 2, 1),
        c.reshape(1, 2, 1),
    ]
    expected = np.kron(np.kron(a, b), c)

    np.testing.assert_allclose(
        mps_to_state(tensors), expected, atol=1e-12
    )


@pytest.mark.parametrize("n", [2, 3, 5])
def test_exact_round_trip(n):
    """An untruncated factorization preserves amplitudes and scale."""
    rng = np.random.default_rng(100 + n)
    state = (
        rng.normal(size=2**n)
        + 1j * rng.normal(size=2**n)
    )

    tensors = state_to_left_mps(state)
    reconstructed = mps_to_state(tensors)

    np.testing.assert_allclose(
        reconstructed, state, atol=1e-11
    )


def test_bond_one_truncation():
    """The biased state loses its smaller Schmidt term."""
    state = np.array(
        [np.sqrt(0.9), 0, 0, 1j * np.sqrt(0.1)]
    )

    tensors = state_to_left_mps(state, max_bond=1)
    reconstructed = mps_to_state(tensors)
    expected = np.array([np.sqrt(0.9), 0, 0, 0])

    assert tensors[0].shape == (1, 2, 1)
    assert tensors[1].shape == (1, 2, 1)
    np.testing.assert_allclose(
        reconstructed, expected, atol=1e-12
    )
