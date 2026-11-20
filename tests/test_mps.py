"""State factorization, reconstruction, canonical forms, and QR center moves.

Compare represented states and normalization identities rather than QR factors,
since valid factorizations can differ by phases and internal gauge choices.
"""

import numpy as np
import pytest

from ising_dmrg.mps import (
    assert_mixed_canonical,
    canonicalize_at_zero,
    mps_to_state,
    random_mps,
    shift_left,
    shift_right,
    state_to_mps,
)

ATOL = 1e-10


def assert_canonical_identities(tensors, center):
    """Check the actual Gram matrices, independently of the package checker."""
    for site, tensor in enumerate(tensors):
        left, physical, right = tensor.shape
        if site < center:
            matrix = tensor.reshape(left * physical, right)
            np.testing.assert_allclose(
                matrix.conj().T @ matrix, np.eye(right), atol=ATOL, rtol=0
            )
        elif site > center:
            matrix = tensor.reshape(left, physical * right)
            np.testing.assert_allclose(
                matrix @ matrix.conj().T, np.eye(left), atol=ATOL, rtol=0
            )



def test_reconstruct_known_product_state():
    local = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    tensors = [
        local.reshape(1, 2, 1),
        np.array([0, 1], dtype=complex).reshape(1, 2, 1),
    ]
    expected = np.array([0, 1, 0, 1j], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(mps_to_state(tensors), expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_exact_round_trip(n):
    rng = np.random.default_rng(100 + n)
    state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    state /= np.linalg.norm(state)

    tensors = state_to_mps(state)
    np.testing.assert_allclose(mps_to_state(tensors), state, atol=ATOL, rtol=0)
    assert_canonical_identities(tensors, center=n - 1)


def test_bond_one_truncation():
    state = np.array([np.sqrt(0.9), 0, 0, np.sqrt(0.1)], dtype=complex)
    tensors = state_to_mps(state, max_bond=1)
    approximation = mps_to_state(tensors)

    assert [tensor.shape for tensor in tensors] == [(1, 2, 1), (1, 2, 1)]
    np.testing.assert_allclose(
        approximation, [np.sqrt(0.9), 0, 0, 0], atol=ATOL, rtol=0
    )
    # The discarded squared singular value is the squared reconstruction error.
    np.testing.assert_allclose(
        np.linalg.norm(state - approximation)**2, 0.1, atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(
        np.linalg.norm(approximation), np.sqrt(0.9), atol=ATOL, rtol=0
    )


def test_svd_isometries_and_center_norm_for_unnormalized_complex_state():
    rng = np.random.default_rng(137)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    tensors = state_to_mps(state)

    assert [tensor.shape for tensor in tensors] == [
        (1, 2, 2), (2, 2, 4), (4, 2, 2), (2, 2, 1)
    ]
    np.testing.assert_allclose(mps_to_state(tensors), state, atol=ATOL, rtol=0)
    assert_canonical_identities(tensors, center=3)
    np.testing.assert_allclose(np.linalg.norm(tensors[3]), np.linalg.norm(state))


@pytest.mark.parametrize("normalize", [False, True])
def test_canonicalize_at_zero_preserves_or_normalizes_the_state(normalize):
    rng = np.random.default_rng(827)
    state = rng.normal(size=32) + 1j * rng.normal(size=32)
    tensors = state_to_mps(state)
    original = [tensor.copy() for tensor in tensors]

    centered = canonicalize_at_zero(tensors, normalize=normalize)
    expected = state / np.linalg.norm(state) if normalize else state
    np.testing.assert_allclose(mps_to_state(centered), expected, atol=ATOL, rtol=0)
    assert_canonical_identities(centered, center=0)
    np.testing.assert_allclose(np.linalg.norm(centered[0]), np.linalg.norm(expected))
    for actual, before in zip(tensors, original):
        np.testing.assert_array_equal(actual, before)


@pytest.mark.parametrize("n,seed", [(2, 99), (4, 137), (5, 99)])
def test_qr_moves_preserve_complex_state_and_do_not_mutate_inputs(n, seed):
    """Move the center to the last site and back, including boundary cases."""
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    centered_at_zero = canonicalize_at_zero(state_to_mps(state), normalize=False)
    baseline = mps_to_state(centered_at_zero)
    snapshot = [tensor.copy() for tensor in centered_at_zero]
    current = centered_at_zero
    global_norm = np.linalg.norm(baseline)

    np.testing.assert_allclose(baseline, state, atol=ATOL, rtol=0)
    assert_canonical_identities(current, center=0)
    assert_mixed_canonical(current, center=0)
    moves = [(shift_right, site, site + 1) for site in range(n - 1)]
    moves += [(shift_left, site, site - 1) for site in range(n - 1, 0, -1)]

    for move, site, center in moves:
        before = [tensor.copy() for tensor in current]
        output = move(current, site)
        np.testing.assert_allclose(mps_to_state(output), baseline, atol=ATOL, rtol=0)
        assert_canonical_identities(output, center)
        assert_mixed_canonical(output, center=center)
        np.testing.assert_allclose(
            np.linalg.norm(output[center]), global_norm, atol=ATOL, rtol=0
        )
        for actual, saved in zip(current, before):
            np.testing.assert_array_equal(actual, saved)
        # Mutating the output must not change the input, including untouched sites.
        for tensor in output:
            tensor.flat[0] += 1
        for actual, saved in zip(current, before):
            np.testing.assert_array_equal(actual, saved)
        current = move(current, site)

    # Verify the final state, norm, and original centered input.
    assert_mixed_canonical(current, center=0)
    np.testing.assert_allclose(mps_to_state(current), baseline, atol=ATOL, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(current[0]), global_norm, atol=ATOL, rtol=0)
    for original, unchanged in zip(snapshot, centered_at_zero):
        np.testing.assert_array_equal(original, unchanged)


@pytest.mark.parametrize("cap", [1, 2, 4, 8])
def test_random_mps_has_feasible_bonds_and_is_normalized(cap):
    n = 6
    tensors = random_mps(n, max_bond=cap, seed=500 + cap)
    bonds = [min(cap, 2**j, 2 ** (n - j)) for j in range(n + 1)]
    assert [tensor.shape for tensor in tensors] == [
        (bonds[j], 2, bonds[j + 1]) for j in range(n)
    ]
    assert_canonical_identities(tensors, center=0)
    np.testing.assert_allclose(np.linalg.norm(mps_to_state(tensors)), 1, atol=ATOL)


def test_canonical_checker_does_not_require_normalized_center():
    tensors = random_mps(4, max_bond=2, seed=46)
    tensors[0] *= 3j
    assert_mixed_canonical(tensors, center=0)
    np.testing.assert_allclose(np.linalg.norm(mps_to_state(tensors)), 3, atol=ATOL)

    # An exterior tensor must still obey its isometry condition.
    tensors[2] *= 2
    with pytest.raises(AssertionError):
        assert_mixed_canonical(tensors, center=0)
