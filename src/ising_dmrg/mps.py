"""Matrix product state construction and reconstruction."""

import numpy as np


def state_to_left_mps(state, max_bond=None):
    """Factor a flat N-qubit vector into left-canonical MPS tensors.

    Return tensors with axes (left bond, physical qubit, right bond).
    The last tensor carries the remaining coefficients.
    max_bond=None retains all singular directions; a positive integer
    caps each bond dimension. The output is not renormalized.
    """
    state = np.asarray(state, dtype=complex)
    N = int(round(np.log2(state.size)))
    assert state.ndim == 1 and 2**N == state.size and N >= 2

    tensors = []
    remainder = state.copy()
    chi_left = 1

    for site in range(N - 1):
        matrix = remainder.reshape(chi_left * 2, -1)
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)

        if max_bond is None:
            chi_right = len(singular_values)
        else:
            chi_right = min(max_bond, len(singular_values))

        u_kept = u[:, :chi_right]
        site_tensor = u_kept.reshape(chi_left, 2, chi_right)
        tensors.append(site_tensor)

        remainder = (np.diag(singular_values[:chi_right]) @ vh[:chi_right, :])
        chi_left = chi_right

    tensors.append(remainder.reshape(chi_left, 2, 1))
    return tensors


def mps_to_state(tensors):
    """Contract an open-boundary qubit MPS into a flat state vector."""
    contracted = tensors[0][0, :, :]
    for tensor in tensors[1:]:
        contracted = np.tensordot(contracted, tensor, axes=(-1, 0))
    return contracted.reshape(-1)
