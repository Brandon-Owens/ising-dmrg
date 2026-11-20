"""MPS representation and QR gauge moves.

Each tensor has axes (left bond, physical qubit, right bond).
Qubit 0 is the leftmost factor; flattening uses NumPy C order.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray, ArrayLike

ComplexArray = NDArray[np.complex128]


def validate_mps(tensors: Sequence[NDArray]) -> None:
    """Check qubit MPS shapes, matching bonds, and dimension-one boundaries."""
    if len(tensors) < 2:
        raise ValueError("Need at least two sites.")
    for site, tensor in enumerate(tensors):
        if tensor.ndim != 3 or tensor.shape[1] != 2 or min(tensor.shape) < 1:
            raise ValueError(f"Site {site} needs shape (alpha, 2, beta).")
        if site and tensors[site - 1].shape[2] != tensor.shape[0]:
            raise ValueError(f"Bond mismatch before site {site}.")
    if tensors[0].shape[0] != 1 or tensors[-1].shape[2] != 1:
        raise ValueError("Boundary bonds must have dimension one.")


def copy_mps(tensors: Sequence[NDArray]) -> list[ComplexArray]:
    """Return independent complex copies of all site tensors."""
    return [np.asarray(tensor, dtype=complex).copy() for tensor in tensors]


def mps_to_state(tensors: Sequence[NDArray]) -> NDArray:
    """Contract an open-boundary qubit MPS into a flat state vector.

    Each site tensor has axes (left bond, physical, right bond).
    The first left-bond dimension and last right-bond dimension are 1.
    """
    # Select the only value of the first tensor's left boundary index
    contracted = tensors[0][0, :, :]

    # Contract the running right bond with each next tensor's left bond
    for tensor in tensors[1:]:
        contracted = np.tensordot(contracted, tensor, axes=(-1, 0))

    # Return the resulting coefficient tensor as a flat array
    return contracted.reshape(-1)


def state_to_mps(state: ArrayLike, max_bond: int | None = None) -> list[NDArray[np.complex128]]:
    """Factor a flat N-qubit vector into an MPS by successive economy SVDs, with N >= 2.

    All site tensors except the last are left-normalized.

    Return a list of tensors with axes (left bond, physical qubit, right bond).
    The last tensor carries the remaining coefficients.
    max_bond=None retains all economy SVD directions; a positive integer
    caps each bond dimension. The output is not renormalized.
    """

    state = np.asarray(state, dtype=complex)

    # Require a flat vector whose length is 2**N, with N >= 2.
    if state.ndim != 1 or state.size < 4 or state.size & (state.size - 1):
        raise ValueError("Need a flat vector of length 2**N, N >= 2.")

    # Require a positive integer bond cap, excluding boolean values.
    if max_bond is not None and (
        isinstance(max_bond, (bool, np.bool_))
        or not isinstance(max_bond, (int, np.integer))
        or max_bond < 1
    ):
        raise ValueError("max_bond must be a positive integer or None.")

    N = int(round(np.log2(state.size)))
    tensors, remainder, l_bond_dim = [], state.copy(), 1
    # Successive economy SVD rounds extract left-normalized tensors for the first N - 1 sites.
    for _ in range(N - 1):
        matrix = remainder.reshape(l_bond_dim * 2, -1)
        u, singular_values, v_dag = np.linalg.svd(matrix, full_matrices=False)

        if max_bond is None:
            r_bond_dim = len(singular_values)
        else:
            r_bond_dim = min(max_bond, len(singular_values))

        u_kept = u[:, :r_bond_dim]
        site_tensor = u_kept.reshape(l_bond_dim, 2, r_bond_dim)
        tensors.append(site_tensor)

        # Absorb singular values into right singular vectors
        remainder = (np.diag(singular_values[:r_bond_dim]) @ v_dag[:r_bond_dim, :])
        l_bond_dim = r_bond_dim

    # Store the last site with a dimension-one right boundary bond.
    tensors.append(remainder.reshape(l_bond_dim, 2, 1))
    return tensors


def shift_right(tensors: Sequence[NDArray], site: int) -> list[NDArray]:
    """Move a mixed-canonical center right using QR; preserve state and input.

    Assumes valid open-boundary MPS structure.
    """
    if not 0 <= site < len(tensors) - 1:
        raise ValueError("A right neighbor is required.")

    output = copy_mps(tensors)
    l_bond_dim, physical_dim, r_bond_dim = output[site].shape

    # Factor M = QR; retain Q and absorb R into the right neighbor.
    matrix = output[site].reshape(l_bond_dim * physical_dim, r_bond_dim)
    q, r = np.linalg.qr(matrix, mode="reduced")
    output[site] = q.reshape(l_bond_dim, physical_dim, q.shape[1])
    output[site + 1] = np.tensordot(r, output[site + 1], axes=(1, 0))
    return output


def shift_left(tensors: Sequence[NDArray], site: int) -> list[NDArray]:
    """Move a mixed-canonical center left using QR of the adjoint; preserve state and input.

    Assumes valid open-boundary MPS structure.
    """
    if not 0 < site < len(tensors):
        raise ValueError("A left neighbor is required.")

    output = copy_mps(tensors)
    l_bond_dim, physical_dim, r_bond_dim = output[site].shape

    # Factor M† = QR; retain Q† and absorb R† into the left neighbor.
    matrix = output[site].reshape(l_bond_dim, physical_dim * r_bond_dim)
    q, r = np.linalg.qr(matrix.conj().T, mode="reduced")
    output[site] = q.conj().T.reshape(q.shape[1], physical_dim, r_bond_dim)
    output[site - 1] = np.tensordot(output[site - 1], r.conj().T, axes=(2, 0))
    return output

def assert_mixed_canonical(tensors: Sequence[NDArray], center: int, atol: float = 1e-10) -> None:
    """Raise AssertionError unless all tensors outside center are isometries."""
    validate_mps(tensors)
    if not 0 <= center < len(tensors):
        raise ValueError("Invalid center site.")
    for site, tensor in enumerate(tensors):
        l_bond_dim, physical_dim, r_bond_dim = tensor.shape
        if site < center:
            matrix = tensor.reshape(l_bond_dim * physical_dim, r_bond_dim)
            np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(r_bond_dim), atol=atol, rtol=0)
        elif site > center:
            matrix = tensor.reshape(l_bond_dim, physical_dim * r_bond_dim)
            np.testing.assert_allclose(matrix @ matrix.conj().T, np.eye(l_bond_dim), atol=atol, rtol=0)


def canonicalize_at_zero(tensors: Sequence[NDArray], normalize: bool = True) -> list[ComplexArray]:
    """Right-normalize sites 1 through N-1 while preserving the state.

    If normalize=True, also normalize the represented state by scaling site 0.
    """
    validate_mps(tensors)
    output = copy_mps(tensors)
    # Working backwards from site N-1 to site 1, right-normalize each site, absorbing each QR remainder leftward.
    # This establishes the orthogonality center at site 0.
    for site in range((len(output) - 1), 0, -1):
        output = shift_left(output, site)

    if normalize:
        # The Frobenius norm of the center of orthogonality (site-tensor 0) is equal to
        # the global norm (i.e. of the full MPS state) after right-canonicalizing.
        norm = np.linalg.norm(output[0])
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("The computed state norm must be finite and nonzero.")
        output[0] /= norm
    return output


def random_mps(n: int, max_bond: int, seed: int | None = 0) -> list[ComplexArray]:
    """Generate normalized complex factors directly, with center at site 0.

    Bond j has dimension min(max_bond, 2**j, 2**(n-j)). This does not
    construct a dense random state before factorization.
    """
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in (n, max_bond)
    ) or n < 2 or max_bond < 1:
        raise ValueError("Need integer n >= 2 and max_bond >= 1.")
    rng = np.random.default_rng(seed)
    bonds = [min(max_bond, 2**site, 2 ** (n - site)) for site in range(n + 1)]
    tensors = []
    for site in range(n):
        shape = (bonds[site], 2, bonds[site + 1])
        tensors.append(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    return canonicalize_at_zero(tensors)
