"""Dense local embeddings, one-site updates, and finite DMRG sweeps.

This implementation is intended for small-system validation. It explicitly
stores the many-body Hamiltonian, block embedding, and effective Hamiltonian.
"""
from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
ComplexArray = NDArray[np.complex128]

from .hamiltonian import I, kron_all, energy_expectation
from .mps import (
    assert_mixed_canonical,
    canonicalize_at_zero,
    copy_mps,
    mps_to_state,
    shift_left,
    shift_right,
    validate_mps,
)

def local_embedding(tensors: Sequence[NDArray], site: int) -> ComplexArray:
    """Construct the map from a flattened site tensor to the full state."""
    validate_mps(tensors)

    if not 0 <= site < len(tensors):
        raise ValueError("Site index is out of range.")

    # Empty left block with a dimension-one boundary bond.
    left_block = np.ones((1, 1), dtype=complex)

    for tensor in tensors[:site]:
        # Contract the block's open bond with this tensor's left bond.
        left_block = np.tensordot(left_block, tensor, axes=(-1, 0))

        # Combine physical indices while retaining the open right bond.
        left_block = left_block.reshape(-1, tensor.shape[2])

    # Empty right block with a dimension-one boundary bond.
    right_block = np.ones((1, 1), dtype=complex)

    for tensor in reversed(tensors[site + 1:]):
        # Contract this tensor's right bond with the block's open left bond.
        right_block = np.tensordot(tensor, right_block, axes=(2,0))

        # Retain the open left bond and combine physical indices.
        right_block = right_block.reshape(tensor.shape[0], -1)

    physical_identity = np.eye(2, dtype=complex)
    return kron_all([left_block, physical_identity, right_block.T])

def optimize_site(tensors: Sequence[NDArray], H: NDArray, site: int) -> tuple[list[ComplexArray], float]:
    """Minimize energy at a mixed-canonical center, preserving the input."""
    assert_mixed_canonical(tensors, center=site)
    output = copy_mps(tensors)

    W = local_embedding(output, site)

    # Express H in the orthonormal basis supplied by the embedding.
    H_eff = W.conj().T @ H @ W

    # Eigenvalues are ascending; eigenvectors are the corresponding columns.
    eigenvalues, eigenvectors = np.linalg.eigh(H_eff)

    # Restore the lowest-energy eigenvector to the center tensor's shape.
    output[site] = eigenvectors[:,0].reshape(output[site].shape)

    return output, float(eigenvalues[0])

def dmrg_sweep(tensors: Sequence[NDArray], H: NDArray) -> tuple[list[ComplexArray], list[float]]:
    """Perform a full sweep starting and ending with the center at site 0.

    Return the updated MPS and energy after each local optimization.
    Leave the input arrays unchanged.
    """
    assert_mixed_canonical(tensors, center=0)
    output = copy_mps(tensors)
    n = len(output)
    local_energies = []

    # Optimize each site from left to right, shifting the center to the next site.
    for site in range(n):
        output, energy = optimize_site(output, H, site)
        local_energies.append(energy)

        if site < n - 1:
            output = shift_right(output, site)

    # Shift the center left, then optimize; the last site was already optimized.
    for site in range(n - 2, -1, -1):
        output = shift_left(output, site + 1)
        output, energy = optimize_site(output, H, site)
        local_energies.append(energy)

    return output, local_energies

def run_dmrg(
    H: NDArray,
    tensors: Sequence[NDArray],
    max_sweeps: int = 24,
    energy_tol: float = 1e-11,
) -> dict[str, object]:
    """Repeat full sweeps until the energy stabilizes or the limit is reached."""
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be positive.")
    if not np.isfinite(energy_tol) or energy_tol < 0:
        raise ValueError("energy_tol must be finite and nonnegative.")

    # Normalize the initial state and establish its center at site 0.
    output = canonicalize_at_zero(tensors)

    energies = [energy_expectation(H, mps_to_state(output))]
    local_energies = []
    energy_converged = False

    for sweep in range(1, max_sweeps + 1):
        # Complete a sweep and record every local optimization's energy.
        output, sweep_energies = dmrg_sweep(output, H)
        local_energies.extend(sweep_energies)

        # Evaluate the updated full state's energy.
        energy = energy_expectation(H, mps_to_state(output))
        energies.append(energy)

        if sweep >= 2:
            energy_change = abs(energies[-1] - energies[-2])
            if energy_change <= energy_tol * max(1.0, abs(energy)):
                energy_converged = True
                break

    return {
        "mps": output,
        "energies": np.asarray(energies),
        "local_energies": np.asarray(local_energies),
        "sweeps": sweep,
        "energy_converged": energy_converged,
    }