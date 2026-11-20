"""Dense open-chain Ising Hamiltonian and full-state diagnostics."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray, ArrayLike

ComplexArray = NDArray[np.complex128]

# The single qubit Pauli operators we will need.
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)

# Helper function for constructing tensor products of Pauli operators
def kron_all(operators: Sequence[NDArray]) -> ComplexArray:
    """Return the Kronecker product of the operators in site order."""
    result = np.ones((1, 1), dtype=complex)
    for operator in operators:
        result = np.kron(result, operator)
    return result

def ising_hamiltonian(n: int, J: float = 1.0, h: float = 1.0) -> ComplexArray:
    """Construct the dense open-chain transverse-field Ising Hamiltonian."""
    if n < 2:
        raise ValueError("Require at least two sites.")

    H = np.zeros((2**n, 2**n), dtype=complex)

    # Add the nearest-neighbor interaction terms.
    for site in range(n - 1):
        operators = [I] * n
        operators[site] = Z
        operators[site + 1] = Z
        H -= J * kron_all(operators)

    # Add the single-site transverse-field terms.
    for site in range(n):
        operators = [I] * n
        operators[site] = X
        H -= h * kron_all(operators)
    return H

def normalized(state: ArrayLike) -> ComplexArray:
    """Return a normalized copy of a nonzero state vector."""
    state = np.asarray(state, dtype=complex)
    if state.ndim != 1:
        raise ValueError("Require a flat state vector.")
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("The computed state norm must be finite and nonzero.")

    return state / norm

def energy_expectation(H: NDArray, state: ArrayLike) -> float:
    """Return the energy expectation of a state for a Hermitian Hamiltonian."""
    psi = normalized(state)

    # Apply the Hamiltonian to the normalized state.
    H_psi = H @ psi

    # Form the inner product <psi|H|psi>.
    energy = np.vdot(psi, H_psi)

    # For Hermitian H, the energy expectation is real.
    # Allow a small imaginary component from numerical error.
    if abs(energy.imag) > 1e-8:
        raise ValueError(
            f"Expected a real energy expectation for Hermitian H; got {energy}."
            )

    return float(energy.real)

def eigenstate_metrics(H: NDArray, state: ArrayLike, eigenvector_ref: ArrayLike | None = None) -> dict[str, float]:
    """Return energy, eigenstate residual, and optional fidelity to a reference vector."""
    psi = normalized(state)
    energy = energy_expectation(H, psi)
    residual = np.linalg.norm((H @ psi) - (energy * psi))
    metrics = {"energy": energy, "residual": float(residual)}

    if eigenvector_ref is not None:
        normalized_ref = normalized(eigenvector_ref)
        metrics["fidelity"] = float(abs(np.vdot(normalized_ref, psi)) ** 2)

        # Squared magnitude of the overlap between normalized states.
        # It is unchanged by an overall phase on either state.
        fidelity = np.abs(np.vdot(normalized_ref, psi))**2
        metrics["fidelity"] = float(fidelity)

    return metrics