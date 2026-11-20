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

    # Since H is Hermitian, its eigenvalues are necessarily real,
    # so the expectation value should be a real number.
    # Taking .real discards any tiny imaginary component introduced by floating-point arithmetic.
    return float(energy.real)

# Measures how close the trial state is to an eigenstate and, when an exact ground state is available, compares the two.
def state_diagnostics(H: NDArray, state: ArrayLike, ground: ArrayLike | None = None) -> dict[str, float]:
    """Return energy, eigenstate residual, and optional ground-state fidelity."""
    psi = normalized(state)
    energy = energy_expectation(H, psi)

    # Measure the mismatch H|psi> - E(psi)|psi>, where E(psi) is the energy expectation.
    residual = np.linalg.norm((H @ psi) - (energy * psi))

    diagnostics = {"energy": energy, "residual": float(residual)}

    if ground is not None:
        ground_state = normalized(ground)

        # Squared magnitude of the overlap between normalized states.
        # It is unchanged by an overall phase on either state.
        fidelity = np.abs(np.vdot(ground_state, psi))**2
        diagnostics["fidelity"] = float(fidelity)

    return diagnostics