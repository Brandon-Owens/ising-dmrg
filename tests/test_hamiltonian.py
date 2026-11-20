"""Dense Ising Hamiltonians, energy expectations, and state diagnostics.
"""

import numpy as np
import pytest

from ising_dmrg.hamiltonian import (
    ising_hamiltonian,
    energy_expectation,
    state_diagnostics,
)


def test_two_site_hamiltonian_matches_explicit_matrix_and_spectrum():
    # This explicit matrix independently fixes signs,
    # basis ordering, the number of field terms, and open boundaries.
    H = ising_hamiltonian(2)
    expected = np.array([
        [-1, -1, -1, 0],
        [-1, 1, 0, -1],
        [-1, 0, 1, -1],
        [0, -1, -1, -1],
    ], dtype=complex)
    np.testing.assert_allclose(H, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(H), [-np.sqrt(5), -1, 1, np.sqrt(5)],
        atol=1e-12, rtol=0,
    )


@pytest.mark.parametrize("J,h,expected_energy", [(1, 0, -3), (0, 1, -4)])
def test_four_site_ground_energy_in_exactly_solvable_limits(J, h, expected_energy):
    # Three bonds for the classical open chain; four independent field terms.
    values = np.linalg.eigvalsh(ising_hamiltonian(4, J=J, h=h))
    np.testing.assert_allclose(values[0], expected_energy, atol=1e-12, rtol=0)


def test_ground_state_diagnostics_ignore_nonzero_complex_scale():
    # Scaling by 3j changes both norm and global phase.
    H = ising_hamiltonian(2)
    values, vectors = np.linalg.eigh(H)
    ground = vectors[:, 0]
    diagnostics = state_diagnostics(H, 3j * ground, ground)
    np.testing.assert_allclose(diagnostics["energy"], values[0], atol=1e-12)
    np.testing.assert_allclose(diagnostics["residual"], 0, atol=1e-12)
    np.testing.assert_allclose(diagnostics["fidelity"], 1, atol=1e-12)
    np.testing.assert_allclose(energy_expectation(H, 3j * ground), values[0], atol=1e-12)


def test_diagnostics_for_superposition_of_two_energy_eigenstates():
    H = ising_hamiltonian(2)
    values, vectors = np.linalg.eigh(H)
    state = (vectors[:, 0] + 1j * vectors[:, 1]) / np.sqrt(2)
    diagnostics = state_diagnostics(H, state, vectors[:, 0])
    half_gap = (values[1] - values[0]) / 2

    np.testing.assert_allclose(diagnostics["energy"], (values[0] + values[1]) / 2)
    np.testing.assert_allclose(diagnostics["residual"], half_gap, atol=1e-12)
    np.testing.assert_allclose(diagnostics["fidelity"], 0.5, atol=1e-12)
