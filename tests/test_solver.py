"""Local embeddings, one-site optimization, and complete DMRG sweeps.

Small dense systems allow independent checks against full state vectors
and exact diagonalization.
"""

import numpy as np
import pytest

from ising_dmrg.hamiltonian import ising_hamiltonian, energy_expectation, eigenstate_metrics
from ising_dmrg.mps import assert_mixed_canonical, mps_to_state, random_mps, shift_right
from ising_dmrg.solver import dmrg_sweep, local_embedding, optimize_site, run_dmrg


def centered_random_mps(n, cap, center, seed):
    tensors = random_mps(n, max_bond=cap, seed=seed)
    for site in range(center):
        tensors = shift_right(tensors, site)
    return tensors


@pytest.mark.parametrize("n,site", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_embedding_reconstructs_center_and_has_orthonormal_columns(n, site):
    # Check both boundary and interior sites.
    tensors = centered_random_mps(n, 2, site, seed=314)
    W = local_embedding(tensors, site=site)
    local_dimension = tensors[site].size
    assert W.shape == (2**n, local_dimension)
    np.testing.assert_allclose(
        W @ tensors[site].reshape(-1), mps_to_state(tensors), atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        W.conj().T @ W, np.eye(local_dimension), atol=1e-10, rtol=0
    )

    # Verify the map for other center coefficients, not just the initial state.
    rng = np.random.default_rng(78)
    theta = rng.normal(size=local_dimension) + 1j * rng.normal(size=local_dimension)
    replaced = [tensor.copy() for tensor in tensors]
    replaced[site] = theta.reshape(tensors[site].shape)
    np.testing.assert_allclose(W @ theta, mps_to_state(replaced), atol=1e-10, rtol=0)


@pytest.mark.parametrize("site", [0, 1, 3])
def test_local_update_minimizes_restricted_energy_and_changes_only_center(site):
    H = ising_hamiltonian(4)
    tensors = centered_random_mps(4, 2, site, seed=314)
    saved = [tensor.copy() for tensor in tensors]
    W = local_embedding(tensors, site)
    expected_energy = np.linalg.eigvalsh(W.conj().T @ H @ W)[0]
    before = energy_expectation(H, mps_to_state(tensors))

    updated, local_energy = optimize_site(tensors, H, site)
    state = mps_to_state(updated)
    np.testing.assert_allclose(local_energy, expected_energy, atol=1e-10, rtol=0)
    np.testing.assert_allclose(local_energy, energy_expectation(H, state), atol=1e-10, rtol=0)
    assert local_energy <= before + 1e-10
    assert local_energy >= np.linalg.eigvalsh(H)[0] - 1e-10
    np.testing.assert_allclose(np.linalg.norm(state), 1, atol=1e-10)
    assert_mixed_canonical(updated, center=site)
    for j, tensor in enumerate(updated):
        assert tensor.shape == tensors[j].shape
        if j != site:
            np.testing.assert_array_equal(tensor, saved[j])
    for tensor, original in zip(tensors, saved):
        np.testing.assert_array_equal(tensor, original)


def test_full_sweep_returns_center_zero_with_nonincreasing_local_energies():
    H = ising_hamiltonian(4)
    initial = random_mps(4, max_bond=2, seed=99)
    saved = [tensor.copy() for tensor in initial]
    output, energies = dmrg_sweep(initial, H)
    assert len(energies) == 2 * len(initial) - 1
    history = np.r_[energy_expectation(H, mps_to_state(initial)), energies]
    assert np.all(np.diff(history) <= 1e-9)
    np.testing.assert_allclose(energies[-1], energy_expectation(H, mps_to_state(output)), atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(mps_to_state(output)), 1, atol=1e-10)
    assert_mixed_canonical(output, center=0)
    for tensor, original in zip(initial, saved):
        np.testing.assert_array_equal(tensor, original)


@pytest.mark.parametrize("n,cap,seed", [(2, 2, 412), (4, 4, 414), (6, 8, 508)])
def test_full_bond_solver_agrees_with_exact_ground_state(n, cap, seed):
    # Include a six-site integration case.
    H = ising_hamiltonian(n)
    values, vectors = np.linalg.eigh(H)
    result = run_dmrg(H, random_mps(n, max_bond=cap, seed=seed), max_sweeps=8)
    state = mps_to_state(result["mps"])
    metrics = eigenstate_metrics(H, state, vectors[:, 0])

    np.testing.assert_allclose(metrics["energy"], values[0], atol=1e-9, rtol=0)
    assert metrics["residual"] < 1e-7
    assert metrics["fidelity"] > 1 - 1e-9
    np.testing.assert_allclose(np.linalg.norm(state), 1, atol=1e-10)
    assert_mixed_canonical(result["mps"], center=0)
    assert len(result["energies"]) == result["sweeps"] + 1
    assert len(result["local_energies"]) == result["sweeps"] * (2 * n - 1)
    assert np.all(np.diff(result["energies"]) <= 1e-9)
    assert np.all(np.diff(result["local_energies"]) <= 1e-9)
    np.testing.assert_allclose(result["energies"][-1], metrics["energy"], atol=1e-10)


@pytest.mark.parametrize("cap", [1, 2, 4])
def test_capped_runs_respect_bonds_normalization_and_energy_bound(cap):
    H = ising_hamiltonian(6)
    result = run_dmrg(
        H, random_mps(6, max_bond=cap, seed=500 + cap),
        max_sweeps=24, energy_tol=1e-11,
    )
    state = mps_to_state(result["mps"])
    assert energy_expectation(H, state) >= np.linalg.eigvalsh(H)[0] - 1e-9
    assert all(tensor.shape[2] <= cap for tensor in result["mps"])
    np.testing.assert_allclose(np.linalg.norm(state), 1, atol=1e-10)
    assert np.all(np.diff(result["local_energies"]) <= 1e-9)


def test_one_sweep_budget_does_not_claim_energy_convergence():
    result = run_dmrg(
        ising_hamiltonian(4), random_mps(4, max_bond=2, seed=46), max_sweeps=1
    )
    assert result["sweeps"] == 1
    assert not result["energy_converged"]
