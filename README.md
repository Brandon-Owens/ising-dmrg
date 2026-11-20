# One-site DMRG for the transverse-field Ising chain

A NumPy implementation of the one-site density matrix renormalization group (DMRG) algorithm for ground-state calculations in an open Ising chain. The accompanying notebook develops the matrix product state (MPS) representation, QR center moves, and local energy minimization used by the solver, then compares its results with exact diagonalization.

The model is

$$
H=-J\sum_{j=0}^{N-2}Z_jZ_{j+1}-h\sum_{j=0}^{N-1}X_j,
$$

where $X_j$ and $Z_j$ are Pauli operators acting on site $j$, with the identity on every other site. Open boundaries omit the interaction between the last and first sites. Site $0$ is the leftmost qubit, arrays use NumPy's default C order, and MPS tensors have axes `(left bond, physical qubit, right bond)` with dimension-one boundary bonds.

## Implementation

The solver initializes an MPS directly from random site tensors, brings it into canonical form, and minimizes the energy by updating one site tensor at a time. QR decompositions move the orthogonality center between updates while preserving the represented state. Full sweeps move through the chain in both directions; the solver records local and sweep energies and stops when the energy-change criterion is met or the sweep budget is exhausted.

| Path | Contents |
|---|---|
| `src/ising_dmrg/mps.py` | Successive-SVD factorization, bond truncation, reconstruction, canonicalization, and QR center moves. |
| `src/ising_dmrg/hamiltonian.py` | Dense Ising Hamiltonians, energy expectations, residuals, and fidelity calculations. |
| `src/ising_dmrg/solver.py` | Local embedding matrices, effective Hamiltonians, one-site updates, and sweeps. |
| `notebooks/One_Site_DMRG_Ising.ipynb` | Mathematical exposition, demonstrations, and the bond-dimension experiment. |
| `tests/` | Tests for state representation, canonical forms, Hamiltonians, diagnostics, and solver behavior. |

## Setup and tests

Developed with Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root, install the project dependencies and run the tests:

```bash
uv sync
uv run pytest -v
```

The automated test suite checks complex-state reconstruction, truncation, state-preserving QR moves, input preservation, canonical conditions, independently specified Hamiltonians, energy expectations, local optimization, and agreement with exact ground states for small systems.

Open [the notebook](notebooks/One_Site_DMRG_Ising.ipynb) in VS Code and select the repository's `.venv` Python environment as its kernel. The notebook uses NumPy, Matplotlib, and IPython for its calculations and displays.

## Example

Run this in the project environment to solve a four-qubit chain and compare the result with exact diagonalization:

```python
import numpy as np

from ising_dmrg.hamiltonian import ising_hamiltonian, eigenstate_metrics
from ising_dmrg.mps import mps_to_state, random_mps
from ising_dmrg.solver import run_dmrg

H = ising_hamiltonian(4, J=1.0, h=1.0)
initial = random_mps(4, max_bond=4, seed=414)
result = run_dmrg(H, initial, max_sweeps=8)
state = mps_to_state(result["mps"])

exact_values, exact_vectors = np.linalg.eigh(H)
metrics = eigenstate_metrics(H, state, exact_vectors[:, 0])

print(f"Energy error: {metrics['energy'] - exact_values[0]:.3e}")
print(f"Residual: {metrics['residual']:.3e}")
print(f"Fidelity: {metrics['fidelity']:.12f}")
print("Sweeps:", result["sweeps"])
```

## Six-qubit bond-dimension experiment

The notebook compares bond caps $1,2,4,8$ for $N=6$ and $J=h=1$. Each run uses seed $500+\chi_{\max}$, a maximum of 24 full sweeps, and energy-change tolerance $10^{-11}$. Representative results are shown below; the exact reference energy is approximately $-7.296229810559$.

| Bond cap | Energy error | Residual | Fidelity |
|---:|---:|---:|---:|
| 1 | 3.94e-1 | 7.68e-1 | 0.46908064 |
| 2 | 1.91e-3 | 9.51e-2 | 0.99950652 |
| 4 | 8.18e-8 | 7.31e-4 | 0.99999999 |
| 8 | 2.7e-15 | 6.2e-15 | 1.00000000 |

All four runs satisfy the energy-change stopping criterion, despite their differing accuracies. Cap 1 restricts the MPS to product states, while cap 8 permits an exact representation of any six-qubit state. The cap-8 run agrees with exact diagonalization up to floating-point roundoff. These results describe one initialization per cap; they do not establish reliability across initializations.

## Scope

This is a dense reference implementation for small systems. It stores all $4^N$ Hamiltonian entries, constructs explicit local embedding matrices, and reconstructs full state vectors for diagnostics. These costs grow exponentially with the number of qubits, even when the MPS bond dimensions are small. The current solver runs on the CPU using NumPy.

One-site updates do not increase bond dimensions, and an energy plateau can occur above the ground-state energy. Energy error, residual, and fidelity are therefore checked separately. A possible extension would replace the full Hamiltonian and embedding matrices with contractions of tensors; two-site updates could also allow bond dimensions to change during optimization.

## References

- Ulrich Schollwöck, [The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477), *Annals of Physics* 326 (2011).
- Glen Evenbly, [Tensors.net tutorials](https://www.tensors.net/tutorials).
- [TensorNetwork.org: Matrix Product State / Tensor Train](https://tensornetwork.org/mps/) and [DMRG](https://tensornetwork.org/mps/algorithms/dmrg/).
