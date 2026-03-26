"""
molecular --- Quantum circuit SKQD for molecular Hamiltonians
==============================================================

End-to-end pipeline that uses :class:`QuantumCircuitSKQD` with
Trotterized ``exp_pauli`` circuits for Krylov state generation on
molecular systems.

The pipeline:
    1. Jordan-Wigner transform the molecular Hamiltonian to Pauli form
    2. Generate Krylov states via Trotterized evolution (CUDA-Q or
       classical GPU fallback)
    3. Sample computational-basis configurations from each Krylov state
    4. Accumulate basis (cumulative union across Krylov powers)
    5. Project H onto basis and diagonalise classically

This is the quantum-circuit counterpart to the classical
``FlowGuidedKrylovDiag`` path used by most experiment scripts.

Functions
---------
run_quantum_skqd
    Execute the molecular quantum circuit SKQD pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from qvartools.krylov.circuits.circuit_skqd import (
    QuantumCircuitSKQD,
    QuantumSKQDConfig,
)
from qvartools.solvers.solver import SolverResult

__all__ = [
    "QuantumSKQDMethodConfig",
    "run_quantum_skqd",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantumSKQDMethodConfig:
    """Configuration for the quantum circuit SKQD molecular pipeline.

    Parameters
    ----------
    max_krylov_dim : int
        Maximum Krylov subspace dimension (default ``12``).
    total_evolution_time : float
        Total time for one application of ``U = e^{-iHT}``
        (default ``pi``).
    num_trotter_steps : int
        Trotter steps per single evolution ``U`` (default ``1``).
    trotter_order : int
        Suzuki-Trotter order: ``1`` (first-order) or ``2``
        (second-order, default).
    shots : int
        Measurement shots per Krylov state (default ``100_000``).
    num_eigenvalues : int
        Number of lowest eigenvalues to compute (default ``2``).
    backend : str
        Sampling backend: ``"auto"`` (CUDA-Q if available, else
        classical), ``"cudaq"``, ``"classical"``, ``"exact"``,
        ``"lanczos"`` (default ``"auto"``).
    cudaq_target : str
        CUDA-Q simulation target (default ``"nvidia"``).
    cudaq_option : str
        CUDA-Q target option (default ``"fp64"``).
    use_gpu : bool
        Use GPU for post-processing (default ``True``).
    seed : int
        Random seed for reproducibility (default ``42``).
    device : str
        Torch device for tensor operations (default ``"cpu"``).
    """

    max_krylov_dim: int = 12
    total_evolution_time: float = np.pi
    num_trotter_steps: int = 1
    trotter_order: int = 2
    shots: int = 100_000
    num_eigenvalues: int = 2
    backend: str = "auto"
    cudaq_target: str = "nvidia"
    cudaq_option: str = "fp64"
    use_gpu: bool = True
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.max_krylov_dim < 2:
            raise ValueError(f"max_krylov_dim must be >= 2, got {self.max_krylov_dim}")
        if self.total_evolution_time <= 0.0:
            raise ValueError(
                f"total_evolution_time must be > 0, got {self.total_evolution_time}"
            )
        if self.num_trotter_steps < 1:
            raise ValueError(
                f"num_trotter_steps must be >= 1, got {self.num_trotter_steps}"
            )
        if self.trotter_order not in (1, 2):
            raise ValueError(f"trotter_order must be 1 or 2, got {self.trotter_order}")
        if self.shots < 1:
            raise ValueError(f"shots must be >= 1, got {self.shots}")

    def to_quantum_skqd_config(self) -> QuantumSKQDConfig:
        """Convert to the low-level :class:`QuantumSKQDConfig`."""
        return QuantumSKQDConfig(
            max_krylov_dim=self.max_krylov_dim,
            total_evolution_time=self.total_evolution_time,
            num_trotter_steps=self.num_trotter_steps,
            trotter_order=self.trotter_order,
            shots=self.shots,
            num_eigenvalues=self.num_eigenvalues,
            cudaq_target=self.cudaq_target,
            cudaq_option=self.cudaq_option,
            use_gpu=self.use_gpu,
            seed=self.seed,
            initial_state="hf",
            backend=self.backend,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_quantum_skqd(
    hamiltonian: Any,
    mol_info: dict[str, Any],
    config: QuantumSKQDMethodConfig | None = None,
) -> SolverResult:
    """Execute the quantum circuit SKQD pipeline for a molecular system.

    Pipeline:
        1. Jordan-Wigner mapping to Pauli form
        2. Trotterized Krylov evolution (CUDA-Q or GPU classical fallback)
        3. Cumulative basis sampling
        4. Projected Hamiltonian diagonalisation

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Molecular Hamiltonian (must have ``integrals``, ``h1e``, ``h2e``,
        ``nuclear_repulsion``, ``n_orbitals``).
    mol_info : dict
        Molecular metadata with keys ``"n_orbitals"``, ``"n_alpha"``,
        ``"n_beta"``, ``"n_qubits"``, ``"name"``.
    config : QuantumSKQDMethodConfig or None
        Pipeline configuration.  If ``None``, uses defaults.

    Returns
    -------
    SolverResult
        Energy, basis dimension, wall time, and detailed metadata
        including per-Krylov-step energies and basis sizes.
    """
    cfg = config or QuantumSKQDMethodConfig()

    n_orb: int = mol_info.get("n_orbitals", hamiltonian.n_orbitals)
    n_qubits: int = mol_info.get("n_qubits", 2 * n_orb)
    mol_name: str = mol_info.get("name", "unknown")

    logger.info(
        "run_quantum_skqd: %s (%d orbitals, %d qubits)",
        mol_name,
        n_orb,
        n_qubits,
    )

    t_start = time.perf_counter()

    # --- Build QuantumCircuitSKQD via factory ---
    quantum_skqd_config = cfg.to_quantum_skqd_config()
    qskqd = QuantumCircuitSKQD.from_molecular_hamiltonian(
        hamiltonian, config=quantum_skqd_config
    )

    logger.info(
        "  Pauli decomposition: %d terms, constant = %.8f",
        len(qskqd.pauli_coefficients),
        qskqd.constant_energy,
    )

    # --- Run full pipeline ---
    results = qskqd.run(progress=True)

    wall_time = time.perf_counter() - t_start

    best_energy = results["best_energy"]
    energies = results["energies"]
    basis_sizes = results["basis_sizes"]

    logger.info("  Best energy: %.10f Ha (wall: %.2f s)", best_energy, wall_time)

    return SolverResult(
        energy=float(best_energy),
        diag_dim=int(basis_sizes[-1]) if basis_sizes else 0,
        wall_time=wall_time,
        method="QuantumCircuit-SKQD",
        converged=True,
        metadata={
            "energies_per_krylov": energies,
            "basis_sizes": basis_sizes,
            "krylov_dims": results["krylov_dims"],
            "final_energy": results["final_energy"],
            "backend": results["backend"],
            "device": results["device"],
            "constant_energy": results["constant_energy"],
            "n_pauli_terms": results["n_pauli_terms"],
            "config": results["config"],
            "molecule": mol_name,
        },
    )
