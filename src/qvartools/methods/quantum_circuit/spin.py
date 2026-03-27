"""
spin --- Quantum circuit SKQD for spin-lattice Hamiltonians
============================================================

End-to-end pipeline that uses :class:`QuantumCircuitSKQD` with
Trotterized ``exp_pauli`` circuits for Krylov state generation on
Heisenberg spin chains.  Matches the NVIDIA CUDA-Q SKQD tutorial setup.

The pipeline:
    1. Construct Heisenberg Pauli decomposition directly
    2. Prepare Neel state |010101...> as reference
    3. Generate Krylov states via Trotterized evolution
    4. Sample computational-basis configurations
    5. Accumulate basis and diagonalise classically

Functions
---------
run_quantum_skqd_spin
    Execute the spin-lattice quantum circuit SKQD pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from qvartools.krylov.circuits.circuit_skqd import (
    QuantumCircuitSKQD,
    QuantumSKQDConfig,
)
from qvartools.solvers.solver import SolverResult

__all__ = [
    "QuantumSKQDSpinConfig",
    "run_quantum_skqd_spin",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantumSKQDSpinConfig:
    """Configuration for the quantum circuit SKQD spin pipeline.

    Parameters
    ----------
    n_spins : int
        Number of spins in the chain (default ``10``).
    Jx : float
        XX coupling constant (default ``1.0``).
    Jy : float
        YY coupling constant (default ``1.0``).
    Jz : float
        ZZ coupling constant (default ``1.0``).
    max_krylov_dim : int
        Maximum Krylov subspace dimension (default ``12``).
    total_evolution_time : float
        Total time for one evolution ``U = e^{-iHT}`` (default ``pi``).
    num_trotter_steps : int
        Trotter steps per evolution (default ``8``).
    trotter_order : int
        Suzuki-Trotter order: ``1`` or ``2`` (default ``2``).
    shots : int
        Measurement shots per Krylov state (default ``100_000``).
    num_eigenvalues : int
        Number of lowest eigenvalues to compute (default ``2``).
    backend : str
        Sampling backend (default ``"auto"``).
    cudaq_target : str
        CUDA-Q simulation target (default ``"nvidia"``).
    cudaq_option : str
        CUDA-Q target option (default ``"fp64"``).
    use_gpu : bool
        Use GPU for post-processing (default ``True``).
    seed : int
        Random seed (default ``42``).
    exact_ground_state_energy : float or None
        Pre-computed exact energy for error reporting.  If ``None``,
        no error is computed.
    """

    n_spins: int = 10
    Jx: float = 1.0
    Jy: float = 1.0
    Jz: float = 1.0
    max_krylov_dim: int = 12
    total_evolution_time: float = np.pi
    num_trotter_steps: int = 8
    trotter_order: int = 2
    shots: int = 100_000
    num_eigenvalues: int = 2
    backend: str = "auto"
    cudaq_target: str = "nvidia"
    cudaq_option: str = "fp64"
    use_gpu: bool = True
    seed: int = 42
    exact_ground_state_energy: float | None = None

    def __post_init__(self) -> None:
        if self.n_spins < 2:
            raise ValueError(f"n_spins must be >= 2, got {self.n_spins}")
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
            initial_state="neel",
            backend=self.backend,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_quantum_skqd_spin(
    config: QuantumSKQDSpinConfig | None = None,
) -> SolverResult:
    """Execute the quantum circuit SKQD pipeline for a Heisenberg chain.

    Parameters
    ----------
    config : QuantumSKQDSpinConfig or None
        Pipeline configuration.  If ``None``, uses defaults (10 spins).

    Returns
    -------
    SolverResult
        Energy, basis dimension, wall time, and detailed metadata.
    """
    cfg = config or QuantumSKQDSpinConfig()

    logger.info(
        "run_quantum_skqd_spin: %d spins, J=(%.2f, %.2f, %.2f)",
        cfg.n_spins,
        cfg.Jx,
        cfg.Jy,
        cfg.Jz,
    )

    t_start = time.perf_counter()

    # --- Build QuantumCircuitSKQD via Heisenberg factory ---
    quantum_skqd_config = cfg.to_quantum_skqd_config()
    qskqd = QuantumCircuitSKQD.from_heisenberg(
        n_spins=cfg.n_spins,
        Jx=cfg.Jx,
        Jy=cfg.Jy,
        Jz=cfg.Jz,
        config=quantum_skqd_config,
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

    # Error vs exact (if provided)
    error_mha: float | None = None
    if cfg.exact_ground_state_energy is not None:
        error_mha = (best_energy - cfg.exact_ground_state_energy) * 1000.0

    logger.info("  Best energy: %.10f (wall: %.2f s)", best_energy, wall_time)

    return SolverResult(
        energy=float(best_energy),
        diag_dim=int(basis_sizes[-1]) if basis_sizes else 0,
        wall_time=wall_time,
        method="QuantumCircuit-SKQD-Spin",
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
            "n_spins": cfg.n_spins,
            "J": (cfg.Jx, cfg.Jy, cfg.Jz),
            "exact_ground_state_energy": cfg.exact_ground_state_energy,
            "error_mha": error_mha,
        },
    )
