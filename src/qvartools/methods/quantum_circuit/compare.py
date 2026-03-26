"""
compare --- Side-by-side classical vs quantum circuit SKQD comparison
=====================================================================

Runs both :class:`ClassicalKrylovDiagonalization` (exact ``matrix_exp``)
and :class:`QuantumCircuitSKQD` (Trotterized ``exp_pauli``) on the same
Hamiltonian and reports energy, basis size, and wall time for each.

Functions
---------
compare_krylov_generators
    Run both generators and return a structured comparison dict.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from qvartools.krylov.basis.skqd import (
    ClassicalKrylovDiagonalization,
    SKQDConfig,
)
from qvartools.krylov.circuits.circuit_skqd import (
    QuantumCircuitSKQD,
    QuantumSKQDConfig,
)

__all__ = [
    "ComparisonConfig",
    "compare_krylov_generators",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for the classical-vs-quantum SKQD comparison.

    Parameters
    ----------
    max_krylov_dim : int
        Krylov subspace dimension used by both generators (default ``10``).
    total_evolution_time : float
        Total time ``T`` for one evolution ``U = e^{-iHT}``
        (default ``pi``).
    shots : int
        Shots per Krylov state (both generators, default ``100_000``).
    num_trotter_steps : int
        Trotter steps for the quantum circuit generator (default ``8``).
    trotter_order : int
        Suzuki-Trotter order for the quantum generator (default ``2``).
    classical_time_step : float or None
        Time step for the classical generator.  If ``None``, computed as
        ``total_evolution_time / num_trotter_steps`` for a fair comparison.
    backend : str
        Backend for the quantum generator (default ``"auto"``).
    use_gpu : bool
        GPU acceleration for both generators (default ``True``).
    seed : int
        Random seed (default ``42``).
    """

    max_krylov_dim: int = 10
    total_evolution_time: float = np.pi
    shots: int = 100_000
    num_trotter_steps: int = 8
    trotter_order: int = 2
    classical_time_step: float | None = None
    backend: str = "auto"
    use_gpu: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_krylov_dim < 2:
            raise ValueError(
                f"max_krylov_dim must be >= 2, got {self.max_krylov_dim}"
            )


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------


def compare_krylov_generators(
    hamiltonian: Any,
    mol_info: dict[str, Any],
    exact_energy: float,
    config: ComparisonConfig | None = None,
) -> dict[str, Any]:
    """Run classical and quantum SKQD side-by-side and compare.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Molecular Hamiltonian (must support both Slater-Condon and
        Pauli form via JW mapping).
    mol_info : dict
        Molecular metadata (``n_orbitals``, ``n_alpha``, ``n_beta``,
        ``n_qubits``, ``name``).
    exact_energy : float
        Exact ground-state energy for error computation.
    config : ComparisonConfig or None
        Comparison parameters.  If ``None``, uses defaults.

    Returns
    -------
    dict
        Structured comparison with keys ``"classical"``, ``"quantum"``,
        and ``"summary"``.
    """
    cfg = config or ComparisonConfig()
    dt = cfg.classical_time_step or (
        cfg.total_evolution_time / cfg.num_trotter_steps
    )
    mol_name = mol_info.get("name", "unknown")

    print(f"{'=' * 70}")
    print("  SKQD Comparison: Classical vs Quantum Circuit")
    print(f"  Molecule: {mol_name} ({mol_info['n_qubits']} qubits)")
    print(f"  Krylov dim: {cfg.max_krylov_dim}, shots: {cfg.shots:,}")
    print(f"  Trotter: order={cfg.trotter_order}, steps={cfg.num_trotter_steps}")
    print(f"  Exact energy: {exact_energy:.10f} Ha")
    print(f"{'=' * 70}")

    # ---- Classical SKQD ----
    print(f"\n{'─' * 40}")
    print("  Classical SKQD (exact matrix_exp)")
    print(f"{'─' * 40}")

    classical_config = SKQDConfig(
        max_krylov_dim=cfg.max_krylov_dim,
        time_step=dt,
        shots_per_krylov=cfg.shots,
        use_cumulative_basis=True,
        num_eigenvalues=2,
        regularization=1e-8,
        use_gpu=cfg.use_gpu,
    )

    t_start = time.perf_counter()
    classical_skqd = ClassicalKrylovDiagonalization(
        hamiltonian, classical_config
    )
    classical_eigenvalues, classical_info = classical_skqd.run()
    classical_time = time.perf_counter() - t_start

    classical_energy = float(classical_eigenvalues[0])
    classical_error_mha = (classical_energy - exact_energy) * 1000.0

    print(f"  Energy : {classical_energy:.10f} Ha")
    print(f"  Error  : {classical_error_mha:.4f} mHa")
    print(f"  Basis  : {classical_info.get('final_basis_size', '?')}")
    print(f"  Time   : {classical_time:.2f} s")

    # ---- Quantum Circuit SKQD ----
    print(f"\n{'─' * 40}")
    print("  Quantum Circuit SKQD (Trotterized exp_pauli)")
    print(f"{'─' * 40}")

    quantum_config = QuantumSKQDConfig(
        max_krylov_dim=cfg.max_krylov_dim,
        total_evolution_time=cfg.total_evolution_time,
        num_trotter_steps=cfg.num_trotter_steps,
        trotter_order=cfg.trotter_order,
        shots=cfg.shots,
        num_eigenvalues=2,
        use_gpu=cfg.use_gpu,
        seed=cfg.seed,
        initial_state="hf",
        backend=cfg.backend,
    )

    t_start = time.perf_counter()
    qskqd = QuantumCircuitSKQD.from_molecular_hamiltonian(
        hamiltonian, config=quantum_config
    )
    quantum_results = qskqd.run(progress=True)
    quantum_time = time.perf_counter() - t_start

    quantum_energy = quantum_results["best_energy"]
    quantum_error_mha = (quantum_energy - exact_energy) * 1000.0

    print(f"  Energy : {quantum_energy:.10f} Ha")
    print(f"  Error  : {quantum_error_mha:.4f} mHa")
    print(f"  Basis  : {quantum_results['basis_sizes'][-1] if quantum_results['basis_sizes'] else '?'}")
    print(f"  Time   : {quantum_time:.2f} s")
    print(f"  Backend: {quantum_results['backend']}")

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<25} {'Classical':>15} {'Quantum':>15} {'Delta':>12}")
    print(f"  {'─' * 67}")
    print(
        f"  {'Energy (Ha)':<25} {classical_energy:>15.8f} "
        f"{quantum_energy:>15.8f} {quantum_energy - classical_energy:>12.6f}"
    )
    print(
        f"  {'Error (mHa)':<25} {classical_error_mha:>15.4f} "
        f"{quantum_error_mha:>15.4f} {quantum_error_mha - classical_error_mha:>12.4f}"
    )
    print(
        f"  {'Wall time (s)':<25} {classical_time:>15.2f} "
        f"{quantum_time:>15.2f} {quantum_time - classical_time:>12.2f}"
    )
    print(f"{'=' * 70}")

    return {
        "classical": {
            "energy": classical_energy,
            "error_mha": classical_error_mha,
            "wall_time": classical_time,
            "eigenvalues": classical_eigenvalues.tolist(),
            "info": classical_info,
        },
        "quantum": {
            "energy": quantum_energy,
            "error_mha": quantum_error_mha,
            "wall_time": quantum_time,
            "energies_per_krylov": quantum_results["energies"],
            "basis_sizes": quantum_results["basis_sizes"],
            "backend": quantum_results["backend"],
        },
        "summary": {
            "molecule": mol_name,
            "n_qubits": mol_info["n_qubits"],
            "exact_energy": exact_energy,
            "energy_delta": quantum_energy - classical_energy,
            "error_delta_mha": quantum_error_mha - classical_error_mha,
            "speedup": classical_time / quantum_time if quantum_time > 0 else float("inf"),
        },
    }
