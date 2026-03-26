"""Iterative NQS + DCI+PT2 seed -> Quantum Circuit Krylov.

Pipeline:
  Phase 1: Generate DCI basis (HF + singles + doubles), then expand via
           PT2-style H-connection growth. Diagonalise to get seed energy.
  Phase 2: Iterative NQS warmup via SQD (few iterations).
  Phase 3: Quantum circuit Krylov (Trotterized exp_pauli) via
           run_quantum_skqd for the final diagonalisation.
  Report best energy across all three phases.

The PT2 expansion enriches the DCI seed with configurations connected
via the Hamiltonian to the most important reference determinants.
"""

from __future__ import annotations

import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config_loader import create_base_parser, load_config

from qvartools._utils.gpu.diagnostics import gpu_solve_fermion
from qvartools.krylov.expansion.krylov_expand import expand_basis_via_connections
from qvartools.methods.nqs.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd
from qvartools.methods.quantum_circuit.molecular import (
    QuantumSKQDMethodConfig,
    run_quantum_skqd,
)
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6


def generate_dci_configs(hamiltonian, device="cpu"):
    """Generate HF + singles + doubles deterministically."""
    n_orb = hamiltonian.integrals.n_orbitals
    n_alpha = hamiltonian.integrals.n_alpha
    n_beta = hamiltonian.integrals.n_beta
    n_qubits = 2 * n_orb

    hf = [0] * n_qubits
    for i in range(n_alpha):
        hf[i] = 1
    for i in range(n_beta):
        hf[n_orb + i] = 1

    configs = [list(hf)]

    # Singles (alpha)
    for i in range(n_alpha):
        for a in range(n_alpha, n_orb):
            c = list(hf)
            c[i], c[a] = 0, 1
            configs.append(c)
    # Singles (beta)
    for i in range(n_beta):
        for a in range(n_beta, n_orb):
            c = list(hf)
            c[n_orb + i], c[n_orb + a] = 0, 1
            configs.append(c)

    # Doubles (alpha-alpha)
    for i, j in combinations(range(n_alpha), 2):
        for a, b in combinations(range(n_alpha, n_orb), 2):
            c = list(hf)
            c[i], c[j], c[a], c[b] = 0, 0, 1, 1
            configs.append(c)
    # Doubles (beta-beta)
    for i, j in combinations(range(n_beta), 2):
        for a, b in combinations(range(n_beta, n_orb), 2):
            c = list(hf)
            c[n_orb + i], c[n_orb + j], c[n_orb + a], c[n_orb + b] = 0, 0, 1, 1
            configs.append(c)
    # Doubles (alpha-beta)
    for i in range(n_alpha):
        for j in range(n_beta):
            for a in range(n_alpha, n_orb):
                for b in range(n_beta, n_orb):
                    c = list(hf)
                    c[i], c[a] = 0, 1
                    c[n_orb + j], c[n_orb + b] = 0, 1
                    configs.append(c)

    t = torch.tensor(configs, dtype=torch.long, device=device)
    return torch.unique(t, dim=0)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = create_base_parser(
        "Iterative NQS + DCI+PT2 seed -> Quantum Circuit Krylov."
    )
    parser.add_argument(
        "--n-warmup-iterations",
        type=int,
        default=None,
        help="NQS warmup iterations (SQD phase).",
    )
    parser.add_argument(
        "--n-samples-per-iter",
        type=int,
        default=None,
        help="NQS samples per iteration.",
    )
    parser.add_argument(
        "--pt2-max-new",
        type=int,
        default=None,
        help="Max new configs from PT2 expansion of DCI seed.",
    )
    parser.add_argument(
        "--pt2-n-ref",
        type=int,
        default=None,
        help="Reference configs for PT2 expansion.",
    )
    parser.add_argument(
        "--max-krylov-dim",
        type=int,
        default=None,
        help="Maximum Krylov subspace dimension.",
    )
    parser.add_argument(
        "--num-trotter-steps",
        type=int,
        default=None,
        help="Trotter steps per evolution U.",
    )
    parser.add_argument(
        "--trotter-order",
        type=int,
        default=None,
        help="Suzuki-Trotter order (1 or 2).",
    )
    parser.add_argument(
        "--shots", type=int, default=None, help="Measurement shots per Krylov state."
    )
    parser.add_argument(
        "--total-evolution-time",
        type=float,
        default=None,
        help="Total evolution time T for U = e^{-iHT}.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["auto", "cudaq", "classical", "exact", "lanczos"],
        help="Sampling backend for quantum Krylov.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=None, help="Enable verbose logging."
    )
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load molecule ---
    hamiltonian, mol_info = get_molecule(config.get("molecule", "h2"), device=device)
    n_qubits = mol_info["n_qubits"]
    print(f"Molecule : {mol_info['name']}")
    print(f"Qubits   : {n_qubits}")
    print(f"Basis set: {mol_info['basis']}")
    print(f"Device   : {device}")
    print("=" * 60)

    # --- Exact energy ---
    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")
    print("-" * 60)

    t_start = time.perf_counter()

    # === Phase 1: DCI + PT2 seed ===
    print("\n[Phase 1] Generating DCI basis + PT2 expansion...")
    dci_basis = generate_dci_configs(hamiltonian, device)
    print(f"  DCI basis (pre-PT2): {dci_basis.shape[0]} configs")

    pt2_max_new = config.get("pt2_max_new", 500)
    pt2_n_ref = config.get("pt2_n_ref", 10)
    dci_pt2_basis = expand_basis_via_connections(
        dci_basis, hamiltonian, max_new=pt2_max_new, n_ref=pt2_n_ref
    )
    dci_pt2_basis = dci_pt2_basis.to(device)
    print(f"  DCI+PT2 basis      : {dci_pt2_basis.shape[0]} configs (+{dci_pt2_basis.shape[0] - dci_basis.shape[0]} from PT2)")

    dci_energy, dci_coeffs, dci_occs = gpu_solve_fermion(dci_pt2_basis, hamiltonian)
    dci_energy = float(dci_energy)
    print(f"  DCI+PT2 energy     : {dci_energy:.10f} Ha")
    print(f"  DCI+PT2 error      : {(dci_energy - exact_energy) * 1000:.4f} mHa")

    # === Phase 2: Iterative NQS warmup (SQD) ===
    n_warmup = config.get("n_warmup_iterations", 3)
    print(f"\n[Phase 2] NQS warmup ({n_warmup} iterations via SQD)...")
    warmup_config = HINQSSQDConfig(
        n_iterations=n_warmup,
        n_samples_per_iter=config.get("n_samples_per_iter", 5000),
        n_batches=config.get("n_batches", 5),
        max_configs_per_batch=config.get("max_configs_per_batch", 5000),
        energy_tol=config.get("energy_tol", 1e-5),
        nqs_lr=config.get("nqs_lr", 1e-3),
        nqs_train_epochs=config.get("nqs_train_epochs", 50),
        embed_dim=config.get("embed_dim", 64),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 4),
        temperature=config.get("temperature", 1.0),
        device=device,
    )
    warmup_result = run_hi_nqs_sqd(hamiltonian, mol_info, config=warmup_config)
    print(f"  Warmup energy  : {warmup_result.energy:.10f} Ha")
    print(f"  Warmup basis   : {warmup_result.diag_dim}")

    # === Phase 3: Quantum Circuit Krylov ===
    print("\n[Phase 3] Running Quantum Circuit Krylov (Trotterized)...")
    quantum_config = QuantumSKQDMethodConfig(
        max_krylov_dim=config.get("max_krylov_dim", 12),
        total_evolution_time=config.get("total_evolution_time", np.pi),
        num_trotter_steps=config.get("num_trotter_steps", 1),
        trotter_order=config.get("trotter_order", 2),
        shots=config.get("shots", 100_000),
        backend=config.get("backend", "auto"),
        use_gpu=device != "cpu",
        device=device,
    )
    quantum_result = run_quantum_skqd(hamiltonian, mol_info, config=quantum_config)
    print(f"  Quantum energy : {quantum_result.energy:.10f} Ha")
    print(f"  Quantum basis  : {quantum_result.diag_dim}")

    # === Combine results ===
    final_energy = min(dci_energy, warmup_result.energy, quantum_result.energy)
    wall_time = time.perf_counter() - t_start
    error_mha = (final_energy - exact_energy) * 1000.0
    within = "YES" if abs(error_mha) < CHEMICAL_ACCURACY_MHA else "NO"

    # --- Results summary ---
    print("\n" + "=" * 60)
    print("ITERATIVE NQS + DCI+PT2 -> QUANTUM KRYLOV RESULTS")
    print("=" * 60)
    print(f"DCI+PT2 energy   : {dci_energy:.10f} Ha  ({(dci_energy - exact_energy) * 1000:.4f} mHa)")
    print(f"NQS warmup energy: {warmup_result.energy:.10f} Ha  ({(warmup_result.energy - exact_energy) * 1000:.4f} mHa)")
    print(f"Quantum energy   : {quantum_result.energy:.10f} Ha  ({(quantum_result.energy - exact_energy) * 1000:.4f} mHa)")

    energies = quantum_result.metadata.get("energies_per_krylov", [])
    if energies:
        print("\n  Quantum Krylov convergence:")
        for i, e in enumerate(energies):
            step_err = (e - exact_energy) * 1000.0
            print(f"    k={i + 2:>3}: {e:.10f} Ha  (error: {step_err:.4f} mHa)")

    print(f"\nFinal energy : {final_energy:.10f} Ha")
    print(f"Exact energy : {exact_energy:.10f} Ha")
    print(f"Error        : {error_mha:.4f} mHa")
    print(f"Chemical acc.: {within}")
    print(f"Wall time    : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
