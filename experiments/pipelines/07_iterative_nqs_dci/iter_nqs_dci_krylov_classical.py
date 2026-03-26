"""Iterative NQS + DCI seed -> Krylov classical expansion.

Pipeline:
  Phase 1: Generate DCI basis (HF + singles + doubles) deterministically,
           diagonalise to get seed energy.
  Phase 2: Run iterative NQS + SKQD (Krylov expansion via H-connections).
  Phase 3: Report best energy across DCI seed and iterative NQS+SKQD.

The DCI seed provides a deterministic lower-bound reference; the iterative
NQS loop explores beyond the CISD manifold using autoregressive sampling
and Krylov basis growth.
"""

from __future__ import annotations

import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config_loader import create_base_parser, load_config

from qvartools._utils.gpu.diagnostics import gpu_solve_fermion
from qvartools.methods.nqs.hi_nqs_skqd import HINQSSKQDConfig, run_hi_nqs_skqd
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
        "Iterative NQS + DCI seed -> Krylov classical expansion."
    )
    parser.add_argument(
        "--n-iterations", type=int, default=None, help="NQS outer iterations."
    )
    parser.add_argument(
        "--n-samples-per-iter",
        type=int,
        default=None,
        help="NQS samples per iteration.",
    )
    parser.add_argument(
        "--nqs-train-epochs",
        type=int,
        default=None,
        help="NQS training epochs per iteration.",
    )
    parser.add_argument(
        "--krylov-max-new",
        type=int,
        default=None,
        help="Max new configs from Krylov expansion per iteration.",
    )
    parser.add_argument(
        "--krylov-n-ref",
        type=int,
        default=None,
        help="Reference configs for Krylov expansion.",
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

    # === Phase 1: DCI seed ===
    print("\n[Phase 1] Generating DCI basis (HF + singles + doubles)...")
    dci_basis = generate_dci_configs(hamiltonian, device)
    dci_energy, dci_coeffs, dci_occs = gpu_solve_fermion(dci_basis, hamiltonian)
    dci_energy = float(dci_energy)
    print(f"  DCI basis size : {dci_basis.shape[0]}")
    print(f"  DCI seed energy: {dci_energy:.10f} Ha")
    print(f"  DCI error      : {(dci_energy - exact_energy) * 1000:.4f} mHa")

    # === Phase 2: Iterative NQS + Krylov ===
    print("\n[Phase 2] Running iterative NQS + Krylov (SKQD)...")
    skqd_config = HINQSSKQDConfig(
        n_iterations=config.get("n_iterations", 10),
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
        krylov_max_new=config.get("krylov_max_new", 500),
        krylov_n_ref=config.get("krylov_n_ref", 10),
        device=device,
    )
    nqs_result = run_hi_nqs_skqd(hamiltonian, mol_info, config=skqd_config)

    # === Phase 3: Combine results ===
    final_energy = min(dci_energy, nqs_result.energy)
    wall_time = time.perf_counter() - t_start
    error_mha = (final_energy - exact_energy) * 1000.0
    within = "YES" if abs(error_mha) < CHEMICAL_ACCURACY_MHA else "NO"

    # --- Results summary ---
    print("\n" + "=" * 60)
    print("ITERATIVE NQS + DCI -> KRYLOV CLASSICAL RESULTS")
    print("=" * 60)
    print(f"DCI seed energy  : {dci_energy:.10f} Ha")
    print(f"DCI basis size   : {dci_basis.shape[0]}")
    print(f"NQS+SKQD energy  : {nqs_result.energy:.10f} Ha")
    print(f"NQS+SKQD basis   : {nqs_result.diag_dim}")
    print(f"NQS+SKQD iters   : {nqs_result.metadata.get('n_iterations', '?')}")
    print(f"NQS converged    : {nqs_result.converged}")

    energy_history = nqs_result.metadata.get("energy_history", [])
    if energy_history:
        print("\n  NQS+SKQD energy convergence:")
        for i, e in enumerate(energy_history):
            err = (e - exact_energy) * 1000.0
            print(f"    iter {i + 1:>3}: {e:.10f} Ha  (error: {err:.4f} mHa)")

    print(f"\nFinal energy : {final_energy:.10f} Ha")
    print(f"Exact energy : {exact_energy:.10f} Ha")
    print(f"Error        : {error_mha:.4f} mHa")
    print(f"Chemical acc.: {within}")
    print(f"Wall time    : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
