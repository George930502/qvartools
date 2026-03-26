"""Classical vs Quantum Circuit SKQD --- Side-by-side comparison.

Runs both Krylov basis generators on the same molecule with matched
parameters and reports the energy, error, basis size, and wall time
for each.

Comparison:
    - Classical: exact matrix_exp (no Trotter error, no shot noise)
    - Quantum:   Trotterized exp_pauli (Trotter error + shot noise)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import create_base_parser, load_config

from qvartools.methods.quantum_circuit.compare import (
    ComparisonConfig,
    compare_krylov_generators,
)
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = create_base_parser(
        "Classical vs Quantum Circuit SKQD side-by-side comparison."
    )
    parser.add_argument(
        "--max-krylov-dim",
        type=int,
        default=None,
        help="Krylov subspace dimension (both generators).",
    )
    parser.add_argument(
        "--num-trotter-steps",
        type=int,
        default=None,
        help="Trotter steps for quantum generator.",
    )
    parser.add_argument(
        "--trotter-order",
        type=int,
        default=None,
        help="Suzuki-Trotter order (1 or 2).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Measurement shots per Krylov state.",
    )
    parser.add_argument(
        "--total-evolution-time",
        type=float,
        default=None,
        help="Total evolution time T.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["auto", "cudaq", "classical", "exact", "lanczos"],
        help="Quantum generator backend.",
    )
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load molecule ---
    hamiltonian, mol_info = get_molecule(config.get("molecule", "h2"), device=device)
    print(f"Molecule : {mol_info['name']}")
    print(f"Qubits   : {mol_info['n_qubits']}")
    print(f"Device   : {device}")

    # --- Exact energy ---
    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")

    # --- Run comparison ---
    comp_config = ComparisonConfig(
        max_krylov_dim=config.get("max_krylov_dim", 10),
        total_evolution_time=config.get("total_evolution_time", np.pi),
        shots=config.get("shots", 100_000),
        num_trotter_steps=config.get("num_trotter_steps", 8),
        trotter_order=config.get("trotter_order", 2),
        backend=config.get("backend", "auto"),
        use_gpu=device != "cpu",
    )

    results = compare_krylov_generators(
        hamiltonian, mol_info, exact_energy, config=comp_config
    )

    # --- Final verdict ---
    summary = results["summary"]
    c_err = abs(results["classical"]["error_mha"])
    q_err = abs(results["quantum"]["error_mha"])

    print(f"\nClassical chemical accuracy: {'YES' if c_err < CHEMICAL_ACCURACY_MHA else 'NO'}")
    print(f"Quantum   chemical accuracy: {'YES' if q_err < CHEMICAL_ACCURACY_MHA else 'NO'}")

    if summary["speedup"] > 1:
        print(f"Classical is {summary['speedup']:.1f}x faster")
    else:
        print(f"Quantum is {1 / summary['speedup']:.1f}x faster")


if __name__ == "__main__":
    main()
