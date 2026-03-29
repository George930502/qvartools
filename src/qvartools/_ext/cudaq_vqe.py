"""CUDA-QX VQE and ADAPT-VQE pipeline wrapper.

Provides GPU-accelerated VQE using NVIDIA CUDA-Q + CUDA-QX Solvers.
Supports VQE with UCCSD ansatz and ADAPT-VQE with spin-complement GSD
operator pool.

Usage::

    from qvartools._ext.cudaq_vqe import run_cudaq_vqe
    result = run_cudaq_vqe(
        geometry=[("H", (0., 0., 0.)), ("H", (0., 0., 0.7474))],
        basis="sto-3g",
        method="adapt-vqe",
    )
    print(result["energy"], result["error_mha"])

Requires: cudaq >= 0.14, cudaq-solvers >= 0.5
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def run_cudaq_vqe(
    geometry: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    method: str = "vqe",
    optimizer: str = "cobyla",
    max_iterations: int = 200,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run VQE or ADAPT-VQE using CUDA-QX Solvers on GPU.

    Parameters
    ----------
    geometry : list of (str, (float, float, float))
        Molecular geometry in Angstroms.
    basis : str
        Gaussian basis set (default ``"sto-3g"``).
    charge : int
        Net charge (default ``0``).
    spin : int
        Spin multiplicity minus one (default ``0`` for singlet).
    method : str
        ``"vqe"`` for VQE-UCCSD or ``"adapt-vqe"`` for ADAPT-VQE.
    optimizer : str
        Optimizer name (default ``"cobyla"``).
    max_iterations : int
        Maximum optimizer iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Keys: ``energy``, ``fci_energy``, ``hf_energy``, ``error_mha``,
        ``wall_time``, ``n_params``, ``iterations``, ``method``,
        ``n_qubits``, ``n_electrons``.
    """
    import cudaq
    import cudaq_solvers as solvers

    cudaq.set_target("nvidia")

    # Create molecule (uses PySCF internally, no openfermion)
    molecule = solvers.create_molecule(
        geometry=geometry,
        basis=basis,
        spin=spin,
        charge=charge,
        casci=True,
    )

    n_qubits = molecule.n_orbitals * 2
    n_electrons = molecule.n_electrons
    hamiltonian = molecule.hamiltonian

    hf_energy = molecule.energies.get("hf_energy", None)
    fci_energy = molecule.energies.get("fci_energy", None)

    logger.info(
        "CUDA-QX %s: %d qubits, %d electrons, basis=%s",
        method,
        n_qubits,
        n_electrons,
        basis,
    )

    t0 = time.perf_counter()

    if method == "adapt-vqe":
        energy, params, n_params, iterations = _run_adapt_vqe(
            molecule, n_qubits, n_electrons, hamiltonian, verbose
        )
    else:
        energy, params, n_params, iterations = _run_vqe_uccsd(
            n_qubits, n_electrons, hamiltonian, optimizer, max_iterations, verbose
        )

    wall_time = time.perf_counter() - t0

    error_mha = abs(energy - fci_energy) * 1000 if fci_energy is not None else None

    return {
        "energy": energy,
        "fci_energy": fci_energy,
        "hf_energy": hf_energy,
        "error_mha": error_mha,
        "wall_time": wall_time,
        "n_params": n_params,
        "iterations": iterations,
        "method": method,
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "optimal_parameters": params,
    }


def _run_vqe_uccsd(
    n_qubits, n_electrons, hamiltonian, optimizer, max_iterations, verbose
):
    """VQE with UCCSD ansatz."""
    import cudaq
    import cudaq_solvers as solvers

    num_params = solvers.stateprep.get_num_uccsd_parameters(n_electrons, n_qubits)

    # Need to capture these in closure for the kernel
    _nq = n_qubits
    _ne = n_electrons

    @cudaq.kernel
    def uccsd_kernel(thetas: list[float]):
        q = cudaq.qvector(_nq)
        for i in range(_ne):
            x(q[i])
        solvers.stateprep.uccsd(q, thetas, _ne, 0)

    energy, params, data = solvers.vqe(
        uccsd_kernel,
        hamiltonian,
        initial_parameters=[0.0] * num_params,
        optimizer=optimizer,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    return energy, list(params), num_params, len(data)


def _run_adapt_vqe(molecule, n_qubits, n_electrons, hamiltonian, verbose):
    """ADAPT-VQE with spin-complement GSD operator pool."""
    import cudaq
    import cudaq_solvers as solvers

    operators = solvers.get_operator_pool(
        "spin_complement_gsd",
        num_orbitals=molecule.n_orbitals,
    )

    @cudaq.kernel
    def initial_state(q: cudaq.qview):
        for i in range(n_electrons):
            x(q[i])

    energy, params, ops = solvers.adapt_vqe(
        initial_state,
        hamiltonian,
        operators,
        verbose=verbose,
    )

    return energy, list(params), len(params), len(ops)
