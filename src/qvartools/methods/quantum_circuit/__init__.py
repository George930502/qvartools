"""quantum_circuit --- Quantum circuit SKQD method pipelines.

Wires :class:`QuantumCircuitSKQD` (Trotterized ``exp_pauli`` Krylov
evolution) into end-to-end pipelines for direct comparison with the
classical exact-matrix-exponential path used by
:class:`ClassicalKrylovDiagonalization`.

Functions
---------
run_quantum_skqd
    Molecular pipeline: JW mapping -> QuantumCircuitSKQD.
run_quantum_skqd_spin
    Spin-lattice pipeline: Heisenberg model -> QuantumCircuitSKQD.
"""

from __future__ import annotations

from qvartools.methods.quantum_circuit.molecular import (
    QuantumSKQDMethodConfig,
    run_quantum_skqd,
)
from qvartools.methods.quantum_circuit.spin import (
    QuantumSKQDSpinConfig,
    run_quantum_skqd_spin,
)

__all__ = [
    "QuantumSKQDMethodConfig",
    "run_quantum_skqd",
    "QuantumSKQDSpinConfig",
    "run_quantum_skqd_spin",
]
