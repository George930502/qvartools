Pipeline Methods
================

qvartools provides 24 pipeline methods organized in 8 groups. Each group
combines a basis generation strategy (rows) with one of three diagonalization
modes (columns): Classical Krylov, Quantum Circuit Krylov, or SQD.

Pipeline Overview
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 12 12 51

   * - Group
     - NF Training
     - Diag Modes
     - Description
   * - 01 DCI
     - No
     - C / Q / SQD
     - Deterministic HF + singles + doubles
   * - 02 NF+DCI
     - Yes
     - C / Q / SQD
     - NF training + DCI merge
   * - 03 NF+DCI+PT2
     - Yes
     - C / Q / SQD
     - NF + DCI + perturbative expansion
   * - 04 NF-Only
     - Yes
     - C / Q / SQD
     - NF-only basis (ablation, no DCI)
   * - 05 HF-Only
     - No
     - C / Q / SQD
     - Single HF reference state (baseline)
   * - 06 Iterative NQS
     - Iterative
     - C / Q / SQD
     - Autoregressive NQS with eigenvector feedback
   * - 07 NF+DCI -> Iter NQS
     - Yes + Iterative
     - C / Q / SQD
     - NF+DCI merge then iterative NQS refinement
   * - 08 NF+DCI+PT2 -> Iter NQS
     - Yes + Iterative
     - C / Q / SQD
     - NF+DCI+PT2 then iterative NQS refinement

**Diag mode key:** C = Classical Krylov (SKQD), Q = Quantum Circuit Krylov
(Trotterized), SQD = batch diag with noise + S-CORE.

The FlowGuidedKrylovPipeline
-----------------------------

The main pipeline class orchestrates up to four stages:

**Stage 1: Train** — Joint physics-guided training of the normalizing flow and
NQS using a mixed objective (teacher KL-divergence + variational energy +
entropy regularization).

**Stage 2: Select** — Extract accumulated basis configurations from the trained
flow and apply diversity-aware selection to ensure representation across
excitation ranks.

**Stage 2.5: Expand** (Groups 03, 08 only) — Enlarge the basis via CIPSI-style
perturbative selection using Hamiltonian connections.

**Stage 3: Diagonalize** — Run Classical Krylov (SKQD), Quantum Circuit Krylov,
or SQD (batch diag) to compute the ground-state energy.

.. code-block:: python

   from qvartools import PipelineConfig, FlowGuidedKrylovPipeline
   from qvartools.molecules import get_molecule

   hamiltonian, mol_info = get_molecule("BeH2")

   config = PipelineConfig(
       skip_nf_training=False,
       subspace_mode="classical_krylov",   # "classical_krylov", "skqd", or "sqd"
       teacher_weight=0.5,
       physics_weight=0.4,
       entropy_weight=0.1,
   )

   pipeline = FlowGuidedKrylovPipeline(
       hamiltonian=hamiltonian,
       config=config,
       auto_adapt=True,  # auto-scale parameters to system size
   )

   results = pipeline.run()

Iterative Pipelines
--------------------

Groups 06-08 use an iterative loop where the ground-state eigenvector
from each diagonalization is fed back as a training target for the next NQS
iteration:

.. code-block:: python

   from qvartools.methods.nqs.hi_nqs_skqd import HINQSSKQDConfig, run_hi_nqs_skqd
   from qvartools.molecules import get_molecule

   hamiltonian, mol_info = get_molecule("H2")
   mol_info["n_orbitals"] = hamiltonian.integrals.n_orbitals
   mol_info["n_alpha"] = hamiltonian.integrals.n_alpha
   mol_info["n_beta"] = hamiltonian.integrals.n_beta

   config = HINQSSKQDConfig(
       n_iterations=10,
       n_samples_per_iter=5000,
       device="cuda",
   )

   result = run_hi_nqs_skqd(hamiltonian, mol_info, config=config)
   print(f"Energy: {result.energy:.10f} Ha")
   print(f"Converged: {result.converged}")

Running Experiment Scripts
--------------------------

All 24 pipelines live in ``experiments/pipelines/``:

.. code-block:: bash

   # Run a single pipeline
   python experiments/pipelines/01_dci/dci_krylov_classical.py h2 --device cuda

   # Run all 24 pipelines and compare
   python experiments/pipelines/run_all_pipelines.py h2 --device cuda

   # Run with a YAML config
   python experiments/pipelines/02_nf_dci/nf_dci_krylov_classical.py lih \
       --config experiments/pipelines/configs/02_nf_dci.yaml --max-epochs 200

See :doc:`yaml_configs` for details on the configuration system.
