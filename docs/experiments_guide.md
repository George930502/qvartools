# Experiments Guide

This guide covers the 24 experiment pipeline scripts in `experiments/pipelines/`. Each pipeline combines a basis generation strategy with a diagonalization method. All pipelines can be configured via YAML files with CLI overrides.

---

## Pipeline Overview

All pipelines follow the same pattern:
1. Load a molecule from the registry
2. Compute the exact FCI energy for reference
3. Run the method-specific pipeline stages
4. Report energy, error vs exact, and timing

### Common Arguments

All scripts accept:
- Positional `molecule` argument (default: `h2`)
- `--config` flag pointing to a YAML configuration file
- `--device` flag (`cpu`, `cuda`, or `auto`)
- Pipeline-specific flags (see `--help`)

---

## Pipeline Groups

### Group 01: Direct-CI (no NF training)

Generates HF + singles + doubles deterministically, then diagonalizes.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `01_dci/dci_krylov_classical.py` | Classical Krylov | DCI -> SKQD time evolution |
| `01_dci/dci_krylov_quantum.py` | Quantum Krylov | DCI -> Trotterized circuit evolution |
| `01_dci/dci_sqd.py` | SQD | DCI -> noise + S-CORE batch diag |

### Group 02: NF-NQS + DCI Merge

Trains a normalizing flow, merges NF-sampled basis with Direct-CI essentials.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `02_nf_dci/nf_dci_krylov_classical.py` | Classical Krylov | NF+DCI merge -> SKQD |
| `02_nf_dci/nf_dci_krylov_quantum.py` | Quantum Krylov | NF+DCI merge -> Trotterized |
| `02_nf_dci/nf_dci_sqd.py` | SQD | NF+DCI merge -> noise + S-CORE |

### Group 03: NF + DCI + PT2 Expansion

Same as Group 02, plus CIPSI-style perturbative basis expansion via Hamiltonian connections.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `03_nf_dci_pt2/nf_dci_pt2_krylov_classical.py` | Classical Krylov | NF+DCI+PT2 -> SKQD |
| `03_nf_dci_pt2/nf_dci_pt2_krylov_quantum.py` | Quantum Krylov | NF+DCI+PT2 -> Trotterized |
| `03_nf_dci_pt2/nf_dci_pt2_sqd.py` | SQD | NF+DCI+PT2 -> noise + S-CORE |

### Group 04: NF-Only (Ablation)

NF training without DCI scaffolding. Tests pure NF generative power.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `04_nf_only/nf_krylov_classical.py` | Classical Krylov | NF-only -> SKQD |
| `04_nf_only/nf_krylov_quantum.py` | Quantum Krylov | NF-only -> Trotterized |
| `04_nf_only/nf_sqd.py` | SQD | NF-only -> noise + S-CORE |

### Group 05: HF-Only (Baseline)

Minimal baseline starting from a single Hartree-Fock reference state.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `05_hf_only/hf_krylov_classical.py` | Classical Krylov | HF -> Krylov discovers configs |
| `05_hf_only/hf_krylov_quantum.py` | Quantum Krylov | HF -> Trotterized circuit |
| `05_hf_only/hf_sqd.py` | SQD | HF -> noise + S-CORE |

### Group 06: Iterative NQS

Iterative autoregressive transformer NQS with eigenvector feedback.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `06_iterative_nqs/iter_nqs_krylov_classical.py` | Classical Krylov | NQS loop + H-connection expansion |
| `06_iterative_nqs/iter_nqs_krylov_quantum.py` | Quantum Krylov | NQS warmup + Trotterized |
| `06_iterative_nqs/iter_nqs_sqd.py` | SQD | NQS loop + batch diag |

### Group 07: NF + DCI Merge -> Iterative NQS

NF training and DCI merge (same as Group 02 stages 1-2), then iterative NQS refinement.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `07_iterative_nqs_dci/iter_nqs_dci_krylov_classical.py` | Classical Krylov | NF+DCI -> iterative NQS+Krylov |
| `07_iterative_nqs_dci/iter_nqs_dci_krylov_quantum.py` | Quantum Krylov | NF+DCI -> quantum Krylov |
| `07_iterative_nqs_dci/iter_nqs_dci_sqd.py` | SQD | NF+DCI -> iterative NQS+SQD |

### Group 08: NF + DCI + PT2 -> Iterative NQS

NF training, DCI merge, and PT2 expansion (same as Group 03 stages 1-2.5), then iterative NQS.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `08_iterative_nqs_dci_pt2/iter_nqs_dci_pt2_krylov_classical.py` | Classical Krylov | NF+DCI+PT2 -> iterative NQS+Krylov |
| `08_iterative_nqs_dci_pt2/iter_nqs_dci_pt2_krylov_quantum.py` | Quantum Krylov | NF+DCI+PT2 -> quantum Krylov |
| `08_iterative_nqs_dci_pt2/iter_nqs_dci_pt2_sqd.py` | SQD | NF+DCI+PT2 -> iterative NQS+SQD |

---

## Running All Pipelines

```bash
# Run all 24 pipelines on H2 and compare
python experiments/pipelines/run_all_pipelines.py h2 --device cuda

# Run only specific groups
python experiments/pipelines/run_all_pipelines.py h2 --only 01 02 04

# Skip quantum pipelines (no CUDA-Q needed)
python experiments/pipelines/run_all_pipelines.py h2 --skip-quantum

# Skip slow iterative pipelines
python experiments/pipelines/run_all_pipelines.py h2 --skip-iterative

# Save results to JSON
python experiments/pipelines/run_all_pipelines.py lih --output results.json
```

## Chemical Accuracy Threshold

All experiments compare results against **1.6 milliHartree (mHa)**, the conventional definition of chemical accuracy.

## Prerequisites

- `pyscf` must be installed for molecular integrals and FCI/CCSD
- GPU experiments require CUDA-enabled PyTorch
- Quantum Krylov pipelines require `cudaq`
- Large molecules (N2, CH4, C2H4) may take several minutes on CPU
