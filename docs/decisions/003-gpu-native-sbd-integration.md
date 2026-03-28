# ADR-003: GPU-native Selected Basis Diagonalization for SQD

- **Status**: Proposed
- **Date**: 2026-03-28
- **Author**: George Chang
- **Relates to**: Issue #14, ADR-002

---

## Context

Our SQD `_diagonalize_batch` builds a dense projected Hamiltonian via
`matrix_elements` (89.6% of wall time) and diagonalizes with
`np.linalg.eigh`. This approach has two fundamental limitations:

1. **No alpha×beta Cartesian product** — subspace is `n_configs` instead
   of `|unique_alpha| × |unique_beta|`, giving worse variational energy
   than IBM's `solve_fermion`
2. **Dense H construction scales O(n²)** — infeasible for subspaces
   above ~10K configurations

Two GPU-native SBD implementations published in January 2026 solve
both problems via **matrix-free** Davidson iteration (H|v⟩ computed
on-the-fly, no dense H stored):

| Implementation | Paper | Speedup | Portability |
|---------------|-------|---------|-------------|
| r-ccs-cms/sbd v1.3.0 | arXiv:2601.16637 | 40x (H200) | NVIDIA only (Thrust) |
| AMD-HPC/amd-sbd | arXiv:2601.16169 | 95x (MI250X), 2.64x (GB200) | AMD + NVIDIA (OpenMP) |

Both are open source (Apache 2.0), tested on Fe4S4 (36 qubits,
3.6×10^7 determinants), and actively maintained. Neither has Python
bindings.

## Decision

Integrate `r-ccs-cms/sbd` as an optional GPU-accelerated SBD backend
for qvartools' SQD pipeline, using nanobind for Python bindings.

### Why r-ccs-cms/sbd over AMD-HPC/amd-sbd

- sbd is the upstream project (AMD fork is based on it)
- Thrust backend aligns with our NVIDIA-first hardware (DGX Spark)
- Header-only C++17 library — easier to embed than a full build system
- Active development (35 PRs merged, IBM + RIKEN contributing)

### Why nanobind over alternatives

| Approach | Effort | Data exchange | Iterative loop | Long-term |
|----------|--------|--------------|----------------|-----------|
| subprocess | hours | Disk I/O | Slow | No |
| ctypes + C shim | 1 week | Copy | OK | Fragile |
| **nanobind** | 1-2 weeks | **DLPack zero-copy GPU** | Fast | **Best** |
| pybind11 | 1-2 weeks | Copy or DLPack | Fast | Legacy |

nanobind is the 2026 standard for C++→Python bindings:
- 4x faster compile, 5x smaller binaries vs pybind11
- Native CUDA + DLPack support
- `nb::ndarray` with `device::cuda` for PyTorch tensor exchange
- Ref: https://nanobind.readthedocs.io/en/latest/

## Consequences

### Positive

- Matrix-free Davidson eliminates O(n²) dense H construction bottleneck
- Alpha×beta Cartesian product handled internally by sbd
- 40x GPU speedup over CPU (benchmarked on H200)
- Scales to 30+ orbitals (Fe4S4 tested at 36 qubits)
- FCIDUMP format already standard in quantum chemistry

### Negative

- New build dependency: CUDA toolkit + C++17 compiler on target machine
- nanobind wrapper adds ~500 lines of C++ binding code
- sbd requires MPI even for single-node — must strip or stub MPI
  for DGX Spark single-node use
- DGX Spark ARM64 (aarch64) compilation needs Thrust header path
  fix for CUDA 13.0 CCCL relocation (`-I${CTK_ROOT}/include/cccl`)
- No Windows support (Linux/macOS only)

### Neutral

- Existing Python-based `_diagonalize_batch` remains as fallback for
  environments without C++ toolchain
- `mixed_precision_eigh` (PR #5) remains useful for the Python path

## Implementation plan

### Phase 1: Prove integration (subprocess, ~1 day)

Compile sbd on DGX Spark. Write a Python script that:
1. Exports PySCF integrals as FCIDUMP
2. Exports NF-NQS configs as bitstring files
3. Calls sbd via `subprocess.run()`
4. Parses energy from stdout
5. Compare energy vs our `_diagonalize_batch` and IBM `solve_fermion`

Success criteria: energy matches IBM `solve_fermion` within 1e-8 Ha.

### Phase 2: nanobind wrapper (~1-2 weeks)

Write `qvartools/_ext/sbd_binding.cpp`:
1. Strip MPI for single-node (replace `MPI_COMM_WORLD` with stub)
2. Expose `diag(fcidump, alpha_strs, beta_strs, config)` → `(energy, eigenvector)`
3. Accept PyTorch tensors via DLPack for zero-copy GPU exchange
4. Build with `scikit-build-core` + nanobind + nvcc

### Phase 3: Integration into SQD pipeline (~3 days)

Wire sbd backend into `sqd.py:_diagonalize_batch`:
```python
if sbd_available and n > SBD_THRESHOLD:
    energy, eigvec = sbd_diag(integrals, alpha_strs, beta_strs)
else:
    # Existing Python path (fallback)
    ...
```

### Phase 4: Benchmark and validate (~2 days)

- Compare energy: sbd vs IBM `solve_fermion` vs our dense diag
- Compare speed: sbd GPU vs our `matrix_elements` + eigh
- Test on: H2, LiH, H2O, BeH2, N2
- Run `run_all_pipelines.py --device cuda` → 24/24

## Compilation notes for DGX Spark

```bash
# CUDA 13.0 moved Thrust headers to CCCL location
export CCCL_INCLUDE=/usr/local/cuda/include/cccl

# Compile sbd for Blackwell (sm_121)
nvcc -std=c++17 -O3 \
  -I${CCCL_INCLUDE} \
  -arch=sm_121 \
  -DSBD_THRUST \
  sbd_binding.cpp -o sbd_binding.so

# ARM64: no cross-compilation needed (native on DGX Spark)
```

## References

- r-ccs-cms/sbd: https://github.com/r-ccs-cms/sbd
- arXiv:2601.16637 — GPU-Accelerated SBD with Thrust
- arXiv:2601.16169 — OpenMP offload SQD on GPU
- AMD-HPC/amd-sbd: https://github.com/AMD-HPC/amd-sbd
- nanobind: https://nanobind.readthedocs.io/en/latest/
- DGX Spark porting guide: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/
- CUDA 13.0 Thrust CCCL relocation: https://forums.developer.nvidia.com/t/cuda-13-0-missing-thrust-folder/342958
- qiskit-addon-sqd-hpc: https://github.com/Qiskit/qiskit-addon-sqd-hpc
- Issue #14: SQD alpha×beta Cartesian product
