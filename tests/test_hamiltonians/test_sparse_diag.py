"""Tests for sparse Hamiltonian construction and raised dense limit.

These tests verify Issue #21 Phase 2:
- The dense config limit in ``matrix_elements_fast()`` is raised from 10K to 50K.
- A new ``build_sparse_hamiltonian()`` method returns a ``scipy.sparse.coo_matrix``.
- Sparse eigenvalues match dense eigenvalues within tight tolerance.
- ``gpu_solve_fermion()`` falls back to sparse path for large bases.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.sparse

# ---------------------------------------------------------------------------
# PySCF availability
# ---------------------------------------------------------------------------

try:
    import pyscf  # noqa: F401

    _HAS_PYSCF = True
except ImportError:
    _HAS_PYSCF = False

pyscf_required = pytest.mark.skipif(not _HAS_PYSCF, reason="PySCF is not installed")


# ---------------------------------------------------------------------------
# Test 1: Dense limit raised from 10K to 50K
# ---------------------------------------------------------------------------


class TestDenseLimitRaised:
    """Verify ``matrix_elements_fast`` now allows up to 50K configs."""

    def test_limit_is_50k_in_source(self):
        """The guard in ``matrix_elements_fast`` should refuse at 50K, not 10K."""
        from qvartools.hamiltonians.molecular.hamiltonian import MolecularHamiltonian

        source = inspect.getsource(MolecularHamiltonian.matrix_elements_fast)
        # The old limit of 10000 should no longer appear as the threshold
        assert "n_configs > 10000" not in source, (
            "matrix_elements_fast still uses the old 10K limit"
        )
        # The new limit of 50000 should be present
        assert "50000" in source, "matrix_elements_fast should use a 50K limit"

    @pyscf_required
    def test_matrix_elements_fast_15k_no_memory_error(self, h2_hamiltonian):
        """Calling with 15K configs should NOT raise MemoryError anymore.

        We verify by checking the source code guard, since constructing
        a real 15K-config basis for H₂ (only 4 qubits) is not meaningful.
        This test documents the intent: 15K < 50K, so it should be allowed.
        """
        # The H₂ system has only 4 qubits → at most 16 configs.
        # We cannot create 15K real configs, but we verify the limit is now 50K
        # by checking that 10001 configs would NOT trip the guard (only 50001+ would).
        from qvartools.hamiltonians.molecular.hamiltonian import MolecularHamiltonian

        source = inspect.getsource(MolecularHamiltonian.matrix_elements_fast)
        assert "n_configs > 50000" in source


# ---------------------------------------------------------------------------
# Test 2: build_sparse_hamiltonian returns COO matrix
# ---------------------------------------------------------------------------


class TestBuildSparseHamiltonian:
    """Verify the new ``build_sparse_hamiltonian`` method."""

    @pyscf_required
    def test_returns_coo_matrix(self, h2_hamiltonian):
        """``build_sparse_hamiltonian`` should return ``scipy.sparse.coo_matrix``."""
        ham = h2_hamiltonian
        configs = ham._generate_all_configs()
        H_sparse = ham.build_sparse_hamiltonian(configs)
        assert isinstance(H_sparse, scipy.sparse.coo_matrix)

    @pyscf_required
    def test_sparse_shape_matches_configs(self, h2_hamiltonian):
        """Sparse matrix shape should be ``(n_configs, n_configs)``."""
        ham = h2_hamiltonian
        configs = ham._generate_all_configs()
        n = configs.shape[0]
        H_sparse = ham.build_sparse_hamiltonian(configs)
        assert H_sparse.shape == (n, n)

    @pyscf_required
    def test_sparse_is_symmetric(self, h2_hamiltonian):
        """The sparse Hamiltonian should be Hermitian (symmetric for real)."""
        ham = h2_hamiltonian
        configs = ham._generate_all_configs()
        H_sparse = ham.build_sparse_hamiltonian(configs)
        H_dense = H_sparse.toarray()
        np.testing.assert_allclose(
            H_dense,
            H_dense.T,
            atol=1e-12,
            err_msg="Sparse Hamiltonian is not symmetric",
        )


# ---------------------------------------------------------------------------
# Test 3: Sparse eigenvalues match dense eigenvalues
# ---------------------------------------------------------------------------


class TestSparseEigenvaluesMatchDense:
    """Compare eigenvalues from sparse and dense Hamiltonian construction."""

    @pyscf_required
    def test_ground_state_energy_matches(self, h2_hamiltonian):
        """Lowest eigenvalue from sparse H should match dense H within 1e-10."""
        ham = h2_hamiltonian
        configs = ham._generate_all_configs()

        # Dense path (existing)
        H_dense = ham.matrix_elements_fast(configs)
        H_dense_np = H_dense.detach().cpu().numpy()
        evals_dense = np.linalg.eigh(H_dense_np)[0]

        # Sparse path (new)
        H_sparse = ham.build_sparse_hamiltonian(configs)
        H_sparse_dense = H_sparse.toarray()
        evals_sparse = np.linalg.eigh(H_sparse_dense)[0]

        np.testing.assert_allclose(
            evals_sparse[0],
            evals_dense[0],
            atol=1e-10,
            err_msg="Sparse ground-state energy does not match dense",
        )

    @pyscf_required
    def test_full_spectrum_matches(self, h2_hamiltonian):
        """All eigenvalues from sparse H should match dense H within 1e-10."""
        ham = h2_hamiltonian
        configs = ham._generate_all_configs()

        H_dense = ham.matrix_elements_fast(configs)
        H_dense_np = H_dense.detach().cpu().numpy()
        evals_dense = np.sort(np.linalg.eigh(H_dense_np)[0])

        H_sparse = ham.build_sparse_hamiltonian(configs)
        H_sparse_dense = H_sparse.toarray()
        evals_sparse = np.sort(np.linalg.eigh(H_sparse_dense)[0])

        np.testing.assert_allclose(
            evals_sparse,
            evals_dense,
            atol=1e-10,
            err_msg="Sparse spectrum does not match dense spectrum",
        )


# ---------------------------------------------------------------------------
# Test 4: gpu_solve_fermion sparse fallback
# ---------------------------------------------------------------------------


class TestGpuSolveFermionSparseFallback:
    """Verify ``gpu_solve_fermion`` uses sparse path for large bases."""

    def test_sparse_threshold_in_diagnostics(self):
        """``diagnostics.py`` should reference ``build_sparse_hamiltonian``."""
        from qvartools._utils.gpu import diagnostics

        source = inspect.getsource(diagnostics.gpu_solve_fermion)
        assert "build_sparse_hamiltonian" in source, (
            "gpu_solve_fermion should call build_sparse_hamiltonian "
            "for large config counts"
        )

    @pyscf_required
    def test_small_system_still_works(self, h2_hamiltonian):
        """gpu_solve_fermion should still work for small systems (H₂)."""
        from qvartools._utils.gpu.diagnostics import gpu_solve_fermion

        ham = h2_hamiltonian
        configs = ham._generate_all_configs()
        energy, eigvec, (occ_a, occ_b) = gpu_solve_fermion(configs, ham)

        # H₂ ground state energy should be around -1.85 Ha (with nuclear repulsion)
        assert isinstance(energy, float)
        assert energy < 0.0
        assert eigvec.shape[0] == configs.shape[0]
