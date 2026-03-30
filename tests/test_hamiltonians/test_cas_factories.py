"""Tests for CAS molecule factory functions.

TDD Red phase for Issue #21 Phase 1 — Cr₂, N₂-CAS, Benzene factories.
"""

from __future__ import annotations

import pytest

pyscf = pytest.importorskip("pyscf")

from qvartools.hamiltonians.molecular.hamiltonian import MolecularHamiltonian

# ---------------------------------------------------------------------------
# Tests: N₂-CAS factory
# ---------------------------------------------------------------------------


@pytest.mark.pyscf
@pytest.mark.slow
class TestN2CASFactory:
    """create_n2_cas_hamiltonian should produce a valid CAS Hamiltonian."""

    def test_import_and_call(self):
        from qvartools.molecules.registry import get_molecule

        ham, info = get_molecule("N2-CAS(10,12)", device="cpu")
        assert isinstance(ham, MolecularHamiltonian)

    def test_n2_cas_10_12_has_12_orbitals(self):
        from qvartools.molecules.registry import get_molecule

        ham, info = get_molecule("N2-CAS(10,12)", device="cpu")
        assert ham.integrals.n_orbitals == 12

    def test_n2_cas_10_12_is_cas_flag(self):
        from qvartools.molecules.registry import get_molecule

        _, info = get_molecule("N2-CAS(10,12)", device="cpu")
        assert info.get("is_cas") is True


# ---------------------------------------------------------------------------
# Tests: Cr₂ factory
# ---------------------------------------------------------------------------


@pytest.mark.pyscf
@pytest.mark.slow
class TestCr2Factory:
    """create_cr2_hamiltonian should produce singlet CAS Hamiltonian."""

    def test_cr2_cas_12_12_creates_hamiltonian(self):
        from qvartools.molecules.registry import get_molecule

        ham, info = get_molecule("Cr2", device="cpu")
        assert isinstance(ham, MolecularHamiltonian)
        assert ham.integrals.n_orbitals == 12

    def test_cr2_singlet_not_septet(self):
        """Cr₂ CASSCF must converge to singlet, not septet.

        Without fix_spin_(ss=0), CASSCF converges to septet (S=3).
        Singlet e_core ≈ -2035.4 Ha.  If fix_spin_ is removed, the
        CASSCF converges to a different state with different e_core.
        """
        from qvartools.molecules.registry import get_molecule

        ham, info = get_molecule("Cr2", device="cpu")
        e_core = ham.integrals.nuclear_repulsion
        # Singlet e_core for Cr₂ STO-3G CAS(12,12) is ~-2035.4 Ha.
        # Septet would give a substantially different value.
        assert -2040.0 < e_core < -2030.0, (
            f"e_core={e_core:.4f} outside singlet range [-2040, -2030]. "
            f"CASSCF may have converged to septet (fix_spin_ missing?)."
        )


# ---------------------------------------------------------------------------
# Tests: Benzene factory
# ---------------------------------------------------------------------------


@pytest.mark.pyscf
@pytest.mark.slow
class TestBenzeneFactory:
    """create_benzene_hamiltonian should produce CAS(6,15) Hamiltonian."""

    def test_benzene_creates_hamiltonian(self):
        from qvartools.molecules.registry import get_molecule

        ham, info = get_molecule("Benzene", device="cpu")
        assert isinstance(ham, MolecularHamiltonian)

    def test_benzene_has_15_orbitals(self):
        from qvartools.molecules.registry import get_molecule

        ham, info = get_molecule("Benzene", device="cpu")
        assert ham.integrals.n_orbitals == 15

    def test_benzene_is_cas(self):
        from qvartools.molecules.registry import get_molecule

        _, info = get_molecule("Benzene", device="cpu")
        assert info.get("is_cas") is True


# ---------------------------------------------------------------------------
# Tests: Registry completeness
# ---------------------------------------------------------------------------


class TestCASRegistryCompleteness:
    """Verify all CAS molecules are registered."""

    def test_cas_molecules_in_registry(self):
        from qvartools.molecules.registry import MOLECULE_REGISTRY

        expected_cas = [
            "N2-CAS(10,12)",
            "N2-CAS(10,15)",
            "N2-CAS(10,17)",
            "N2-CAS(10,20)",
            "N2-CAS(10,26)",
            "Cr2",
            "Cr2-CAS(12,18)",
            "Cr2-CAS(12,20)",
            "Cr2-CAS(12,26)",
            "Cr2-CAS(12,28)",
            "Cr2-CAS(12,29)",
            "Benzene",
        ]
        registry_keys = {k.lower() for k in MOLECULE_REGISTRY}
        for name in expected_cas:
            assert name.lower() in registry_keys, f"Missing CAS molecule: {name}"

    def test_is_cas_flag_present(self):
        from qvartools.molecules.registry import MOLECULE_REGISTRY

        for name, entry in MOLECULE_REGISTRY.items():
            if "cas" in name or name in ("cr2", "benzene"):
                assert entry.get("is_cas") is True, f"{name} missing is_cas=True"
