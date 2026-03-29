"""Tests for CUDA-QX VQE pipeline integration."""

from __future__ import annotations

import pytest

cudaq = pytest.importorskip("cudaq")
cudaq_solvers = pytest.importorskip("cudaq_solvers")


class TestCudaqVQE:
    """Tests for run_cudaq_vqe."""

    def test_import(self) -> None:
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        assert callable(run_cudaq_vqe)

    def test_h2_vqe_uccsd_reaches_chemical_accuracy(self) -> None:
        """H2 VQE-UCCSD should reach chemical accuracy (< 1.6 mHa)."""
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        result = run_cudaq_vqe(
            geometry=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))],
            basis="sto-3g",
            method="vqe",
        )
        assert result["energy"] < -1.13
        assert result["error_mha"] < 1.6

    def test_h2_adapt_vqe(self) -> None:
        """H2 ADAPT-VQE should also reach chemical accuracy."""
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        result = run_cudaq_vqe(
            geometry=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))],
            basis="sto-3g",
            method="adapt-vqe",
        )
        assert result["energy"] < -1.13
        assert result["error_mha"] < 1.6

    def test_returns_expected_keys(self) -> None:
        """Result dict should have standard keys."""
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        result = run_cudaq_vqe(
            geometry=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))],
            basis="sto-3g",
            method="vqe",
        )
        for key in [
            "energy",
            "fci_energy",
            "error_mha",
            "wall_time",
            "n_params",
            "iterations",
            "method",
        ]:
            assert key in result, f"Missing key: {key}"
