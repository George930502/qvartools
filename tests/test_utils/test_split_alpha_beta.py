"""Tests for alpha/beta string splitting and Cartesian product expansion."""

from __future__ import annotations

import torch


class TestSplitSpinStrings:
    """Tests for split_spin_strings."""

    def test_import(self) -> None:
        from qvartools._utils.formatting.bitstring_format import split_spin_strings

        assert callable(split_spin_strings)

    def test_basic_split(self) -> None:
        """Split configs with all-unique alpha and beta."""
        from qvartools._utils.formatting.bitstring_format import split_spin_strings

        # All distinct alpha AND beta
        configs = torch.tensor(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ]
        )
        alpha, beta = split_spin_strings(configs, n_orbitals=2)
        assert alpha.shape == (2, 2)
        assert beta.shape == (2, 2)

    def test_unique_strings(self) -> None:
        """Duplicate alpha/beta should be removed."""
        from qvartools._utils.formatting.bitstring_format import split_spin_strings

        configs = torch.tensor(
            [
                [1, 0, 1, 0],  # alpha=[1,0], beta=[1,0]
                [1, 0, 0, 1],  # alpha=[1,0], beta=[0,1] ← same alpha
                [0, 1, 1, 0],  # alpha=[0,1], beta=[1,0] ← same beta as first
            ]
        )
        alpha, beta = split_spin_strings(configs, n_orbitals=2)
        # alpha should be unique: [1,0] and [0,1] → 2 unique
        assert len(alpha) == 2
        # beta should be unique: [1,0] and [0,1] → 2 unique
        assert len(beta) == 2

    def test_n_orbitals_inferred(self) -> None:
        """n_orbitals defaults to num_sites // 2."""
        from qvartools._utils.formatting.bitstring_format import split_spin_strings

        configs = torch.tensor([[1, 0, 0, 1, 1, 0]])
        alpha, beta = split_spin_strings(configs)
        assert alpha.shape[1] == 3
        assert beta.shape[1] == 3

    def test_empty_input(self) -> None:
        """Empty configs should return empty alpha and beta."""
        from qvartools._utils.formatting.bitstring_format import split_spin_strings

        configs = torch.zeros(0, 4, dtype=torch.long)
        alpha, beta = split_spin_strings(configs, n_orbitals=2)
        assert len(alpha) == 0
        assert len(beta) == 0

    def test_preserves_device(self) -> None:
        """Output should be on same device as input."""
        from qvartools._utils.formatting.bitstring_format import split_spin_strings

        configs = torch.tensor([[1, 0, 1, 0]])
        alpha, beta = split_spin_strings(configs, n_orbitals=2)
        assert alpha.device == configs.device
        assert beta.device == configs.device


class TestCartesianProductConfigs:
    """Tests for cartesian_product_configs."""

    def test_import(self) -> None:
        from qvartools._utils.formatting.bitstring_format import (
            cartesian_product_configs,
        )

        assert callable(cartesian_product_configs)

    def test_basic_product(self) -> None:
        """2 alpha × 2 beta = 4 configs."""
        from qvartools._utils.formatting.bitstring_format import (
            cartesian_product_configs,
        )

        alpha = torch.tensor([[1, 0], [0, 1]])
        beta = torch.tensor([[1, 0], [0, 1]])
        result = cartesian_product_configs(alpha, beta)
        assert result.shape == (4, 4)
        # All 4 combinations: [1,0,1,0], [1,0,0,1], [0,1,1,0], [0,1,0,1]
        assert len(torch.unique(result, dim=0)) == 4

    def test_asymmetric_counts(self) -> None:
        """3 alpha × 2 beta = 6 configs."""
        from qvartools._utils.formatting.bitstring_format import (
            cartesian_product_configs,
        )

        alpha = torch.tensor([[1, 0], [0, 1], [1, 1]])
        beta = torch.tensor([[1, 0], [0, 1]])
        result = cartesian_product_configs(alpha, beta)
        assert result.shape == (6, 4)

    def test_single_alpha(self) -> None:
        """1 alpha × 3 beta = 3 configs."""
        from qvartools._utils.formatting.bitstring_format import (
            cartesian_product_configs,
        )

        alpha = torch.tensor([[1, 0]])
        beta = torch.tensor([[1, 0], [0, 1], [1, 1]])
        result = cartesian_product_configs(alpha, beta)
        assert result.shape == (3, 4)
        # All should share the same alpha part
        assert torch.all(result[:, :2] == torch.tensor([1, 0]))

    def test_empty_input(self) -> None:
        """Empty alpha → empty result."""
        from qvartools._utils.formatting.bitstring_format import (
            cartesian_product_configs,
        )

        alpha = torch.zeros(0, 2, dtype=torch.long)
        beta = torch.tensor([[1, 0]])
        result = cartesian_product_configs(alpha, beta)
        assert result.shape[0] == 0

    def test_preserves_device(self) -> None:
        """Output on same device as input."""
        from qvartools._utils.formatting.bitstring_format import (
            cartesian_product_configs,
        )

        alpha = torch.tensor([[1, 0]])
        beta = torch.tensor([[0, 1]])
        result = cartesian_product_configs(alpha, beta)
        assert result.device == alpha.device
