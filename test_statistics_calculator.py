
import pytest

from statistics_calculator import descriptive_stats


def test_descriptive_stats_rejects_empty_dataset():
    with pytest.raises(ValueError, match="Dataset is empty"):
        descriptive_stats([])


def test_descriptive_stats_rejects_fewer_than_four_values():
    with pytest.raises(ValueError, match="at least 4 data points"):
        descriptive_stats([10, 20, 30])


def test_descriptive_stats_returns_correct_count_and_mean():
    result = descriptive_stats([10, 20, 30, 40])

    assert result["count"] == 4
    assert result["mean"] == 25
