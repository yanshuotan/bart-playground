"""The respacing ladder rule in diagnosis/experiment_fixed100.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diagnosis"))
from experiment_fixed100 import _equi_rejection_ladder, _pooled_rejection_curve  # noqa: E402


def _rejections(temps, density):
    """Pair rejections when rejection is the integral of `density` over beta."""
    betas = 1.0 / np.asarray(temps, dtype=float)
    grid = np.linspace(betas[-1], betas[0], 20001)
    cumulative = np.concatenate([[0.0], np.cumsum(density(grid[1:]) * np.diff(grid))])
    at = np.interp(betas, grid, cumulative)
    return -np.diff(at)


def test_respaced_ladder_meets_target_with_fewest_temperatures():
    density = lambda beta: 9.0 * beta ** 2           # barrier 3, concentrated near the cold end
    temps = np.geomspace(1.0, 100.0, 60)             # deliberately too dense
    rates = 1.0 - _rejections(temps, density)
    new_temps, barrier = _equi_rejection_ladder([(temps, rates)], 0.4)
    new_rejections = _rejections(new_temps, density)
    assert new_temps[0] == 1.0 and new_temps[-1] == 100.0
    assert np.all(np.diff(new_temps) > 0)
    assert barrier == pytest.approx(new_rejections.sum(), rel=1e-3)
    assert len(new_temps) == int(np.ceil(barrier / 0.6)) + 1
    assert len(new_temps) < len(temps)
    assert np.all(new_rejections <= 0.6 + 1e-3)
    # equal up to the linear interpolation of Lambda between the measured pairs
    assert np.ptp(new_rejections) < 0.03


def test_respacing_respects_the_cap_and_zero_rates():
    temps = [1.0, 2.0, 10.0, 100.0]
    new_temps, barrier = _equi_rejection_ladder([(temps, [0.0, 0.0, 0.5])], 0.4, max_temperatures=5)
    assert barrier == pytest.approx(2.5, abs=1e-5)
    assert len(new_temps) == 5
    assert new_temps[0] == 1.0 and new_temps[-1] == 100.0


def test_pooling_averages_ladders_and_prefers_measured_pairs():
    density = lambda beta: np.full_like(beta, 3.0)   # barrier 2.97; every pair below rejection 1
    a = np.geomspace(1.0, 100.0, 14)
    b = np.geomspace(1.0, 100.0, 20)
    rounds = [(a, 1.0 - _rejections(a, density)), (b, 1.0 - _rejections(b, density))]
    _, cumulative = _pooled_rejection_curve(rounds)
    assert cumulative[-1] == pytest.approx(3.0, rel=0.02)
    # a never-swapping pair is only a lower bound and gives way to a measured one
    coarse = ([1.0, 100.0], [0.0])
    _, alone = _pooled_rejection_curve([coarse])
    _, pooled = _pooled_rejection_curve([coarse, rounds[1]])
    assert alone[-1] == pytest.approx(1.0)
    assert pooled[-1] == pytest.approx(3.0, rel=0.02)


def test_design_rate_above_target_adds_headroom():
    density = lambda beta: 9.0 * beta ** 2
    temps = np.geomspace(1.0, 100.0, 40)
    rounds = [(temps, 1.0 - _rejections(temps, density))]
    at_target, _ = _equi_rejection_ladder(rounds, 0.4)
    with_margin, _ = _equi_rejection_ladder(rounds, 0.5)
    assert len(with_margin) > len(at_target)
    assert np.all(_rejections(with_margin, density) <= 0.5 + 0.02)
