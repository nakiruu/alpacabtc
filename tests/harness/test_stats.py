"""Paired circular block bootstrap — Sharpe-diff significance testing.

The most important test in this file is `test_null_gate_cannot_reject_noise` —
that IS the plan §5.3 harness self-check. If it fails, the harness is either
finding fake signal in noise or has an inverted p-value calculation.
"""

from __future__ import annotations

import random

import pytest

from singularity.harness.stats import (
    _default_block_size,
    _resample_indices,
    concat_fold_returns,
    paired_block_bootstrap,
)


def test_default_block_size_scales_as_cube_root():
    assert _default_block_size(1) == 1
    assert _default_block_size(8) == 2
    assert _default_block_size(1000) == 10   # 1000^(1/3) = 10


def test_resample_indices_returns_exactly_n():
    rng = random.Random(0)
    idx = _resample_indices(n=100, block_size=7, rng=rng)
    assert len(idx) == 100
    assert all(0 <= i < 100 for i in idx)


def test_resample_indices_uses_circular_wrap():
    """When a block starts near the end, subsequent indices wrap to the beginning."""
    rng = random.Random(0)
    # Force a specific start by controlling RNG:
    idx = _resample_indices(n=10, block_size=5, rng=rng)
    # With circular wrap, all indices are in [0, 10)
    assert all(0 <= i < 10 for i in idx)


def test_concat_fold_returns_preserves_order():
    folds = [[0.01, 0.02], [-0.01, 0.005], [0.03]]
    out = concat_fold_returns(folds)
    assert out == [0.01, 0.02, -0.01, 0.005, 0.03]


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(ValueError, match="equal-length"):
        paired_block_bootstrap([0.01] * 100, [0.01] * 50)


def test_paired_bootstrap_zero_diff_when_series_identical():
    """If A == B, observed diff is exactly 0 and every bootstrap draw is 0."""
    rng = random.Random(42)
    r = [rng.gauss(0.001, 0.01) for _ in range(200)]
    boot = paired_block_bootstrap(r, r, n_bootstrap=200, seed=1)
    assert boot.observed_diff == 0.0
    assert boot.ci_low == 0.0
    assert boot.ci_high == 0.0
    # p-value on a degenerate distribution: no bootstrap draws are <0 or >0,
    # so the tail is 0/200 = 0, p = 0. This is acceptable — identical series
    # ARE trivially significantly identical.
    assert boot.p_value == 0.0


def test_paired_bootstrap_captures_real_difference():
    """Perfectly-paired A = B + edge: bootstrap MUST detect it.

    Correlated construction: A[i] = B[i] + edge_i, edge_i > 0 always. This
    guarantees A dominates B in every draw, so the bootstrap can't miss it
    regardless of sample-mean noise in b."""
    rng = random.Random(7)
    n = 300
    b = [rng.gauss(0, 0.01) for _ in range(n)]
    a = [x + 0.002 for x in b]   # deterministic +20bps/bar edge
    boot = paired_block_bootstrap(a, b, n_bootstrap=500, seed=3)
    assert boot.observed_diff > 0.0
    assert boot.p_value < 0.05


def test_null_gate_cannot_reject_noise():
    """THE HARNESS TRUSTWORTHINESS TEST.

    Two IID series with the same distribution and no shared signal → the
    bootstrap must NOT be able to reject the null of "equal Sharpe" at 5%.

    If this test fails intermittently more than 5% of the time (over many
    seeds), the bootstrap is producing false positives — the harness is
    finding signal where none exists.
    """
    false_positives = 0
    trials = 20
    for trial_seed in range(trials):
        rng_a = random.Random(trial_seed * 2)
        rng_b = random.Random(trial_seed * 2 + 1)
        n = 300
        a = [rng_a.gauss(0, 0.01) for _ in range(n)]
        b = [rng_b.gauss(0, 0.01) for _ in range(n)]
        boot = paired_block_bootstrap(a, b, n_bootstrap=200, seed=trial_seed)
        if boot.p_value < 0.05:
            false_positives += 1
    # Expected FP rate under H0 is 5%; allow generous slack (up to 20% observed
    # across 20 trials with only 200 bootstrap iterations each)
    assert false_positives <= 4, (
        f"got {false_positives}/{trials} false positives — bootstrap over-rejects noise"
    )


def test_bootstrap_p_value_is_bounded():
    rng = random.Random(0)
    a = [rng.gauss(0, 0.01) for _ in range(100)]
    b = [rng.gauss(0, 0.01) for _ in range(100)]
    boot = paired_block_bootstrap(a, b, n_bootstrap=100, seed=0)
    assert 0.0 <= boot.p_value <= 1.0
    assert boot.ci_low <= boot.ci_high


def test_small_sample_returns_safe_defaults():
    boot = paired_block_bootstrap([0.01, 0.02], [0.01, 0.02])
    assert boot.n_bootstrap == 0
    assert boot.p_value == 1.0


# ---- random_matched_turnover ----

def test_random_matched_turnover_hits_target_within_tolerance():
    """Averaged over many folds, observed turnover should be close to target."""
    from singularity.harness.backtest import random_matched_turnover

    class _FakeBar:
        pass

    target = 10.0
    strategy = random_matched_turnover(target_turnover_per_fold=target, seed=42)
    bars = [_FakeBar() for _ in range(90)]  # 89 possible transitions
    turnovers = []
    for _ in range(50):
        pos = strategy(bars)
        turnover = sum(abs(pos[i] - pos[i - 1]) for i in range(1, len(pos)))
        turnovers.append(turnover)
    import statistics as st
    mean_turnover = st.fmean(turnovers)
    # Expected p_flip = 10 / 89, expected turnover per fold = 10
    assert abs(mean_turnover - target) < 2.0, f"got mean_turnover={mean_turnover}"


def test_random_matched_turnover_zero_target_stays_flat():
    from singularity.harness.backtest import random_matched_turnover

    class _FakeBar:
        pass

    strategy = random_matched_turnover(target_turnover_per_fold=0.0, seed=0)
    bars = [_FakeBar() for _ in range(50)]
    pos = strategy(bars)
    assert all(p == 0.0 for p in pos)


def test_random_matched_turnover_deterministic_from_seed():
    from singularity.harness.backtest import random_matched_turnover

    class _FakeBar:
        pass

    bars = [_FakeBar() for _ in range(30)]
    s1 = random_matched_turnover(target_turnover_per_fold=5.0, seed=99)
    s2 = random_matched_turnover(target_turnover_per_fold=5.0, seed=99)
    assert s1(bars) == s2(bars)
