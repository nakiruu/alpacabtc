"""WalkForwardSplitter — fold enumeration semantics."""

from __future__ import annotations

import pytest

from singularity.harness.walkforward import WalkForwardSpec, WalkForwardSplitter


def test_default_spec_daily_is_12_3_3_months():
    s = WalkForwardSpec.default_daily()
    assert s.train_bars == 360
    assert s.val_bars == 90
    assert s.test_bars == 90
    assert s.advance_bars == 90
    assert s.fold_span_bars == 540


def test_no_folds_when_insufficient_bars():
    splitter = WalkForwardSplitter()
    assert splitter.folds(100) == []
    assert splitter.folds(539) == []


def test_one_fold_at_exact_span():
    splitter = WalkForwardSplitter()
    folds = splitter.folds(540)
    assert len(folds) == 1
    f = folds[0]
    assert f.train_start_idx == 0
    assert f.train_end_idx == 360
    assert f.val_start_idx == 360
    assert f.val_end_idx == 450
    assert f.test_start_idx == 450
    assert f.test_end_idx == 540


def test_folds_advance_by_advance_bars():
    splitter = WalkForwardSplitter()
    # 540 + 90 = 630 bars → exactly 2 folds
    folds = splitter.folds(630)
    assert len(folds) == 2
    assert folds[1].train_start_idx == 90    # advanced by 90
    assert folds[1].test_end_idx == 630


def test_non_anchored_train_window_slides_not_grows():
    splitter = WalkForwardSplitter()
    folds = splitter.folds(900)
    for f in folds:
        assert f.train_len == 360  # constant, not growing


def test_custom_spec_bar_counts():
    spec = WalkForwardSpec(train_bars=10, val_bars=2, test_bars=2, advance_bars=2)
    splitter = WalkForwardSplitter(spec)
    folds = splitter.folds(20)
    # First fold uses bars 0..14; second 2..16; third 4..18; fourth 6..20 → 4 folds
    assert len(folds) == 4
    assert folds[0].train_start_idx == 0
    assert folds[-1].test_end_idx == 20


def test_test_windows_do_not_overlap_train_of_same_fold():
    splitter = WalkForwardSplitter()
    for f in splitter.folds(1800):
        assert f.test_start_idx >= f.train_end_idx
        assert f.test_start_idx >= f.val_end_idx
