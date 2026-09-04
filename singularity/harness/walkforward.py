"""Plan §5.1 — non-anchored rolling walk-forward.

    12mo train / 3mo validation / 3mo test, advance 3mo per fold.

Non-anchored: the train window SLIDES forward each fold rather than growing.
That gives every fold the same statistical weight and prevents late-period
folds from being dominated by a large training sample.

Fold indices are computed against a list of bars (any granularity). "Months"
are converted to bar counts using `bars_per_month` (approx 30 for daily bars,
30*24 for hourly, etc). Not calendar-exact, but the plan's ~27-fold count is
approximate anyway and this keeps the splitter dependency-free.

Warm-up buffer: the plan warns that any feature engineering must live INSIDE
each fold and start from the train window; here that means callers should
compute features from bars[fold.train_start_idx:] and discard the first
`warmup_bars` from what they use, so no data before train_start bleeds in.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    index: int
    train_start_idx: int
    train_end_idx: int      # exclusive
    val_start_idx: int
    val_end_idx: int        # exclusive
    test_start_idx: int
    test_end_idx: int       # exclusive

    @property
    def train_len(self) -> int:
        return self.train_end_idx - self.train_start_idx

    @property
    def val_len(self) -> int:
        return self.val_end_idx - self.val_start_idx

    @property
    def test_len(self) -> int:
        return self.test_end_idx - self.test_start_idx


@dataclass(frozen=True)
class WalkForwardSpec:
    """All windows in bar counts. Advance is how far the train window slides per fold."""
    train_bars: int
    val_bars: int
    test_bars: int
    advance_bars: int

    @classmethod
    def default_daily(cls) -> "WalkForwardSpec":
        # 12 / 3 / 3 months on daily bars, advance 3 months
        return cls(train_bars=12 * 30, val_bars=3 * 30, test_bars=3 * 30, advance_bars=3 * 30)

    @property
    def fold_span_bars(self) -> int:
        return self.train_bars + self.val_bars + self.test_bars


class WalkForwardSplitter:
    def __init__(self, spec: WalkForwardSpec | None = None) -> None:
        self.spec = spec or WalkForwardSpec.default_daily()

    def folds(self, n_bars: int) -> list[Fold]:
        """Enumerate every fold that fits fully within `n_bars`."""
        s = self.spec
        out: list[Fold] = []
        i = 0
        train_start = 0
        while True:
            train_end = train_start + s.train_bars
            val_end = train_end + s.val_bars
            test_end = val_end + s.test_bars
            if test_end > n_bars:
                break
            out.append(Fold(
                index=i,
                train_start_idx=train_start, train_end_idx=train_end,
                val_start_idx=train_end,     val_end_idx=val_end,
                test_start_idx=val_end,      test_end_idx=test_end,
            ))
            i += 1
            train_start += s.advance_bars
        return out
