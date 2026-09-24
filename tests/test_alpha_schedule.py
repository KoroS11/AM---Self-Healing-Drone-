import pytest
from train_pipeline import AlphaSchedule

class TestAlphaSchedule:
    def test_zero_at_iteration_zero(self):
        sched = AlphaSchedule(alpha_max=0.5, k_anneal=1000)
        assert sched(0) == 0.0  # exact, not approx — feeds the zero-effect control

    def test_reaches_max_at_k_anneal(self):
        sched = AlphaSchedule(alpha_max=0.5, k_anneal=1000)
        assert sched(1000) == pytest.approx(0.5)

    def test_flat_after_k_anneal(self):
        sched = AlphaSchedule(alpha_max=0.5, k_anneal=1000)
        assert sched(1500) == pytest.approx(0.5)

    def test_monotonic_nondecreasing(self):
        sched = AlphaSchedule(alpha_max=0.5, k_anneal=1000)
        vals = [sched(k) for k in range(0, 1001, 50)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_midpoint_is_half(self):
        sched = AlphaSchedule(alpha_max=0.5, k_anneal=1000)
        assert sched(500) == pytest.approx(0.25)

    def test_k_anneal_zero_returns_max_immediately(self):
        sched = AlphaSchedule(alpha_max=0.5, k_anneal=0)
        assert sched(0) == 0.5
