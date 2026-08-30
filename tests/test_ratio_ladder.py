from hairgs_hybrid.plan_animals60_ratio_ladder import (
    neutral_group_count,
    ratio_budget,
)


def test_ratio_budget_means_fraction_removed():
    assert ratio_budget(400_000, 0.75) == 100_000
    assert ratio_budget(400_000, 0.875) == 50_000
    assert ratio_budget(400_000, 0.9375) == 25_000


def test_neutral_group_count_is_feasible_and_centered():
    source = 700_455
    target = ratio_budget(source, 0.75)
    groups = neutral_group_count(source, target)
    assert groups <= target <= 4 * groups
    assert groups <= source // 4
    assert abs(7 * groups / 3 - target) <= 2
