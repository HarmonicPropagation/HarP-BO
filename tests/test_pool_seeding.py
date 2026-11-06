import torch
from bo import _make_discrete_candidate_pool

CONFIG = {
    "dimension": 4,
    "optimization_space": "log_p_over_L__log_pL__I__inv_sqrt_tau",
    "Lzzr_bounds": (0.015, 1.5),
    "pzr_bounds": (3.0, 300.0),
    "intensity_Wcm2_bounds": (0.01e14, 10e14),
    "tau_fs_bounds": (11.0, 88.0),
}


def pool(seed, iteration):
    return _make_discrete_candidate_pool({**CONFIG, "seed": seed}, iteration, 64, torch.float64)


def test_pool_is_reproducible() -> None:
    assert torch.equal(pool(1, 5), pool(1, 5))


def test_pairs_with_equal_sum_give_different_pools() -> None:
    # With the old default_rng(seed + iteration) these pairs drew identical pools.
    reference = pool(0, 6)
    for seed, iteration in ((1, 5), (2, 4), (3, 3), (6, 0)):
        other = pool(seed, iteration)
        assert not torch.equal(reference, other), (seed, iteration)
        # No candidate point is shared between the two pools.
        assert torch.cdist(reference, other).min().item() > 1e-9, (seed, iteration)
