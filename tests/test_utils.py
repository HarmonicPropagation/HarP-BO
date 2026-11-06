import torch
from utils import norm, denorm

# Just a placeholder so far
def test_norm_denorm_roundtrip() -> None:
    # Test tensor normalization and denormalization roundtrip
    val = torch.tensor([5.0], dtype=torch.float64)
    vmin = 0.0
    vmax = 10.0

    normalized = norm(val, vmin, vmax)
    assert torch.allclose(normalized, torch.tensor([0.5], dtype=torch.float64))

    recovered = denorm(normalized, vmin, vmax)
    assert torch.allclose(recovered, val)
