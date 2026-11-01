import pytest
import torch

from vjms.model import LanguageModel, ModelConfig, PRESETS


def test_exact_parameter_budget_and_presets():
    for preset in PRESETS.values():
        config = ModelConfig(vocab_size=4096, **preset)
        model = LanguageModel(config)
        assert sum(p.numel() for p in model.parameters()) == config.parameters
        assert config.parameters <= 20_000_000
    with pytest.raises(ValueError, match="20M"):
        ModelConfig(vocab_size=16384, **PRESETS["base"])
    with pytest.raises(ValueError, match="偶数"):
        ModelConfig(vocab_size=258, dim=33, heads=3)


def test_attention_cannot_see_future_and_backward_is_finite():
    torch.manual_seed(1)
    model = LanguageModel(ModelConfig(vocab_size=258, dim=32, layers=2, heads=2, hidden_dim=64, context_length=16))
    a = torch.randint(0, 258, (2, 16))
    b = a.clone()
    b[:, 8:] = (b[:, 8:] + 17) % 258
    logits_a, loss = model(a, a)
    logits_b, _ = model(b)
    torch.testing.assert_close(logits_a[:, :8], logits_b[:, :8], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(logits_a[:, 9:], logits_b[:, 9:])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
