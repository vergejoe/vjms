"""Optional real-device smoke test; random data is not a model-quality benchmark."""

import argparse
import json
import time

import torch

from vjms.hardware import resolve_device
from vjms.model import LanguageModel, ModelConfig, PRESETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--preset", choices=PRESETS, default="base")
    args = parser.parse_args()
    device = resolve_device(args.device)
    config = ModelConfig(vocab_size=4096, context_length=64, **PRESETS[args.preset])
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = LanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    started = time.monotonic()
    for _ in range(2):
        tokens = torch.randint(0, config.vocab_size, (1, 65)).to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(tokens[:, :-1], tokens[:, 1:])
        loss.backward()
        assert torch.isfinite(loss), "non-finite loss"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    print(json.dumps({"device": device, "parameters": config.parameters, "steps": 2,
                      "loss": float(loss.detach()), "seconds": round(time.monotonic() - started, 3),
                      "note": "Random-data FP32 training smoke test, not a quality benchmark."}, indent=2))


if __name__ == "__main__":
    main()
