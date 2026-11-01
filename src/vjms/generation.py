import time

import torch
from tokenizers import Tokenizer

from .hardware import resolve_device
from .model import LanguageModel, ModelConfig
from .storage import Store
from .training import load_checkpoint


def generate(store: Store, request):
    path = store.path("runs", request.run_id)
    checkpoint_path = path / f"{request.checkpoint}.pt"
    if not checkpoint_path.exists():
        raise ValueError("所选检查点尚不存在，请先完成至少一次验证并保存检查点。")
    device = resolve_device(request.device)
    saved = load_checkpoint(checkpoint_path)
    model = LanguageModel(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["model"])
    del saved
    model = model.to(device).eval()
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    eos = tokenizer.token_to_id("<|endoftext|>")
    pad = tokenizer.token_to_id("<|pad|>")
    prompt_ids = tokenizer.encode(request.prompt).ids
    context = (prompt_ids or [eos])[-model.config.context_length:]
    tokens = torch.tensor([context], dtype=torch.long, device=device)
    output_ids = []
    # Sample on CPU for consistent backend support (including MPS generators).
    rng = torch.Generator().manual_seed(request.seed)
    started = time.monotonic()
    finish_reason = "length"
    with torch.inference_mode():
        for _ in range(request.max_new_tokens):
            logits, _ = model(tokens[:, -model.config.context_length:])
            scores = logits[0, -1].float()
            if request.repetition_penalty != 1:
                for token in set(context + output_ids):
                    scores[token] = torch.where(scores[token] < 0, scores[token] * request.repetition_penalty,
                                                 scores[token] / request.repetition_penalty)
            scores[pad] = float("-inf")
            if request.temperature == 0:
                token = int(scores.argmax())
            else:
                scores = scores / request.temperature
                if request.top_k:
                    threshold = torch.topk(scores, min(request.top_k, len(scores))).values[-1]
                    scores[scores < threshold] = float("-inf")
                if request.top_p < 1:
                    sorted_scores, indices = torch.sort(scores, descending=True)
                    remove = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1) > request.top_p
                    remove[1:] = remove[:-1].clone()
                    remove[0] = False
                    scores[indices[remove]] = float("-inf")
                token = int(torch.multinomial(torch.softmax(scores, dim=-1).cpu(), 1, generator=rng))
            if token == eos:
                finish_reason = "stop"
                break
            output_ids.append(token)
            tokens = torch.cat([tokens, torch.tensor([[token]], device=device)], dim=1)
    elapsed = time.monotonic() - started
    completion = tokenizer.decode(output_ids, skip_special_tokens=True)
    return {
        "prompt": request.prompt, "completion": completion, "text": request.prompt + completion,
        "prompt_tokens": len(prompt_ids), "context_tokens": len(context), "generated_tokens": len(output_ids),
        "seconds": elapsed, "tokens_per_second": len(output_ids) / max(elapsed, 1e-6),
        "device": device, "finish_reason": finish_reason,
    }
