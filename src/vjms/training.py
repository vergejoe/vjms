"""One training subprocess; checkpoints include optimizer and sampling RNG states."""

import argparse
import contextlib
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .hardware import resolve_device
from .model import LanguageModel, ModelConfig
from .storage import Store, atomic_json, now, read_json


def load_checkpoint(path):
    # Only locally generated checkpoints are supported. Never load external pickles.
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def batch(tokens, batch_size, length, generator, device):
    starts = torch.randint(len(tokens) - length, (batch_size,), generator=generator)
    packed = np.stack([tokens[int(i):int(i) + length + 1] for i in starts]).astype(np.int64)
    packed = torch.from_numpy(packed).to(device)
    return packed[:, :-1].contiguous(), packed[:, 1:].contiguous()


def train(root, run_id, resume=False):
    store = Store(root)
    path = store.path("runs", run_id)
    cfg = read_json(path / "config.json")
    meta = store.get("runs", run_id)
    step = meta.get("step", 0)
    model = optimizer = scaler = None
    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        device = resolve_device(cfg["device"])
        torch.manual_seed(cfg["seed"])
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = LanguageModel(ModelConfig(**cfg["model"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp_dtype == torch.float16))
        sample_rng = torch.Generator().manual_seed(cfg["seed"])
        best_loss = float("inf")
        tokens_seen = 0
        elapsed_before = 0.0
        if resume:
            saved = load_checkpoint(path / "latest.pt")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            if scaler.is_enabled() and saved["scaler"]:
                scaler.load_state_dict(saved["scaler"])
            sample_rng.set_state(saved["sample_rng"])
            torch.set_rng_state(saved["torch_rng"])
            step, best_loss, tokens_seen = saved["step"], saved["best_loss"], saved["tokens_seen"]
            elapsed_before = saved.get("elapsed_seconds", 0.0)
        source = store.path("tokenizers", cfg["tokenizer_id"])
        train_tokens = np.memmap(source / "train.bin", dtype=np.uint16, mode="r")
        val_tokens = np.memmap(source / "validation.bin", dtype=np.uint16, mode="r")
        length = cfg["model"]["context_length"]
        if min(len(train_tokens), len(val_tokens)) <= length:
            raise ValueError("训练集或验证集 token 数不足。请增加文本或缩短上下文长度。")
        meta.update(status="running", device=device, pid=os.getpid(), error=None, updated_at=now(), step=step,
                    precision=str(amp_dtype).replace("torch.", "") if device == "cuda" else "float32")
        atomic_json(path / "meta.json", meta)
        started = time.monotonic()

        def autocast():
            return torch.autocast("cuda", dtype=amp_dtype) if device == "cuda" else contextlib.nullcontext()

        def checkpoint():
            return {
                "format_version": 1, "model": model.state_dict(), "model_config": cfg["model"],
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "best_loss": best_loss, "tokens_seen": tokens_seen,
                "sample_rng": sample_rng.get_state(), "torch_rng": torch.get_rng_state(),
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
            }

        def evaluate():
            model.eval()
            rng = torch.Generator().manual_seed(cfg["seed"] + 1)
            losses = []
            with torch.inference_mode():
                for _ in range(cfg["eval_batches"]):
                    x, y = batch(val_tokens, cfg["batch_size"], length, rng, device)
                    with autocast():
                        _, loss = model(x, y)
                    losses.append(float(loss))
            model.train()
            result = sum(losses) / len(losses)
            if not math.isfinite(result):
                raise ValueError("验证损失不是有限数，请降低学习率并检查数据。")
            return result

        def persist_status(status="running"):
            meta.update(status=status, step=step, tokens_seen=tokens_seen,
                        elapsed_seconds=elapsed_before + time.monotonic() - started,
                        best_val_loss=best_loss if math.isfinite(best_loss) else None, updated_at=now())
            atomic_json(path / "meta.json", meta)

        model.train()
        while step < cfg["max_steps"]:
            if (path / "pause.request").exists():
                save_checkpoint(path / "latest.pt", checkpoint())
                persist_status("paused")
                return
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            lr = cfg["learning_rate"] * min(1, (step + 1) / max(1, cfg["warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_total = 0.0
            for _ in range(cfg["gradient_accumulation"]):
                x, y = batch(train_tokens, cfg["batch_size"], length, sample_rng, device)
                with autocast():
                    _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise ValueError("训练损失不是有限数，请降低学习率或改用 FP32。")
                loss_total += float(loss.detach()) / cfg["gradient_accumulation"]
                scaler.scale(loss / cfg["gradient_accumulation"]).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                # FP16 scaler can recover from overflow without changing model weights.
                if not scaler.is_enabled():
                    raise ValueError("梯度出现非有限数，请降低学习率。")
            scaler.step(optimizer)
            scaler.update()
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            step += 1
            processed = cfg["batch_size"] * length * cfg["gradient_accumulation"]
            tokens_seen += processed
            train_seconds = time.monotonic() - tick
            val_loss = None
            improved = False
            if step == 1 or step % cfg["eval_interval"] == 0 or step == cfg["max_steps"]:
                val_loss = evaluate()
                improved = val_loss < best_loss
                best_loss = min(best_loss, val_loss)
            metric = {
                "step": step, "train_loss": loss_total, "val_loss": val_loss, "learning_rate": lr,
                "tokens_per_second": processed / max(train_seconds, 1e-6), "tokens_seen": tokens_seen,
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
                "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else
                    torch.mps.current_allocated_memory() / 1024**2 if device == "mps" else None,
            }
            if improved or step % cfg["checkpoint_interval"] == 0 or step == cfg["max_steps"]:
                saved = checkpoint()
                save_checkpoint(path / "latest.pt", saved)
                if improved:
                    save_checkpoint(path / "best.pt", saved)
            with (path / "metrics.jsonl").open("a", encoding="utf-8") as out:
                import json
                out.write(json.dumps(metric, allow_nan=False) + "\n")
            meta["last_metric"] = metric
            persist_status()
        persist_status("completed")
    except Exception as exc:
        meta.update(status="failed", error=str(exc), updated_at=now())
        atomic_json(path / "meta.json", meta)
        (path / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from filelock import FileLock
    with FileLock(str(Path(args.root) / "training.lock"), timeout=0):
        train(args.root, args.run, args.resume)
