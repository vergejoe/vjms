import shutil

import torch

from vjms.data import import_text, sample_text, train_tokenizer
from vjms.model import ModelConfig
from vjms.schemas import Architecture, RunRequest
from vjms.storage import Store, atomic_json, identifier, now
from vjms.training import load_checkpoint, train


def prepare(store, tok, steps):
    run_id = identifier()
    path = store.path("runs", run_id)
    path.mkdir()
    request = RunRequest(tokenizer_id=tok["id"], name="test", architecture=Architecture(
        dim=32, layers=1, heads=2, hidden_dim=64, context_length=16), device="cpu", max_steps=steps,
        batch_size=2, gradient_accumulation=1, eval_interval=2, checkpoint_interval=2, eval_batches=1)
    config = request.model_dump(exclude={"architecture"})
    config["model"] = ModelConfig(vocab_size=tok["vocab_size"], **request.architecture.model_dump()).to_dict()
    atomic_json(path / "config.json", config)
    atomic_json(path / "meta.json", {"id": run_id, "name": "test", "created_at": now(), "status": "queued", "step": 0})
    shutil.copyfile(store.path("tokenizers", tok["id"]) / "tokenizer.json", path / "tokenizer.json")
    return run_id, path, config


def test_resumed_training_matches_uninterrupted_optimizer_and_rng(tmp_path):
    store = Store(tmp_path)
    data = import_text(store, "sample", sample_text())
    tok = train_tokenizer(store, data["id"], 512)
    full_id, full_path, _ = prepare(store, tok, 4)
    train(str(tmp_path), full_id)
    resumed_id, resumed_path, config = prepare(store, tok, 2)
    train(str(tmp_path), resumed_id)
    config["max_steps"] = 4
    atomic_json(resumed_path / "config.json", config)
    train(str(tmp_path), resumed_id, resume=True)
    a, b = load_checkpoint(full_path / "latest.pt"), load_checkpoint(resumed_path / "latest.pt")
    assert a["step"] == b["step"] == 4
    assert a["tokens_seen"] == b["tokens_seen"] == 128
    assert torch.equal(a["sample_rng"], b["sample_rng"])
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    for state_id, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][state_id][key], atol=0, rtol=0)
