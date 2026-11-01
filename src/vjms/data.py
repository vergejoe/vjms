"""Document-level splitting happens before tokenizer fitting or token packing."""

import hashlib
import random
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .storage import Store, atomic_json, identifier, now, read_json

MAX_BYTES = 32 * 1024 * 1024
SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>"]


def import_text(store: Store, name: str, text: str, validation_fraction=0.1, seed=42):
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("第一版单次导入上限为 32MB UTF-8 文本。")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Blank lines mark documents. Fall back to lines for line-oriented corpora.
    documents = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(documents) < 2:
        documents = [p.strip() for p in text.splitlines() if p.strip()]
    documents = list(dict.fromkeys(documents))
    if len(documents) < 2:
        raise ValueError("至少需要两篇不同的文本。请用空行分隔文档，或每行提供一篇文本。")
    if not 0.05 <= validation_fraction <= 0.3:
        raise ValueError("验证集比例必须在 5% 到 30% 之间。")
    random.Random(seed).shuffle(documents)
    n_val = min(len(documents) - 1, max(1, round(len(documents) * validation_fraction)))
    splits = {"validation": documents[:n_val], "train": documents[n_val:]}
    item_id = identifier()
    path = store.path("datasets", item_id)
    path.mkdir()
    atomic_json(path / "documents.json", splits)
    meta = {
        "id": item_id, "name": name.strip()[:100] or "未命名数据集", "created_at": now(),
        "documents": len(documents), "train_documents": len(splits["train"]),
        "validation_documents": n_val, "characters": sum(map(len, documents)),
        "validation_fraction": validation_fraction, "seed": seed,
        "sha256": hashlib.sha256(text.encode()).hexdigest(), "preview": documents[0][:800],
    }
    atomic_json(path / "meta.json", meta)
    return meta


def train_tokenizer(store: Store, dataset_id: str, vocab_size: int):
    dataset = store.get("datasets", dataset_id)
    documents = read_json(store.path("datasets", dataset_id) / "documents.json")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    tok.train_from_iterator(documents["train"], trainer=trainer)
    item_id = identifier()
    path = store.path("tokenizers", item_id)
    path.mkdir()
    tok.save(str(path / "tokenizer.json"))
    counts = {}
    eos = tok.token_to_id(SPECIAL_TOKENS[1])
    for split, docs in documents.items():
        # Process one document at a time; the raw dataset is bounded by MAX_BYTES.
        with (path / f"{split}.bin").open("wb") as out:
            counts[split] = 0
            for doc in docs:
                ids = tok.encode(doc).ids + [eos]
                np.asarray(ids, dtype=np.uint16).tofile(out)
                counts[split] += len(ids)
    meta = {
        "id": item_id, "dataset_id": dataset_id, "dataset_name": dataset["name"],
        "created_at": now(), "vocab_size": tok.get_vocab_size(), "requested_vocab_size": vocab_size,
        "train_tokens": counts["train"], "validation_tokens": counts["validation"],
        "type": "byte-level BPE", "fit_split": "train", "eos_id": eos,
    }
    atomic_json(path / "meta.json", meta)
    return meta


def sample_text():
    return (Path(__file__).parent / "samples" / "bilingual.txt").read_text(encoding="utf-8")
