import pytest
from tokenizers import Tokenizer

from vjms.data import import_text, sample_text, train_tokenizer
from vjms.storage import Store, read_json


def test_split_is_deduplicated_reproducible_and_tokenizer_roundtrips(tmp_path):
    store = Store(tmp_path)
    text = sample_text()
    first = import_text(store, "test", text + "\n\n" + text, seed=4)
    second = import_text(store, "test2", text, seed=4)
    a = read_json(store.path("datasets", first["id"]) / "documents.json")
    b = read_json(store.path("datasets", second["id"]) / "documents.json")
    assert a == b
    assert set(a["train"]).isdisjoint(a["validation"])
    assert first["documents"] == 40
    meta = train_tokenizer(store, first["id"], 4096)
    tok = Tokenizer.from_file(str(store.path("tokenizers", meta["id"]) / "tokenizer.json"))
    for sample in ["中文 English 123", "🦊 café Ελληνικά", "前导 空格\n下一行"]:
        assert tok.decode(tok.encode(sample).ids) == sample
    assert meta["fit_split"] == "train"
    assert meta["validation_tokens"] > 16
    for split in ["train", "validation"]:
        assert (store.path("tokenizers", meta["id"]) / f"{split}.bin").stat().st_size == meta[f"{split}_tokens"] * 2


def test_tokenizer_never_fits_validation_documents(tmp_path):
    store = Store(tmp_path)
    dataset = import_text(store, "test", "alpha alpha alpha alpha\n\nvalidationonlyword validationonlyword validationonlyword", seed=42, validation_fraction=.3)
    documents = read_json(store.path("datasets", dataset["id"]) / "documents.json")
    meta = train_tokenizer(store, dataset["id"], 1024)
    raw = read_json(store.path("tokenizers", meta["id"]) / "tokenizer.json")
    validation_word = documents["validation"][0].split()[0]
    assert validation_word not in raw["model"]["vocab"]


def test_reject_empty_repeated_document_and_path_traversal(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        import_text(store, "empty", "\n")
    with pytest.raises(ValueError):
        import_text(store, "duplicate", "same\n\nsame")
    with pytest.raises(ValueError):
        store.path("runs", "../../etc/passwd")
