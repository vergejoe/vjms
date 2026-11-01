"""Local artifacts with atomic metadata writes and strictly scoped identifiers."""

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier():
    return uuid.uuid4().hex[:16]


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Store:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("VJMS_HOME", ".vjms")).resolve()
        for kind in ("datasets", "tokenizers", "runs"):
            (self.root / kind).mkdir(parents=True, exist_ok=True)

    def path(self, kind, item_id):
        if kind not in {"datasets", "tokenizers", "runs"} or not re.fullmatch(r"[a-f0-9]{16}", item_id):
            raise ValueError("无效的资源标识。")
        return self.root / kind / item_id

    def get(self, kind, item_id):
        return read_json(self.path(kind, item_id) / "meta.json")

    def list(self, kind):
        items = [read_json(p) for p in (self.root / kind).glob("*/meta.json")]
        return sorted(items, key=lambda x: x["created_at"], reverse=True)
