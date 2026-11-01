"""Serialize training and inference; use child processes on Windows and macOS."""

import json
import os
import shutil
import subprocess
import sys
import threading
from collections import deque

from .hardware import resolve_device
from .model import ModelConfig
from .storage import Store, atomic_json, identifier, now, read_json


class Jobs:
    def __init__(self, store: Store):
        self.store = store
        self.lock = threading.Lock()
        self.processes = {}
        self.logs = {}
        # A single server owns the workspace lock (acquired by the app lifespan).
        # If its previous child survived a server crash, do not start a second worker.
        self.orphans = {}
        import psutil
        for meta in store.list("runs"):
            if meta["status"] in {"queued", "running", "pausing"}:
                process_file = store.path("runs", meta["id"]) / "process.json"
                pid = read_json(process_file).get("pid") if process_file.exists() else meta.get("pid")
                try:
                    proc = psutil.Process(pid) if pid else None
                    cmd = proc.cmdline() if proc else []
                    if proc and "vjms.training" in cmd and meta["id"] in cmd:
                        self.orphans[meta["id"]] = proc
                        continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                meta.update(status="interrupted", error="服务上次退出时训练被中断；可从 latest 检查点恢复。", updated_at=now())
                atomic_json(store.path("runs", meta["id"]) / "meta.json", meta)

    def reap(self):
        import psutil
        for run_id, proc in list(self.processes.items()):
            if proc.poll() is None:
                # Terminal metadata is published before Python finishes releasing
                # the model. Wait briefly so an immediate Resume is reliable.
                if self.store.get("runs", run_id)["status"] in {"paused", "completed", "failed"}:
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        continue
                else:
                    continue
            self.logs.pop(run_id).close()
            del self.processes[run_id]
            self._mark_interrupted(run_id)
        for run_id, proc in list(self.orphans.items()):
            try:
                alive = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                alive = False
            if not alive:
                del self.orphans[run_id]
                self._mark_interrupted(run_id)

    def _mark_interrupted(self, run_id):
        meta = self.store.get("runs", run_id)
        if meta["status"] in {"queued", "running", "pausing"}:
            meta.update(status="interrupted", error="训练进程已退出；可从最近的检查点恢复。", updated_at=now())
            atomic_json(self.store.path("runs", run_id) / "meta.json", meta)

    def ensure_idle(self):
        self.reap()
        if self.processes or self.orphans:
            raise RuntimeError("当前有训练任务运行。请先暂停，释放设备后再执行此操作。")

    def _launch(self, run_id, resume=False):
        path = self.store.path("runs", run_id)
        log = (path / "worker.log").open("a", encoding="utf-8")
        command = [sys.executable, "-m", "vjms.training", "--root", str(self.store.root), "--run", run_id]
        if resume:
            command.append("--resume")
        try:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except Exception:
            log.close()
            meta = self.store.get("runs", run_id)
            meta.update(status="failed", error="无法启动训练进程。", updated_at=now())
            atomic_json(path / "meta.json", meta)
            raise
        self.processes[run_id], self.logs[run_id] = proc, log
        atomic_json(path / "process.json", {"pid": proc.pid})

    def create(self, request):
        with self.lock:
            self.ensure_idle()
            tok = self.store.get("tokenizers", request.tokenizer_id)
            model = ModelConfig(vocab_size=tok["vocab_size"], **request.architecture.model_dump())
            resolve_device(request.device)
            if min(tok["train_tokens"], tok["validation_tokens"]) <= model.context_length:
                raise ValueError("训练集和验证集各自必须多于上下文长度的 token 数；请增加数据或缩短上下文。")
            run_id = identifier()
            path = self.store.path("runs", run_id)
            path.mkdir()
            config = request.model_dump(exclude={"architecture"})
            config["model"] = model.to_dict()
            atomic_json(path / "config.json", config)
            shutil.copyfile(self.store.path("tokenizers", request.tokenizer_id) / "tokenizer.json", path / "tokenizer.json")
            meta = {
                "id": run_id, "name": request.name.strip() or "新训练实验", "created_at": now(), "updated_at": now(),
                "status": "queued", "step": 0, "max_steps": request.max_steps, "parameters": model.parameters,
                "tokenizer_id": request.tokenizer_id, "dataset_name": tok["dataset_name"], "error": None,
            }
            atomic_json(path / "meta.json", meta)
            self._launch(run_id)
            return meta

    def pause(self, run_id):
        with self.lock:
            self.reap()
            if run_id not in self.processes and run_id not in self.orphans:
                raise ValueError("该任务当前没有运行。")
            path = self.store.path("runs", run_id)
            (path / "pause.request").touch()
            return {"status": "pausing", "message": "将在当前训练步结束后保存检查点并暂停。"}

    def resume(self, run_id, request):
        with self.lock:
            self.ensure_idle()
            path = self.store.path("runs", run_id)
            if not (path / "latest.pt").exists():
                raise ValueError("尚无可恢复的检查点，请创建新任务。")
            resolve_device(request.device)
            from .training import load_checkpoint
            saved = load_checkpoint(path / "latest.pt")
            step = saved["step"]
            del saved
            # Discard metrics beyond the durable checkpoint before replaying steps.
            metrics = read_metrics(path, limit=None)
            with (path / "metrics.jsonl").open("w", encoding="utf-8") as out:
                for metric in metrics:
                    if metric["step"] <= step:
                        out.write(json.dumps(metric) + "\n")
            cfg = read_json(path / "config.json")
            cfg.update(device=request.device, max_steps=step + request.additional_steps)
            atomic_json(path / "config.json", cfg)
            meta = self.store.get("runs", run_id)
            meta.update(status="queued", step=step, max_steps=cfg["max_steps"], error=None, updated_at=now())
            atomic_json(path / "meta.json", meta)
            (path / "pause.request").unlink(missing_ok=True)
            self._launch(run_id, resume=True)
            return meta

    def shutdown(self):
        for run_id in [*self.processes, *self.orphans]:
            (self.store.path("runs", run_id) / "pause.request").touch()
        for run_id, proc in list(self.processes.items()):
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            self.logs.pop(run_id).close()
        self.processes.clear()


def read_metrics(path, limit=2000):
    metrics = deque(maxlen=limit)
    file = path / "metrics.jsonl"
    if file.exists():
        with file.open(encoding="utf-8") as source:
            for line in source:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Reader may see the worker's final, partially written line.
    return list(metrics)
