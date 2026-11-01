"""Loopback-only API and buildless local studio."""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from filelock import FileLock, Timeout
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .data import MAX_BYTES, import_text, sample_text, train_tokenizer
from .generation import generate
from .hardware import hardware_info
from .jobs import Jobs, read_metrics
from .model import PRESETS, ModelConfig
from .schemas import CompletionRequest, DatasetRequest, GenerateRequest, ResumeRequest, RunRequest, TokenizerRequest
from .storage import Store, read_json

STATIC = Path(__file__).parent / "static"


def create_app(root=None):
    store = Store(root)

    @asynccontextmanager
    async def lifespan(app):
        workspace_lock = FileLock(str(store.root / "studio.lock"))
        try:
            workspace_lock.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError("此工作目录已有 vjms 服务运行，请关闭旧服务或指定其他 --data-dir。") from exc
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        jobs = Jobs(store)
        app.state.jobs = jobs
        try:
            yield
        finally:
            await asyncio.to_thread(jobs.shutdown)
            workspace_lock.release()

    app = FastAPI(title="vjms", description="Local tiny-language-model studio", version=__version__, lifespan=lifespan)
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"detail": "仅允许工作台同源请求。"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path == "/" or request.url.path.startswith("/assets/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            )
        return response

    @app.exception_handler(ValueError)
    async def invalid_request(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing_resource(request, exc):
        return JSONResponse({"detail": "资源不存在。"}, status_code=404)

    @app.exception_handler(RuntimeError)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/api/system")
    def system():
        return {**hardware_info(), "version": __version__, "data_dir": str(store.root), "presets": PRESETS}

    @app.get("/api/datasets")
    def datasets():
        return store.list("datasets")

    @app.post("/api/datasets", status_code=201)
    def add_dataset(body: DatasetRequest):
        return import_text(store, **body.model_dump())

    @app.post("/api/datasets/upload", status_code=201)
    async def upload_dataset(file: UploadFile = File(...), validation_fraction: float = Form(0.1)):
        raw = await file.read(MAX_BYTES + 1)
        await file.close()
        if len(raw) > MAX_BYTES:
            raise ValueError("文件超过 32MB 上限。")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("请上传 UTF-8 编码的 TXT 文件。") from exc
        return await asyncio.to_thread(import_text, store, file.filename or "导入文本", text, validation_fraction)

    @app.post("/api/datasets/sample", status_code=201)
    def add_sample():
        return import_text(store, "中英文流程示例 · 仅验证功能", sample_text())

    @app.get("/api/datasets/{item_id}")
    def dataset(item_id: str):
        return store.get("datasets", item_id)

    @app.get("/api/tokenizers")
    def tokenizers():
        return store.list("tokenizers")

    @app.post("/api/tokenizers", status_code=201)
    def add_tokenizer(body: TokenizerRequest):
        # Serialize heavy data operations with training and inference on small machines.
        jobs = app.state.jobs
        with jobs.lock:
            jobs.ensure_idle()
            return train_tokenizer(store, body.dataset_id, body.vocab_size)

    @app.post("/api/tokenizers/{item_id}/encode")
    def encode(item_id: str, body: DatasetRequest):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(store.path("tokenizers", item_id) / "tokenizer.json"))
        encoded = tok.encode(body.text[:10000])
        return {"ids": encoded.ids, "tokens": encoded.tokens, "decoded": tok.decode(encoded.ids)}

    @app.post("/api/models/estimate")
    def estimate(body: RunRequest):
        tok = store.get("tokenizers", body.tokenizer_id)
        cfg = ModelConfig(vocab_size=tok["vocab_size"], **body.architecture.model_dump())
        return {"parameters": cfg.parameters, "weight_mb_fp32": cfg.parameters * 4 / 1e6,
                "training_state_mb_fp32": cfg.parameters * 16 / 1e6, "config": cfg.to_dict(),
                "note": "仅含权重、梯度和 Adam 状态；不含激活、临时张量及框架开销。"}

    @app.get("/api/runs")
    def runs():
        with app.state.jobs.lock:
            app.state.jobs.reap()
            return store.list("runs")

    @app.post("/api/runs", status_code=201)
    def add_run(body: RunRequest):
        return app.state.jobs.create(body)

    @app.get("/api/runs/{item_id}")
    def run(item_id: str):
        with app.state.jobs.lock:
            app.state.jobs.reap()
            path = store.path("runs", item_id)
            return {**store.get("runs", item_id), "config": read_json(path / "config.json"),
                    "metrics": read_metrics(path),
                    "checkpoints": [name for name in ("latest", "best") if (path / f"{name}.pt").exists()]}

    @app.post("/api/runs/{item_id}/pause")
    def pause_run(item_id: str):
        return app.state.jobs.pause(item_id)

    @app.post("/api/runs/{item_id}/resume")
    def resume_run(item_id: str, body: ResumeRequest):
        return app.state.jobs.resume(item_id, body)

    @app.get("/api/runs/{item_id}/checkpoints/{checkpoint}")
    def download_checkpoint(item_id: str, checkpoint: str):
        if checkpoint not in {"latest", "best"}:
            raise ValueError("检查点只能是 latest 或 best。")
        path = store.path("runs", item_id) / f"{checkpoint}.pt"
        if not path.exists():
            raise FileNotFoundError()
        return FileResponse(path, media_type="application/octet-stream", filename=f"vjms-{item_id}-{checkpoint}.pt")

    def predict(body: GenerateRequest):
        jobs = app.state.jobs
        if not jobs.lock.acquire(blocking=False):
            raise RuntimeError("设备正在处理其他操作，请稍后重试。")
        try:
            jobs.ensure_idle()
            return generate(store, body)
        finally:
            jobs.lock.release()

    @app.post("/api/generate")
    def generate_text(body: GenerateRequest):
        return predict(body)

    @app.get("/v1/models")
    def deployed_models():
        return {"object": "list", "data": [
            {"id": meta["id"], "object": "model", "created": int(time.time()), "owned_by": "local"}
            for meta in store.list("runs") if (store.path("runs", meta["id"]) / "best.pt").exists()
        ]}

    @app.post("/v1/completions")
    def completions(body: CompletionRequest):
        result = predict(GenerateRequest(run_id=body.model, prompt=body.prompt, max_new_tokens=body.max_tokens,
                                        temperature=body.temperature, top_p=body.top_p))
        return {"id": "cmpl-" + uuid.uuid4().hex, "object": "text_completion", "created": int(time.time()),
                "model": body.model, "choices": [{"text": result["completion"], "index": 0,
                "finish_reason": result["finish_reason"], "logprobs": None}],
                "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["generated_tokens"],
                          "total_tokens": result["prompt_tokens"] + result["generated_tokens"]}}

    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return app
