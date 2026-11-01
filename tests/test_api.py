import time

import pytest
from fastapi.testclient import TestClient

from vjms.app import create_app


def wait_for(client, run_id, statuses, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        run = response.json()
        if run["status"] in statuses:
            return run
        if run["status"] == "failed":
            pytest.fail(run.get("error"))
        time.sleep(.1)
    pytest.fail(f"training worker timed out: {run}")


def test_real_subprocess_train_pause_resume_generate_and_completion(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/system").status_code == 200
        data = client.post("/api/datasets/sample").json()
        response = client.post("/api/tokenizers", json={"dataset_id": data["id"], "vocab_size": 512})
        assert response.status_code == 201, response.text
        tokenizer = response.json()
        config = {"tokenizer_id": tokenizer["id"], "device": "cpu", "max_steps": 1000, "batch_size": 2,
                  "gradient_accumulation": 1, "eval_interval": 2, "checkpoint_interval": 2, "eval_batches": 1,
                  "architecture": {"dim": 32, "layers": 1, "heads": 2, "hidden_dim": 64, "context_length": 16}}
        response = client.post("/api/runs", json=config)
        assert response.status_code == 201, response.text
        run_id = response.json()["id"]
        assert client.post("/api/runs", json=config).status_code == 409
        assert client.post("/api/generate", json={"run_id": run_id}).status_code == 409
        wait_for(client, run_id, {"running"})
        assert client.post(f"/api/runs/{run_id}/pause").status_code == 200
        paused = wait_for(client, run_id, {"paused"})
        assert "latest" in paused["checkpoints"]
        response = client.post(f"/api/runs/{run_id}/resume", json={"additional_steps": 2, "device": "cpu"})
        assert response.status_code == 200, response.text
        completed = wait_for(client, run_id, {"completed"})
        assert completed["step"] == paused["step"] + 2
        assert "best" in completed["checkpoints"]
        response = client.post("/api/generate", json={"run_id": run_id, "prompt": "你好 hello", "max_new_tokens": 4})
        assert response.status_code == 200, response.text
        assert response.json()["generated_tokens"] <= 4
        assert client.get(f"/api/runs/{run_id}/checkpoints/best").status_code == 200
        assert client.get("/v1/models").json()["data"][0]["id"] == run_id
        response = client.post("/v1/completions", json={"model": run_id, "prompt": "Hello", "max_tokens": 3})
        assert response.status_code == 200, response.text
        assert response.json()["object"] == "text_completion"
        assert client.post("/v1/completions", json={"model": run_id, "prompt": "hello", "stream": True}).status_code == 422


def test_input_validation_and_local_access_boundary(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/datasets", json={"text": "same\n\nsame"}).status_code == 400
        assert client.post("/api/datasets/upload", files={"file": ("bad.txt", b"\xff\xfe")}).status_code == 400
        assert client.post("/api/datasets/sample", headers={"Origin": "https://untrusted.example"}).status_code == 403
        assert client.get("/api/system", headers={"Host": "untrusted.example"}).status_code == 400
        assert client.get("/api/runs/not-an-id").status_code == 400
        assert client.get("/api/runs/1234567890abcdef").status_code == 404
        assert "frame-ancestors 'none'" in client.get("/").headers["Content-Security-Policy"]


def test_workspace_only_allows_one_server(tmp_path):
    with TestClient(create_app(tmp_path)):
        with pytest.raises(RuntimeError, match="已有"):
            with TestClient(create_app(tmp_path)):
                pass
