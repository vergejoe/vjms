# v0.1.0 verification

Verified locally on 2026-11-01 on macOS arm64, 16GB unified memory, Python 3.12.14, PyTorch 2.15.0.

## Automated checks

- `python -m pytest -q`: **9 passed**.
- `node --check src/vjms/static/app.js`: passed.
- `python -m pip check`: passed.
- Checkpoint resume produces exactly the same model tensors, optimizer state and sampling RNG state as uninterrupted CPU training in the regression test.
- API integration test starts a real CPU worker, pauses, resumes, generates text, downloads a checkpoint, and calls the completions endpoint.

## Actual Mac GPU checks

- Browser workflow: import the included bilingual sample → train BPE → train a 1,608,320-parameter model for 40 steps on MPS → save latest / best → generate text on MPS.
- The 40-step trial used context 64, microbatch 2 and accumulation 1, processing 5,120 training tokens. Both training and held-out losses were recorded. This validates the pipeline, not language quality.
- `python scripts/check_device.py --device mps --preset base`: passed a two-step FP32 forward / backward / AdamW smoke test with **19,275,648 parameters**, context 64 and batch 1. Random inputs; no model-quality claim.
- Six UI pages checked at desktop and 390px viewport widths; no horizontal document overflow at 390px. Browser console had no warnings or errors during the tested workflow.

## Limits of this evidence

- Windows CUDA has not been tested on the target machine. The repository includes a cross-platform CPU CI example in `docs/ci-workflow.example.yml`; it is not active until copied into `.github/workflows/` using credentials with workflow permission.
- The near-20M check used a short context and small batch. It does not establish support for every context / batch setting, nor for every 8GB NVIDIA GPU.
- The sample corpus and 40-step trial are intentionally too small to establish useful bilingual language quality.
