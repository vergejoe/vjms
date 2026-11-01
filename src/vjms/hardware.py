import platform

import psutil
import torch


def resolve_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("设备只能为 auto、cpu、cuda 或 mps。")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请检查 NVIDIA 驱动和 PyTorch 安装。")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS 不可用，请检查 Mac 芯片、系统和 PyTorch 版本。")
    return requested


def hardware_info():
    selected = resolve_device()
    memory = psutil.virtual_memory()
    return {
        "platform": platform.system(), "architecture": platform.machine(),
        "device": selected, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if selected == "cuda" else "Apple Metal GPU" if selected == "mps" else "CPU",
        "ram_gb": round(memory.total / 1024**3, 1), "available_ram_gb": round(memory.available / 1024**3, 1),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if selected == "cuda" else None,
        "torch_version": torch.__version__,
    }
