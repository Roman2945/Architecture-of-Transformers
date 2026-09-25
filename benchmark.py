"""Reproducible CPU/CUDA matrix multiplication timing for lab 1."""

import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import torch


def measure(a, b, repeats):
    """Median time per multiplication, excluding device transfer."""
    for _ in range(10):
        torch.matmul(a, b)
    if a.is_cuda:
        torch.cuda.synchronize()

    samples = []
    for _ in range(5):
        start = perf_counter()
        for _ in range(repeats):
            torch.matmul(a, b)
        if a.is_cuda:
            torch.cuda.synchronize()
        samples.append((perf_counter() - start) / repeats * 1000)
    return statistics.median(samples)


def main():
    torch.manual_seed(42)
    result = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cpu": platform.processor(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "measurements": [],
    }
    for size, repeats in ((16, 1000), (128, 200), (1024, 20), (2048, 10)):
        a = torch.randn(size, size, dtype=torch.float32)
        b = torch.randn(size, size, dtype=torch.float32)
        cpu_ms = measure(a, b, repeats)
        record = {"size": size, "cpu_ms": cpu_ms}
        if torch.cuda.is_available():
            a_gpu, b_gpu = a.cuda(), b.cuda()
            gpu_ms = measure(a_gpu, b_gpu, repeats)
            torch.testing.assert_close(a @ b, (a_gpu @ b_gpu).cpu(), rtol=1e-3, atol=1e-3)
            record.update(gpu_ms=gpu_ms, speedup=cpu_ms / gpu_ms)
        result["measurements"].append(record)

    Path("benchmark_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
