"""
Benchmarking utilities:
- Parameter count and memory footprint analysis.
- GPU / CPU inference latency and throughput (FPS) measurement.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """
    Computes total and trainable parameter counts and approximate memory size.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    buffer_size_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 ** 2)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": param_size_mb,
        "buffer_size_mb": buffer_size_mb,
        "total_size_mb": param_size_mb + buffer_size_mb,
    }


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 256),
    device: Optional[torch.device] = None,
    warmup_runs: int = 15,
    test_runs: int = 50,
) -> Dict[str, float]:
    """
    Measures inference latency and throughput.
    Uses CUDA events for exact GPU timing if available, otherwise high-precision perf_counter.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dummy_input = torch.randn(input_size, device=device)
    is_cuda = device.type == "cuda"

    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input)
    if is_cuda:
        torch.cuda.synchronize()

    times_ms = []

    if is_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(test_runs):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    else:
        for _ in range(test_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    fps = 1000.0 / mean_ms * input_size[0]

    return {
        "mean_latency_ms": round(mean_ms, 2),
        "min_latency_ms": round(min_ms, 2),
        "max_latency_ms": round(max_ms, 2),
        "fps": round(fps, 1),
    }
