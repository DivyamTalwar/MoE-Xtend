"""Profile MoE-Xtend's portable gather and grouped dispatch paths."""

import argparse
import time

import torch

from model import MLPBlock, ModelConfigs


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = ModelConfigs(
        num_experts=args.experts,
        experts_per_token=args.top_k,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
    )
    block = MLPBlock(config, device=device).eval()
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    x = torch.randn((1, args.tokens, args.hidden), device=device, dtype=torch.bfloat16)

    for mode in ("gather", "grouped"):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                block(x, dispatch=mode)
            synchronize(device)
            start = time.perf_counter()
            for _ in range(args.steps):
                block(x, dispatch=mode)
            synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000 / args.steps
        peak_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
        print(f"{mode:8s} {elapsed_ms:9.3f} ms/step peak={peak_mib:9.1f} MiB")


if __name__ == "__main__":
    main()
