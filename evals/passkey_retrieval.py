from __future__ import annotations

import argparse
import os
import random

import torch

from inference import GenerationConfig, SamplingConfig, TokenGenerator, run_generation
from prompting import HarmonyMessage, default_system_message, render_harmony


def make_prompt(passkey: str, filler_tokens: int, tokenizer) -> str:
    filler = "This is filler text to build long context. "
    chunks = []
    tokens = []
    while len(tokens) < filler_tokens:
        chunks.append(filler)
        tokens = tokenizer.encode("".join(chunks), allowed_special="all")

    context = "".join(chunks)
    system = default_system_message()
    user = (
        f"Remember this passkey for later: {passkey}\n"
        f"{context}\n\n"
        "What is the passkey?"
    )
    messages = [
        HarmonyMessage(role="system", content=system),
        HarmonyMessage(role="user", content=user),
    ]
    return render_harmony(messages, add_assistant_start=True, assistant_channel="final")


def main() -> None:
    parser = argparse.ArgumentParser(description="Passkey retrieval eval")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--passkey", default="ultra-omega-31415")
    parser.add_argument("--filler_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    checkpoint = args.checkpoint or os.environ.get("MOE_XTEND_CHECKPOINT")
    if not checkpoint:
        raise SystemExit("--checkpoint is required (or set MOE_XTEND_CHECKPOINT)")

    random.seed(args.seed)

    device = torch.device(args.device)
    generator = TokenGenerator(
        checkpoint=checkpoint,
        device=device,
        dtype=args.dtype,
        compile_model=False,
        debug=False,
    )

    prompt = make_prompt(args.passkey, args.filler_tokens, generator.tokenizer)

    gen_cfg = GenerationConfig(
        checkpoint_path=checkpoint,
        device=device,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        prefill_chunk=0,
        truncate_prompt=True,
        max_prompt_tokens=None,
        stream=False,
        metrics_json=None,
        compile_model=False,
    )

    sampling_cfg = SamplingConfig(
        temperature=args.temperature,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        penalize_prompt=True,
        logit_bias={},
    )

    stop_tokens = [generator.eot_token, generator.call_token]

    result = run_generation(
        generator,
        prompt,
        gen_cfg,
        sampling_cfg,
        stop_tokens=stop_tokens,
        stop_sequences=[],
        debug=False,
    )

    print("\n=== MODEL OUTPUT ===\n")
    print(result["text"])

    hit = args.passkey in result["text"]
    print("\n=== RESULT ===")
    print("PASS" if hit else "FAIL")


if __name__ == "__main__":
    main()
