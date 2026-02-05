from __future__ import annotations

import argparse
import os
import random
from typing import List

import torch

from inference import GenerationConfig, SamplingConfig, TokenGenerator, run_generation
from prompting import HarmonyMessage, default_system_message, render_harmony


def build_haystack(tokenizer, target_tokens: int, needle: str, needle_position: str) -> str:
    filler = "This is filler text meant to simulate long context. "
    tokens: List[int] = []
    chunks: List[str] = []
    while len(tokens) < target_tokens:
        chunks.append(filler)
        tokens = tokenizer.encode("".join(chunks), allowed_special="all")

    haystack = "".join(chunks)
    if needle_position == "start":
        return needle + "\n" + haystack
    if needle_position == "end":
        return haystack + "\n" + needle

    # middle
    half = len(haystack) // 2
    return haystack[:half] + "\n" + needle + "\n" + haystack[half:]


def build_prompt(needle_text: str, haystack_text: str, question: str) -> str:
    system = default_system_message()
    user_content = f"{haystack_text}\n\n{question}"
    messages = [
        HarmonyMessage(role="system", content=system),
        HarmonyMessage(role="user", content=user_content),
    ]
    return render_harmony(messages, add_assistant_start=True, assistant_channel="final")


def main() -> None:
    parser = argparse.ArgumentParser(description="Needle-in-haystack eval")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--needle", default="The secret passphrase is: blue-otter-9.")
    parser.add_argument("--question", default="What is the secret passphrase?")
    parser.add_argument("--haystack_tokens", type=int, default=8192)
    parser.add_argument("--needle_position", choices=["start", "middle", "end"], default="middle")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
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

    haystack = build_haystack(generator.tokenizer, args.haystack_tokens, args.needle, args.needle_position)
    prompt = build_prompt(args.needle, haystack, args.question)

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

    hit = args.needle.split(":", 1)[-1].strip() in result["text"]
    print("\n=== RESULT ===")
    print("PASS" if hit else "FAIL")


if __name__ == "__main__":
    main()
