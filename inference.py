from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from model import Cache, Transformer
from openai_harmony import HarmonyEncodingName, load_harmony_encoding
from prompting import HarmonyMessage, default_system_message, load_conversation_json, render_harmony
from sampling import SamplingConfig, apply_logit_bias, apply_penalties, sample_token


DEBUG = False


@dataclass
class GenerationConfig:
    checkpoint_path: str
    device: torch.device
    dtype: str
    max_tokens: int
    prefill_chunk: int
    truncate_prompt: bool
    max_prompt_tokens: Optional[int]
    stream: bool
    metrics_json: Optional[str]
    compile_model: bool


@dataclass
class PrefillState:
    prompt_tokens: List[int]
    caches: List[Cache]
    cache_snapshot: List[tuple]
    logits: torch.Tensor
    prefill_time: float
    max_gen_tokens: int

def debug_print(enabled: bool, *args, **kwargs) -> None:
    if enabled:
        print("[DEBUG]", *args, **kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_logit_bias(value: Optional[str]) -> Dict[int, float]:
    if not value:
        return {}
    if os.path.isfile(value):
        data = load_json(value)
        if not isinstance(data, dict):
            raise ValueError("logit_bias JSON must be an object of token_id -> bias")
        out: Dict[int, float] = {}
        for k, v in data.items():
            out[int(k)] = float(v)
        return out
    out: Dict[int, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        token_str, bias_str = item.split(":", 1)
        out[int(token_str)] = float(bias_str)
    return out


class TokenGenerator:
    _model: Optional[Transformer] = None
    _model_meta: Optional[tuple[str, str, str]] = None
    _tokenizer = None
    _eot_token: Optional[int] = None
    _call_token: Optional[int] = None

    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        dtype: str,
        *,
        compile_model: bool = False,
        debug: bool = False,
    ) -> None:
        self.device = device
        self.dtype = dtype
        meta = (checkpoint, str(device), dtype)

        if TokenGenerator._model is None or TokenGenerator._model_meta != meta:
            debug_print(debug, f"Loading model weights from {checkpoint}...")
            start = time.time()
            TokenGenerator._model = Transformer.from_checkpoint(checkpoint, device=self.device)
            TokenGenerator._model_meta = meta
            if dtype != "bf16":
                torch_dtype = {
                    "fp16": torch.float16,
                    "fp32": torch.float32,
                    "bf16": torch.bfloat16,
                }[dtype]
                TokenGenerator._model = TokenGenerator._model.to(torch_dtype)
            if compile_model and hasattr(torch, "compile"):
                try:
                    TokenGenerator._model = torch.compile(TokenGenerator._model, mode="reduce-overhead")
                except Exception as exc:  # pragma: no cover - best effort
                    print(f"[WARN] torch.compile failed, continuing uncompiled: {exc}")
            TokenGenerator._model.eval()
            print(f"✓ Model weights loaded in {time.time()-start:.2f}s")
        else:
            print("Model weights already loaded. Reusing existing instance.")

        self.model = TokenGenerator._model

        if TokenGenerator._tokenizer is None:
            print("Loading tokenizer...")
            TokenGenerator._tokenizer = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            TokenGenerator._eot_token = TokenGenerator._tokenizer.encode("<|return|>", allowed_special="all")[0]
            TokenGenerator._call_token = TokenGenerator._tokenizer.encode("<|call|>", allowed_special="all")[0]
            debug_print(debug, f"✓ Tokenizer loaded, EOT token: {TokenGenerator._eot_token}")

        self.tokenizer = TokenGenerator._tokenizer
        self.eot_token = TokenGenerator._eot_token
        self.call_token = TokenGenerator._call_token


@torch.inference_mode()
def prefill(
    model: Transformer,
    prompt_tokens: List[int],
    caches: List[Cache],
    device: torch.device,
    *,
    chunk_size: int = 0,
    debug: bool = False,
) -> torch.Tensor:
    if not prompt_tokens:
        raise ValueError("Prompt must contain at least 1 token.")
    logits = None
    if chunk_size <= 0:
        input_tensor = torch.as_tensor([prompt_tokens], dtype=torch.long, device=device)
        debug_print(debug, f"  - Prefill input shape: {input_tensor.shape}")
        logits = model(input_tensor, caches=caches)[:, -1, :].squeeze(0)
        return logits

    for start in range(0, len(prompt_tokens), chunk_size):
        chunk = prompt_tokens[start : start + chunk_size]
        input_tensor = torch.as_tensor([chunk], dtype=torch.long, device=device)
        debug_print(debug, f"  - Prefill chunk shape: {input_tensor.shape}")
        logits = model(input_tensor, caches=caches)[:, -1, :].squeeze(0)
    return logits


def maybe_truncate_prompt(
    prompt_tokens: List[int],
    *,
    max_prompt_tokens: int,
    truncate: bool,
) -> List[int]:
    if len(prompt_tokens) <= max_prompt_tokens:
        return prompt_tokens
    if not truncate:
        raise ValueError(
            f"Prompt too long ({len(prompt_tokens)} tokens), max allowed is {max_prompt_tokens}. "
            "Enable --truncate_prompt to keep the most recent tokens."
        )
    return prompt_tokens[-max_prompt_tokens:]


def check_stop_sequences(tokens: List[int], stop_sequences: List[List[int]]) -> int:
    for seq in stop_sequences:
        if seq and len(tokens) >= len(seq) and tokens[-len(seq) :] == seq:
            return len(seq)
    return 0


def extract_json(text: str) -> Optional[object]:
    # Simple brace-based extractor for first valid JSON object/array.
    stack = []
    start_idx = None
    for i, ch in enumerate(text):
        if ch in "{[":
            if start_idx is None:
                start_idx = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            opening = stack.pop()
            if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                stack.clear()
                start_idx = None
                continue
            if not stack and start_idx is not None:
                candidate = text[start_idx : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    start_idx = None
                    stack.clear()
    return None


def validate_schema(payload: object, schema_path: str) -> List[str]:
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    try:
        import jsonschema
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("jsonschema is required for --json_schema") from exc

    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for err in validator.iter_errors(payload):
        errors.append(err.message)
    return errors


def serialize_harmony_messages(messages: Iterable[object]) -> List[dict]:
    out: List[dict] = []
    for msg in messages:
        if isinstance(msg, dict):
            out.append(msg)
            continue
        if hasattr(msg, "to_dict") and callable(getattr(msg, "to_dict")):
            out.append(msg.to_dict())
            continue
        if hasattr(msg, "__dict__"):
            data = {k: v for k, v in msg.__dict__.items() if not k.startswith("_")}
            out.append(data)
            continue
        out.append({"text": str(msg)})
    return out


def parse_channels(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return ("analysis", "commentary", "final")
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return tuple(parts) if parts else ("analysis", "commentary", "final")


def build_messages_from_args(args) -> List[HarmonyMessage]:
    messages: List[HarmonyMessage] = []

    system_text = None
    if args.system_file:
        system_text = read_text(args.system_file)
    elif args.system:
        system_text = args.system

    if args.add_default_system or system_text:
        if system_text is None:
            system_text = default_system_message(
                knowledge_cutoff=args.knowledge_cutoff,
                current_date=args.current_date,
                reasoning=args.reasoning,
                channels=parse_channels(args.channels),
                tools_required=bool(args.tools_required),
            )
        messages.append(HarmonyMessage(role="system", content=system_text))

    developer_text = None
    if args.developer_file:
        developer_text = read_text(args.developer_file)
    elif args.developer:
        developer_text = args.developer

    if args.tools_file:
        tools_text = read_text(args.tools_file)
        if developer_text:
            developer_text = developer_text.rstrip() + "\n\n" + tools_text
        else:
            developer_text = tools_text

    if developer_text:
        messages.append(HarmonyMessage(role="developer", content=developer_text))

    if args.conversation:
        messages.extend(load_conversation_json(args.conversation))
        return messages

    if args.prompt:
        messages.append(HarmonyMessage(role="user", content=args.prompt))
        return messages

    raise ValueError("No prompt or conversation provided.")


def build_prompt_text(args) -> str:
    if args.format == "plain":
        if not args.prompt:
            raise ValueError("--prompt is required for plain format.")
        return args.prompt

    messages = build_messages_from_args(args)
    assistant_channel = args.assistant_channel
    if assistant_channel == "none":
        assistant_channel = None
    return render_harmony(messages, add_assistant_start=True, assistant_channel=assistant_channel)


def load_prompts(args) -> List[str]:
    prompts: List[str] = []
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    prompts.append(line)

    if args.prompt:
        prompts.append(args.prompt)

    if args.conversation:
        if prompts:
            print("[WARN] --conversation provided; ignoring --prompt/--prompt_file")
        return [build_prompt_text(args)]

    if not prompts:
        prompts = ["Write a python function that prints hello world"]

    if args.format == "plain":
        return prompts

    rendered: List[str] = []
    for prompt in prompts:
        args_copy = argparse.Namespace(**vars(args))
        args_copy.prompt = prompt
        rendered.append(build_prompt_text(args_copy))
    return rendered


def build_prefill_state(
    generator: TokenGenerator,
    prompt_text: str,
    gen_cfg: GenerationConfig,
    sampling_cfg: SamplingConfig,
    *,
    debug: bool = False,
) -> PrefillState:
    tokenizer = generator.tokenizer
    model = generator.model
    model_configs = model.configs

    max_context = model_configs.initial_context_length * int(model_configs.rope_scaling_factor)

    prompt_tokens = tokenizer.encode(prompt_text, allowed_special="all")
    debug_print(debug, f"Encoded prompt length: {len(prompt_tokens)} tokens")

    if gen_cfg.max_prompt_tokens is None:
        if gen_cfg.max_tokens > 0:
            max_prompt_tokens = max_context - gen_cfg.max_tokens
        else:
            max_prompt_tokens = max_context
    else:
        max_prompt_tokens = gen_cfg.max_prompt_tokens

    if max_prompt_tokens < 1:
        raise ValueError(
            "max_prompt_tokens would be < 1; reduce max_tokens or increase model context."
        )

    prompt_tokens = maybe_truncate_prompt(
        prompt_tokens, max_prompt_tokens=max_prompt_tokens, truncate=gen_cfg.truncate_prompt
    )

    if gen_cfg.max_tokens == 0:
        max_gen_tokens = max(0, max_context - len(prompt_tokens))
    else:
        max_gen_tokens = gen_cfg.max_tokens

    total_tokens = len(prompt_tokens) + max_gen_tokens
    cache_size = min(total_tokens, max_context)

    print("Initialising KV caches...")
    print(f"  - Cache size: {cache_size} tokens")
    print(f"  - Number of layers: {model_configs.num_hidden_layers}")
    print(f"  - KV heads: {model_configs.num_key_value_heads}")
    print(f"  - Head dim: {model_configs.head_dim}")

    cache_start = time.time()
    caches = [
        Cache(
            batch_size=1,
            n_ctx=cache_size,
            n_kv_heads=model_configs.num_key_value_heads,
            d_head=model_configs.head_dim,
            device=gen_cfg.device,
        )
        for _ in range(model_configs.num_hidden_layers)
    ]
    print(f"✓ Caches Initialised in {time.time()-cache_start:.2f}s")

    print(f"Prompt length: {len(prompt_tokens)} tokens")

    prefill_start = time.time()
    logits = prefill(
        model,
        prompt_tokens,
        caches,
        gen_cfg.device,
        chunk_size=gen_cfg.prefill_chunk,
        debug=debug,
    )
    prefill_time = time.time() - prefill_start
    print(f"✓ Prefill complete in {prefill_time:.2f}s")

    cache_snapshot = [cache.snapshot() for cache in caches]

    return PrefillState(
        prompt_tokens=prompt_tokens,
        caches=caches,
        cache_snapshot=cache_snapshot,
        logits=logits,
        prefill_time=prefill_time,
        max_gen_tokens=max_gen_tokens,
    )


def decode_from_state(
    generator: TokenGenerator,
    state: PrefillState,
    gen_cfg: GenerationConfig,
    sampling_cfg: SamplingConfig,
    *,
    stop_tokens: List[int],
    stop_sequences: List[List[int]],
    debug: bool = False,
    return_logprobs: bool = False,
) -> dict:
    tokenizer = generator.tokenizer

    for cache, snapshot in zip(state.caches, state.cache_snapshot):
        cache.restore(snapshot)

    token_counts: Dict[int, int] = {}
    if sampling_cfg.penalize_prompt:
        for token in state.prompt_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

    generated_tokens: List[int] = []
    logprobs: List[float] = []
    pending_stream: List[int] = []
    max_stop_len = max((len(seq) for seq in stop_sequences), default=0)

    logits = state.logits.clone()
    decode_start = time.time()
    ttft = None

    num_generated = 0
    while num_generated < state.max_gen_tokens:
        apply_logit_bias(logits, sampling_cfg.logit_bias)
        apply_penalties(
            logits,
            token_counts,
            repetition_penalty=sampling_cfg.repetition_penalty,
            presence_penalty=sampling_cfg.presence_penalty,
            frequency_penalty=sampling_cfg.frequency_penalty,
        )

        predicted_token = sample_token(logits, sampling_cfg)
        generated_tokens.append(predicted_token)
        token_counts[predicted_token] = token_counts.get(predicted_token, 0) + 1

        if return_logprobs:
            token_logprobs = torch.log_softmax(logits, dim=-1)
            logprobs.append(token_logprobs[predicted_token].item())

        if ttft is None:
            ttft = time.time() - decode_start

        if gen_cfg.stream:
            pending_stream.append(predicted_token)
            if max_stop_len == 0 and pending_stream:
                print(tokenizer.decode([pending_stream.pop(0)]), end="", flush=True)
            elif len(pending_stream) > max_stop_len:
                print(tokenizer.decode([pending_stream.pop(0)]), end="", flush=True)

        num_generated += 1

        stop_len = 0
        if predicted_token in stop_tokens:
            stop_len = 1
        else:
            stop_len = check_stop_sequences(generated_tokens, stop_sequences)

        if stop_len > 0:
            if gen_cfg.stream and pending_stream:
                if stop_len <= len(pending_stream):
                    pending_stream = pending_stream[:-stop_len]
                for tok in pending_stream:
                    print(tokenizer.decode([tok]), end="", flush=True)
                pending_stream = []
            if stop_len > 0:
                generated_tokens = generated_tokens[:-stop_len]
                if return_logprobs and logprobs:
                    logprobs = logprobs[:-stop_len]
            break

        input_tensor = torch.as_tensor([[predicted_token]], dtype=torch.long, device=gen_cfg.device)
        logits = generator.model(input_tensor, caches=state.caches)[:, -1, :].squeeze(0)

        if num_generated >= state.max_gen_tokens:
            break

    if gen_cfg.stream:
        if pending_stream:
            for tok in pending_stream:
                print(tokenizer.decode([tok]), end="", flush=True)
        print()

    decode_time = time.time() - decode_start
    if ttft is None:
        ttft = 0.0

    metrics = {
        "prompt_tokens": len(state.prompt_tokens),
        "generated_tokens": len(generated_tokens),
        "ttft_sec": ttft,
        "prefill_time_sec": state.prefill_time,
        "decode_time_sec": decode_time,
        "tokens_per_sec": (len(generated_tokens) / decode_time) if decode_time > 0 else 0.0,
    }

    text = tokenizer.decode(generated_tokens)
    result = {"text": text, "tokens": generated_tokens, "metrics": metrics}
    if return_logprobs:
        result["logprobs"] = logprobs
    return result


def run_generation(
    generator: TokenGenerator,
    prompt_text: str,
    gen_cfg: GenerationConfig,
    sampling_cfg: SamplingConfig,
    *,
    stop_tokens: List[int],
    stop_sequences: List[List[int]],
    debug: bool = False,
    return_logprobs: bool = False,
) -> dict:
    state = build_prefill_state(
        generator, prompt_text, gen_cfg, sampling_cfg, debug=debug
    )
    return decode_from_state(
        generator,
        state,
        gen_cfg,
        sampling_cfg,
        stop_tokens=stop_tokens,
        stop_sequences=stop_sequences,
        debug=debug,
        return_logprobs=return_logprobs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoE-Xtend inference (Harmony-native)")

    parser.add_argument("--config", help="Path to JSON config file", default=None)

    parser.add_argument("--checkpoint", help="Checkpoint path", default=None)
    parser.add_argument("--device", help="Device (cuda, cpu, cuda:0)", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--format", choices=["harmony", "plain"], default=None)
    parser.add_argument("--prompt", help="Prompt text", default=None)
    parser.add_argument("--prompt_file", help="Newline-delimited prompts file", default=None)
    parser.add_argument("--conversation", help="Harmony conversation JSON file", default=None)
    parser.add_argument("--system", help="Harmony system message", default=None)
    parser.add_argument("--system_file", help="Harmony system message file", default=None)
    parser.add_argument("--developer", help="Harmony developer message", default=None)
    parser.add_argument("--developer_file", help="Harmony developer message file", default=None)
    parser.add_argument("--tools_file", help="Raw tools text to append to developer message", default=None)
    parser.add_argument("--assistant_channel", choices=["analysis", "commentary", "final", "none"], default=None)
    parser.add_argument("--add_default_system", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tools_required", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--knowledge_cutoff", default=None)
    parser.add_argument("--current_date", default=None)
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--channels", help="Comma-separated Harmony channels list", default=None)

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--presence_penalty", type=float, default=None)
    parser.add_argument("--frequency_penalty", type=float, default=None)
    parser.add_argument("--penalize_prompt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--logit_bias", help="JSON path or token_id:bias,...", default=None)

    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--logprobs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--logprobs_json", default=None)
    parser.add_argument("--output_json", default=None)

    parser.add_argument("--prefill_chunk", type=int, default=None)
    parser.add_argument("--truncate_prompt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)

    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--metrics_json", default=None)
    parser.add_argument("--parse_harmony", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--parse_harmony_json", default=None)
    parser.add_argument("--extract_json", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--json_schema", default=None)
    parser.add_argument("--json_output", default=None)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)

    return parser


def merge_settings(defaults: dict, config_file: Optional[str], args: argparse.Namespace) -> dict:
    settings = dict(defaults)
    if config_file:
        settings.update(load_json(config_file))
    for key, value in vars(args).items():
        if value is not None and key != "config":
            settings[key] = value
    return settings


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    defaults = {
        "checkpoint": os.environ.get("MOE_XTEND_CHECKPOINT", "./checkpoints/gpt-oss-20b/original"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "bf16",
        "compile": False,
        "format": "harmony",
        "assistant_channel": "final",
        "add_default_system": True,
        "tools_required": False,
        "knowledge_cutoff": "2024-06",
        "current_date": None,
        "reasoning": "medium",
        "channels": "analysis,commentary,final",
        "temperature": 0.1,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "penalize_prompt": True,
        "max_tokens": 100,
        "num_samples": 1,
        "logprobs": False,
        "logprobs_json": None,
        "output_json": None,
        "prefill_chunk": 0,
        "truncate_prompt": False,
        "stream": False,
        "parse_harmony": False,
        "parse_harmony_json": None,
        "extract_json": False,
        "json_schema": None,
        "json_output": None,
        "debug": False,
    }

    settings = merge_settings(defaults, args.config, args)

    global DEBUG
    DEBUG = bool(settings.get("debug", False))

    if settings.get("seed") is not None:
        set_seed(int(settings["seed"]))

    device = torch.device(settings["device"])

    gen_cfg = GenerationConfig(
        checkpoint_path=settings["checkpoint"],
        device=device,
        dtype=settings["dtype"],
        max_tokens=int(settings["max_tokens"]),
        prefill_chunk=int(settings["prefill_chunk"]),
        truncate_prompt=bool(settings["truncate_prompt"]),
        max_prompt_tokens=settings.get("max_prompt_tokens"),
        stream=bool(settings["stream"]),
        metrics_json=settings.get("metrics_json"),
        compile_model=bool(settings["compile"]),
    )

    sampling_cfg = SamplingConfig(
        temperature=float(settings["temperature"]),
        top_k=int(settings["top_k"]),
        top_p=float(settings["top_p"]),
        min_p=float(settings["min_p"]),
        repetition_penalty=float(settings["repetition_penalty"]),
        presence_penalty=float(settings["presence_penalty"]),
        frequency_penalty=float(settings["frequency_penalty"]),
        penalize_prompt=bool(settings["penalize_prompt"]),
        logit_bias=parse_logit_bias(settings.get("logit_bias")),
    )

    generator = TokenGenerator(
        checkpoint=gen_cfg.checkpoint_path,
        device=gen_cfg.device,
        dtype=gen_cfg.dtype,
        compile_model=gen_cfg.compile_model,
        debug=DEBUG,
    )

    prompts = load_prompts(argparse.Namespace(**settings))

    stop_sequences: List[List[int]] = []
    if settings.get("stop"):
        for s in settings["stop"]:
            stop_sequences.append(generator.tokenizer.encode(s, allowed_special="all"))

    stop_tokens = [generator.eot_token]
    if settings.get("format") == "harmony":
        stop_tokens.append(generator.call_token)

    for idx, prompt_text in enumerate(prompts, start=1):
        print("=" * 60)
        print(f"MoE-Xtend Generator ({idx}/{len(prompts)})")
        print(f"DEBUG MODE IS {'ON' if DEBUG else 'OFF'}")
        print("=" * 60)

        print("\n[CONFIG]")
        print(f"  Checkpoint: {gen_cfg.checkpoint_path}")
        print(f"  Device: {gen_cfg.device}")
        print(f"  Prompt format: {settings['format']}")
        print(f"  Temperature: {sampling_cfg.temperature}")
        print(f"  Top-k: {sampling_cfg.top_k}")
        print(f"  Top-p: {sampling_cfg.top_p}")
        print(f"  Min-p: {sampling_cfg.min_p}")
        print(f"  Max tokens: {gen_cfg.max_tokens}")
        print()

        state = build_prefill_state(
            generator,
            prompt_text,
            gen_cfg,
            sampling_cfg,
            debug=DEBUG,
        )

        num_samples = int(settings.get("num_samples", 1))
        for sample_idx in range(1, num_samples + 1):
            if settings.get("seed") is not None:
                set_seed(int(settings["seed"]) + sample_idx - 1)

            if gen_cfg.stream and num_samples > 1:
                print(f"\n--- SAMPLE {sample_idx}/{num_samples} ---\n")

            result = decode_from_state(
                generator,
                state,
                gen_cfg,
                sampling_cfg,
                stop_tokens=stop_tokens,
                stop_sequences=stop_sequences,
                debug=DEBUG,
                return_logprobs=bool(settings.get("logprobs")),
            )

            print("\n" + "=" * 60)
            print(f"FINAL OUTPUT (sample {sample_idx}/{num_samples})")
            print("=" * 60)
            if not gen_cfg.stream:
                print(result["text"])

            metrics = result["metrics"]
            print("\n[METRICS]")
            print(f"  Prompt tokens: {metrics['prompt_tokens']}")
            print(f"  Generated tokens: {metrics['generated_tokens']}")
            print(f"  Prefill time: {metrics['prefill_time_sec']:.4f}s")
            print(f"  TTFT: {metrics['ttft_sec']:.4f}s")
            print(f"  Decode time: {metrics['decode_time_sec']:.4f}s")
            print(f"  Tokens/sec: {metrics['tokens_per_sec']:.2f}")

            if settings.get("parse_harmony"):
                try:
                    from openai_harmony import Role

                    parsed = generator.tokenizer.parse_messages_from_completion_tokens(
                        result["tokens"], role=Role.ASSISTANT
                    )
                    parsed_payload = serialize_harmony_messages(parsed)
                    print("\n[HARMONY PARSED]")
                    print(json.dumps(parsed_payload, indent=2))
                    if settings.get("parse_harmony_json"):
                        with open(settings["parse_harmony_json"], "w", encoding="utf-8") as f:
                            json.dump(parsed_payload, f, indent=2)
                except Exception as exc:
                    print(f"[WARN] Harmony parse failed: {exc}")

            if settings.get("extract_json") or settings.get("json_schema") or settings.get("json_output"):
                payload = extract_json(result["text"])
                if payload is None:
                    print("[WARN] No valid JSON object or array found in output.")
                else:
                    if settings.get("json_schema"):
                        errors = validate_schema(payload, settings["json_schema"])
                        if errors:
                            print("[WARN] JSON schema validation failed:")
                            for err in errors:
                                print(f"  - {err}")
                        else:
                            print("[OK] JSON schema validation passed.")
                    if settings.get("json_output"):
                        with open(settings["json_output"], "w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2)

            if gen_cfg.metrics_json:
                payload = {
                    "prompt_index": idx,
                    "sample_index": sample_idx,
                    "metrics": metrics,
                    "settings": {
                        "temperature": sampling_cfg.temperature,
                        "top_k": sampling_cfg.top_k,
                        "top_p": sampling_cfg.top_p,
                        "min_p": sampling_cfg.min_p,
                        "max_tokens": gen_cfg.max_tokens,
                    },
                }
                if len(prompts) == 1 and num_samples == 1:
                    with open(gen_cfg.metrics_json, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                else:
                    mode = "a" if (idx > 1 or sample_idx > 1) else "w"
                    with open(gen_cfg.metrics_json, mode, encoding="utf-8") as f:
                        f.write(json.dumps(payload) + "\n")

            if settings.get("logprobs_json") and result.get("logprobs") is not None:
                logprob_payload = {
                    "prompt_index": idx,
                    "sample_index": sample_idx,
                    "logprobs": result["logprobs"],
                }
                mode = "a" if (idx > 1 or sample_idx > 1) else "w"
                with open(settings["logprobs_json"], mode, encoding="utf-8") as f:
                    f.write(json.dumps(logprob_payload) + "\n")

            if settings.get("output_json"):
                output_payload = {
                    "prompt_index": idx,
                    "sample_index": sample_idx,
                    "text": result["text"],
                    "tokens": result["tokens"],
                    "metrics": result["metrics"],
                }
                if result.get("logprobs") is not None:
                    output_payload["logprobs"] = result["logprobs"]
                mode = "a" if (idx > 1 or sample_idx > 1) else "w"
                with open(settings["output_json"], mode, encoding="utf-8") as f:
                    f.write(json.dumps(output_payload) + "\n")


if __name__ == "__main__":
    main()
