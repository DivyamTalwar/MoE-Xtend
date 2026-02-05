from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import torch

from inference import GenerationConfig, SamplingConfig, run_generation
from prompting import HarmonyMessage, default_system_message, render_harmony
from inference import TokenGenerator, parse_logit_bias


def build_prompt_from_request(payload: Dict[str, Any]) -> str:
    if "messages" in payload:
        messages = []
        for msg in payload["messages"]:
            messages.append(
                HarmonyMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    channel=msg.get("channel"),
                    to=msg.get("to"),
                )
            )
        return render_harmony(messages, add_assistant_start=True, assistant_channel="final")

    if "input" in payload:
        system = default_system_message()
        messages = [
            HarmonyMessage(role="system", content=system),
            HarmonyMessage(role="user", content=str(payload["input"])),
        ]
        return render_harmony(messages, add_assistant_start=True, assistant_channel="final")

    raise ValueError("Request must include 'messages' or 'input'.")


class ResponseHandler(BaseHTTPRequestHandler):
    server_version = "moe-xtend-local/0.1"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/responses":
            self._send_json(404, {"error": "Not Found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        try:
            prompt_text = build_prompt_from_request(payload)
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})
            return

        temperature = float(payload.get("temperature", self.server.default_temperature))
        max_tokens = int(payload.get("max_output_tokens", self.server.default_max_tokens))
        top_k = int(payload.get("top_k", self.server.default_top_k))
        top_p = float(payload.get("top_p", self.server.default_top_p))
        min_p = float(payload.get("min_p", self.server.default_min_p))
        repetition_penalty = float(payload.get("repetition_penalty", self.server.default_repetition_penalty))
        presence_penalty = float(payload.get("presence_penalty", self.server.default_presence_penalty))
        frequency_penalty = float(payload.get("frequency_penalty", self.server.default_frequency_penalty))
        logit_bias = {}
        if payload.get("logit_bias"):
            if isinstance(payload["logit_bias"], dict):
                logit_bias = {int(k): float(v) for k, v in payload["logit_bias"].items()}
            else:
                logit_bias = parse_logit_bias(payload.get("logit_bias"))

        stop_sequences: List[List[int]] = []
        for s in payload.get("stop", []) or []:
            stop_sequences.append(self.server.generator.tokenizer.encode(s, allowed_special="all"))

        stop_tokens = [self.server.generator.eot_token, self.server.generator.call_token]

        gen_cfg = GenerationConfig(
            checkpoint_path=self.server.checkpoint,
            device=self.server.device,
            dtype=self.server.dtype,
            max_tokens=max_tokens,
            prefill_chunk=self.server.prefill_chunk,
            truncate_prompt=self.server.truncate_prompt,
            max_prompt_tokens=self.server.max_prompt_tokens,
            stream=False,
            metrics_json=None,
            compile_model=False,
        )

        sampling_cfg = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalize_prompt=True,
            logit_bias=logit_bias,
        )

        result = run_generation(
            self.server.generator,
            prompt_text,
            gen_cfg,
            sampling_cfg,
            stop_tokens=stop_tokens,
            stop_sequences=stop_sequences,
            debug=False,
        )

        response = {
            "output_text": result["text"],
            "metrics": result["metrics"],
        }
        self._send_json(200, response)


class LocalServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, *, generator, **kwargs):
        super().__init__(server_address, handler_class)
        self.generator = generator
        for key, value in kwargs.items():
            setattr(self, key, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoE-Xtend local Responses-style server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default=os.environ.get("MOE_XTEND_CHECKPOINT", "./checkpoints/gpt-oss-20b/original"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--prefill_chunk", type=int, default=0)
    parser.add_argument("--truncate_prompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    generator = TokenGenerator(
        checkpoint=args.checkpoint,
        device=device,
        dtype=args.dtype,
        compile_model=False,
        debug=False,
    )

    server = LocalServer(
        (args.host, args.port),
        ResponseHandler,
        generator=generator,
        checkpoint=args.checkpoint,
        device=device,
        dtype=args.dtype,
        prefill_chunk=args.prefill_chunk,
        truncate_prompt=args.truncate_prompt,
        max_prompt_tokens=args.max_prompt_tokens,
        default_temperature=args.temperature,
        default_max_tokens=args.max_tokens,
        default_top_k=args.top_k,
        default_top_p=args.top_p,
        default_min_p=args.min_p,
        default_repetition_penalty=args.repetition_penalty,
        default_presence_penalty=args.presence_penalty,
        default_frequency_penalty=args.frequency_penalty,
    )

    print(f"Serving on http://{args.host}:{args.port}/v1/responses")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
