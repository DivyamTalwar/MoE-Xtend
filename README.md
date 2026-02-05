<div align="center">
  <img src="./assets/hero.svg" alt="MoE-Xtend" width="100%" />
</div>

<h1 align="center">MoE-Xtend</h1>

<p align="center"><strong>Context Unbound. Intelligence Unleashed.</strong></p>

<p align="center">
  <img src="./assets/badges/badge-python.svg" height="28" />
  <img src="./assets/badges/badge-torch.svg" height="28" />
  <img src="./assets/badges/badge-harmony.svg" height="28" />
  <img src="./assets/badges/badge-status.svg" height="28" />
  <img src="./assets/badges/badge-license.svg" height="28" />
</p>

<p align="center">
  <strong>All visuals are local</strong> (no external URLs). <strong>SVG-first</strong> with GitHub-safe animations. <strong>Researcher-focused</strong>: shapes, formulas, failure modes.
</p>

---

<h2 align="center">CONTENTS</h2>

- [Overview](#overview)
- [Quick Start](#quick-start)
- [CLI Examples](#cli-examples)
- [Responses API (Local)](#responses-api-local)
- [System Flow](#system-flow)
- [Architecture](#architecture)
- [Token Stream (Decode)](#token-stream-decode)
- [MoE Routing](#moe-routing)
- [Attention Stack](#attention-stack)
- [RoPE + Scaling](#rope--scaling)
- [KV Cache](#kv-cache)
- [Sampling Controls](#sampling-controls)
- [Core Formulas](#core-formulas)
- [Evaluation](#evaluation)
- [Determinism + Debugging](#determinism--debugging)
- [Notation](#notation)
- [Repository Layout](#repository-layout)
- [Roadmap](#roadmap)

---

<h2 id="overview" align="center">OVERVIEW</h2>

MoE-Xtend is a **long-context MoE transformer system spec** with a heavy emphasis on:

- **Sparse compute, dense capacity** via top-k expert routing.
- **Long-context stability** via RoPE scaling (YaRN + NTK-by-parts) and explicit mask engineering.
- **Transparent inference** via deterministic sampling controls, logprobs, and regression-minded metrics.
- **Readable math**: every major component is paired with formulas and diagrams.

**What you can do with this repo**

- Run Harmony-native inference with strict sampling controls and measurable metrics (`inference.py`).
- Serve a minimal Responses-style endpoint for local integration tests (`server.py`).
- Stress long-context retrieval quickly with eval scripts (`evals/`).
- Use the diagrams as a technical spec that matches the implementation (this `README.md` + `assets/`).

The repo includes:

- `assets/`: all diagrams/animations (local, GitHub-safe)
- `inference.py`: Harmony-native inference + sampling + metrics/logprobs
- `server.py`: local Responses-style HTTP server
- `evals/`: retrieval + long-context sanity checks

---

<h2 id="quick-start" align="center">QUICK START</h2>

**Requirements**

- Python 3.9+
- A checkpoint directory containing `config.json` and `*.safetensors`

**Install**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (GPU kernels):

```bash
pip install -r requirements-gpu.txt
```

**Set checkpoint**

```bash
export MOE_XTEND_CHECKPOINT=/path/to/checkpoint
```

Expected checkpoint layout:

- `config.json`
- one or more `*.safetensors` files

**Run inference**

```bash
python3 inference.py \
  --checkpoint "$MOE_XTEND_CHECKPOINT" \
  --format harmony \
  --prompt "Design a scheduling agent." \
  --max_tokens 256
```

**Run local server**

```bash
python3 server.py --checkpoint "$MOE_XTEND_CHECKPOINT" --port 8000
```

**Run evals**

```bash
python3 evals/needle_haystack.py --checkpoint "$MOE_XTEND_CHECKPOINT"
python3 evals/passkey_retrieval.py --checkpoint "$MOE_XTEND_CHECKPOINT"
```

---

<h2 id="cli-examples" align="center">CLI EXAMPLES</h2>

**1) Multi-sample with shared prefill**

Prefill once, then decode multiple samples by restoring KV snapshots (fast for research sweeps):

```bash
python3 inference.py \
  --checkpoint "$MOE_XTEND_CHECKPOINT" \
  --format harmony \
  --prompt "Summarize MoE routing in 6 bullets." \
  --num_samples 4 \
  --max_tokens 180
```

**2) Logprobs + JSONL output (regression-friendly)**

```bash
python3 inference.py \
  --checkpoint "$MOE_XTEND_CHECKPOINT" \
  --format harmony \
  --prompt "Return a JSON object with keys: title, risks, mitigations." \
  --max_tokens 220 \
  --logprobs \
  --logprobs_json logprobs.jsonl \
  --output_json outputs.jsonl
```

**3) Extract JSON from model output**

```bash
python3 inference.py \
  --checkpoint "$MOE_XTEND_CHECKPOINT" \
  --format harmony \
  --prompt "Output ONLY valid JSON: {\\\"answer\\\": string, \\\"confidence\\\": number}." \
  --max_tokens 120 \
  --extract_json \
  --json_output extracted.json
```

**4) Validate extracted JSON against a schema**

```bash
python3 inference.py \
  --checkpoint "$MOE_XTEND_CHECKPOINT" \
  --format harmony \
  --prompt "Output ONLY valid JSON: {\\\"answer\\\": string, \\\"confidence\\\": number}." \
  --max_tokens 120 \
  --extract_json \
  --json_schema schema.json \
  --json_output extracted.json
```

---

<h2 id="responses-api-local" align="center">RESPONSES API (LOCAL)</h2>

Start the server:

```bash
python3 server.py --checkpoint "$MOE_XTEND_CHECKPOINT" --port 8000
```

Example request (messages):

```bash
curl -s http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON' | python3 -m json.tool
{
  "messages": [
    {"role": "system", "content": "You are a precise research assistant."},
    {"role": "user", "content": "Explain top-k MoE routing."}
  ],
  "temperature": 0.2,
  "max_output_tokens": 200,
  "top_p": 0.95,
  "min_p": 0.05
}
JSON
```

Example request (input):

```bash
curl -s http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input":"Summarize KV cache update rules.","max_output_tokens":120,"temperature":0.1}' \
  | python3 -m json.tool
```

Response shape:

```json
{
  "output_text": "...",
  "metrics": {
    "prefill_ms": 0.0,
    "decode_ms": 0.0,
    "tokens_generated": 0
  }
}
```

Notes:

- Stop tokens include Harmony `return` and `call` markers.
- `logit_bias` supports either a JSON object or `token_id:bias,...`.
- This server is intentionally minimal: it is meant for local testing and profiling loops.

---

<h2 id="system-flow" align="center">SYSTEM FLOW</h2>

<img src="./assets/system-flow.png" width="100%" alt="System Flow"/>

**High-level loop**

1. Harmony formatting builds structured prompts.
2. Tokenization produces ids + masks.
3. The stack alternates **attention** and **MoE** blocks.
4. KV cache is built at prefill, then appended during decode.
5. Sampling converts logits into a stable next-token choice.

---

<h2 id="architecture" align="center">ARCHITECTURE</h2>

<img src="./assets/architecture.svg" width="100%" alt="Architecture"/>

**Prefill vs decode**

- Prefill writes the prompt KV once. In full-attention layers, this is where the `O(T^2)` work lives.
- Decode is a single-token loop: append KV at `offset=t`, then `t++`. With sliding window layers, per-step attention becomes `O(W)`.

---

<h2 id="token-stream-decode" align="center">TOKEN STREAM (DECODE)</h2>

<img src="./assets/token-stream-anim.svg" width="100%" alt="Token Stream"/>

**Decode invariants**

- One token per step.
- KV cache is append-only.
- Absolute position matters: RoPE uses the global token index `t` (not local window index).

---

<h2 id="moe-routing" align="center">MOE ROUTING</h2>

<img src="./assets/moe-routing-anim.svg" width="100%" alt="MoE Routing"/>

<img src="./assets/router-math.svg" width="100%" alt="Router Math"/>

<img src="./assets/moe-heatmap.png" width="100%" alt="MoE Heatmap"/>

<img src="./assets/moe-heat-pulse.svg" width="100%" alt="MoE Heat Pulse"/>

**Canonical equations**

Let the token hidden state be `x_t`.

$$s = W_r x_t$$
$$I = \mathrm{topk}(s, k)$$
$$w = \mathrm{softmax}(s[I])$$
$$y = \sum_{i \in I} w_i \cdot E_i(x_t)$$

**Why routing is hard in practice**

- **Collapse**: without balancing pressure, most tokens pick the same experts.
- **Overflow**: with capacity constraints, many tokens may want the same expert at the same time.
- **Non-determinism**: floating-point ties in `topk` can cause unstable outputs unless tie-breaks are consistent.

**Pseudocode (routing)**

```text
# x_t: token hidden state
s = W_r @ x_t
I = topk(s, k)
w = softmax(s[I])

out = 0
for i in I:
  out += w[i] * Expert_i(x_t)
```

**Implementation notes (this repo)**

- `model.py`: `MLPBlock.forward` computes router logits, selects `topk`, normalizes weights, and evaluates only the selected experts via batched `einsum`.
- No expert-parallel all-to-all and no capacity enforcement. This is clarity-first, not a fused-kernel implementation.
- If you are comparing runs: `topk` ties + dtype can change routing, which changes everything downstream.

---

<h2 id="attention-stack" align="center">ATTENTION STACK</h2>

<img src="./assets/attention-stack.svg" width="100%" alt="Attention Stack"/>

<img src="./assets/gqa-heads.svg" width="100%" alt="GQA Heads"/>

<img src="./assets/attention-matrix-anim.svg" width="100%" alt="Attention Matrix"/>

**Attention equation**

$$A = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right), \quad Y = AV$$

Where `B` decomposes into additive bias terms:

- Causal mask
- Sliding window mask (for windowed layers)
- Sink bias (to stabilize long decode)

**Pseudocode (attention + GQA, schematic)**

```text
Q = X @ W_q
K = X @ W_k
V = X @ W_v

# grouped query attention: many Q heads share fewer KV heads
Q = reshape(Q, Hq, d)
K = reshape(K, Hkv, d)
V = reshape(V, Hkv, d)

B = causal_mask + window_mask + sink_bias
A = softmax((Q K^T)/sqrt(d) + B)
Y = A @ V
```

**Implementation notes (this repo)**

- `model.py`: `AttentionBlock.sdpa` constructs masks aligned to KV offset and appends a per-head sink bias as an extra softmax column.
- Sliding window is applied on alternating layers (`layer_idx % 2 == 0`) to mix dense and banded patterns.
- The easiest long-context bug is an off-by-one diagonal in the causal/window mask when switching prefill -> decode.

---

<h2 id="rope--scaling" align="center">ROPE + SCALING</h2>

<img src="./assets/rope-rotation-anim.svg" width="100%" alt="RoPE Rotation"/>

<img src="./assets/scaling-zones.svg" width="100%" alt="Scaling Zones"/>

**Base RoPE schedule**

$$\theta_i = b^{-2i/d}$$
$$\lambda_i = \frac{2\pi}{\theta_i}$$

**NTK-by-parts ratio**

$$r(i) = \frac{L \cdot \theta_i}{2\pi}$$

Interpretation:

- Fast clocks preserve local detail.
- Slow clocks preserve long-range structure.
- Mid-band blends regimes to avoid phase discontinuities.

**Implementation notes (this repo)**

- `model.py`: `RotaryEmbedding` precomputes `cos/sin` up to `max_content_length` and indexes with `(arange(seq_len) + offset) % max_content_length`.
- YaRN concentration is applied by scaling `cos/sin` (softens attention temperature as length grows).
- NTK-by-parts modifies inverse frequencies using alpha/beta cutpoints and a linear ramp blend.

---

<h2 id="kv-cache" align="center">KV CACHE</h2>

<img src="./assets/kv-cache-anim.svg" width="100%" alt="KV Cache"/>

<img src="./assets/kv-cache-map.svg" width="100%" alt="KV Cache Map"/>

**Memory scaling**

A useful back-of-the-envelope estimator:

$$\text{bytes} \approx 2 \cdot L \cdot B \cdot T \cdot H_{kv} \cdot d \cdot \text{dtype\_size}$$

- The factor `2` is for `K` and `V`.
- GQA reduces `H_kv`.
- Long contexts make `T` the dominant term.

**Implementation notes (this repo)**

- `model.py`: `Cache.extend` writes new K/V into preallocated tensors at indices `[offset:offset+t]`, then increments `offset`.
- `inference.py`: caches are allocated per layer sized to `min(len(prompt)+max_tokens, max_context)`.
- If you see shape mismatches or attention weirdness, check: KV head count (`H_kv`), `head_dim`, and the cache `offset`.

---

<h2 id="sampling-controls" align="center">SAMPLING CONTROLS</h2>

<img src="./assets/sampling-controls.svg" width="100%" alt="Sampling Controls"/>

**Core sampling relations**

$$p = \mathrm{softmax}(z/T)$$

- **Top-k**: keep k highest probability tokens.
- **Top-p**: keep smallest set with cumulative probability >= p.
- **Min-p**: drop tokens with probability below a fraction of the max token probability.

**Implementation notes (this repo)**

- `sampling.py` applies `logit_bias`, repetition/presence/frequency penalties, then truncation (`top_k`, `top_p`, `min_p`), then samples.
- Set `temperature=0` for greedy decode; otherwise sampling uses `torch.multinomial`.

---

<h2 id="core-formulas" align="center">CORE FORMULAS</h2>

<img src="./assets/formulas.png" width="100%" alt="Core Formulas"/>

<img src="./assets/formulas-notes.png" width="100%" alt="Formula Notes"/>

For a searchable ASCII reference inside SVG, see:

<img src="./assets/formulas.svg" width="100%" alt="Formulas (ASCII)"/>

---

<h2 id="evaluation" align="center">EVALUATION</h2>

<img src="./assets/evals.svg" width="100%" alt="Evaluation"/>

Recommended eval axes:

- Context length sweep (where do failures start?)
- Retrieval depth sweep (needle/passkey)
- TTFT distribution (p50/p95)
- Tokens/sec steady-state decode
- Regression tests on logprobs and structured outputs

---

<h2 id="determinism--debugging" align="center">DETERMINISM + DEBUGGING</h2>

If you want stable, comparable runs, treat determinism as a feature:

- Fix seeds and record every sampling parameter.
- Use stable `topk` tie-breaks.
- Prefer deterministic kernels when comparing regression outputs.
- Log `logprobs`, selected experts, and KV cache offsets.

Common long-context failure modes:

- Wrong absolute position indexing (RoPE mismatch between prefill and decode)
- Mask bugs (off-by-one in window, sinks applied to wrong tokens)
- KV cache layout mismatch (stride / head grouping)
- MoE capacity overflow (silent drops -> accuracy cliff)

---

<h2 id="notation" align="center">NOTATION</h2>

- `B`: batch size
- `T`: sequence length (time)
- `L`: transformer layers
- `H_q`: query heads
- `H_kv`: key/value heads (GQA)
- `d`: head dimension
- `W`: sliding window size
- `E`: number of experts
- `k`: experts per token (top-k routing)

**Shape conventions**

- Hidden states: `[B, T, hidden_size]`
- Cache per layer: `K,V: [B, T_cache, H_kv, d]`
- GQA grouping: `H_q = H_kv * groups`

---

<h2 id="repository-layout" align="center">REPOSITORY LAYOUT</h2>

```text
assets/                    # diagrams + animated SVGs (local)
evals/                     # long-context eval scripts
inference.py               # inference + sampling + logprobs/metrics
model.py                   # transformer + MoE + RoPE + KV cache
prompting.py               # Harmony message rendering helpers
sampling.py                # top-k/top-p/min-p + penalties
server.py                  # Responses-style local HTTP server
weights.py                 # safetensors checkpoint loader (MXFP4 decode)
requirements.txt
requirements-gpu.txt
requirments.txt            # compat shim (typo), includes requirements.txt
LICENSE
README.md
```

---

<h2 id="roadmap" align="center">ROADMAP</h2>

- Kernel-level optimizations (Triton attention, fused MoE dispatch/merge).
- Quantization-aware long-context runs (bandwidth first: KV + dequant).
- Long-context calibration sweeps with automated reporting.
- Expanded evals (instruction-following, tool use, structured extraction).

---

## License
MIT. See `LICENSE` for details.
