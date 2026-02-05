<div align="center">

# MoE-Xtend

### Context Unbound. Intelligence Unleashed.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg?style=flat)](LICENSE)
[![PyTorch 2.5+](https://img.shields.io/badge/torch-2.5%2B-red.svg?style=flat)](requirments.txt)
[![Status: Research](https://img.shields.io/badge/status-research-blueviolet.svg?style=flat)](#)
[![Harmony](https://img.shields.io/badge/harmony-native-0ea5e9.svg?style=flat)](#)

<br/>

<img src="./assets/hero.svg" width="100%" alt="MoE-Xtend Hero"/>

</div>

---

## Overview
MoE-Xtend is a research-grade Mixture-of-Experts transformer stack engineered for **long-context stability**, **Harmony-native prompting**, and **precision-controlled inference**. The design emphasizes explicit routing, transparent sampling, and deterministic instrumentation (TTFT, throughput, logprobs, structured outputs).

**Key benefits**
- **Sparse compute, dense capacity**: top-k routing activates only the most relevant experts per token.
- **Long-context alignment**: YaRN + NTK-by-parts RoPE scaling maintains distant recall without softmax collapse.
- **Deterministic decoding controls**: temperature, top-k, top-p, min-p, penalties, and logit bias.
- **Production-ready I/O**: Harmony formatting, structured output validation, JSONL logs.

---

## System Flow
<img src="./assets/system-flow.png" width="100%" alt="System Flow"/>

Harmony formatting creates **structured inputs**, tokenizer converts to indices, the transformer stack alternates **attention** and **MoE** blocks, and a sampling layer deterministically generates tokens while preserving a KV cache for stable latency.

---

## Architecture
<img src="./assets/architecture.svg" width="100%" alt="Architecture"/>

The stack alternates **Attention → MoE** blocks. Attention layers blend full-context and sliding-window patterns, while MoE layers keep computation sparse without sacrificing total capacity.

---

## Token Stream (Decode)
<img src="./assets/token-stream-anim.svg" width="100%" alt="Token Stream"/>

Decode is a single-token loop: read KV cache → apply attention → route through MoE → compute logits → sample next token. The cache offset advances per token and drives absolute RoPE positions.

---

## MoE Routing
<img src="./assets/moe-routing-anim.svg" width="100%" alt="MoE Routing"/>

Routing is token-local: a linear router produces expert scores, **top-k** experts are selected, and their outputs are merged by a normalized weighted sum.

### Router Math
<img src="./assets/router-math.svg" width="100%" alt="Router Math"/>

Let $x_t$ be the token hidden state. The router computes:

$$s = W_r x_t$$
$$I = \text{topk}(s, k)$$
$$w = \text{softmax}(s[I])$$
$$y = \sum_{i \in I} w_i \cdot E_i(x_t)$$

### Expert Activation (Illustrative)
<img src="./assets/moe-heatmap.png" width="100%" alt="MoE Heatmap"/>

### Expert Heat Pulse
<img src="./assets/moe-heat-pulse.svg" width="100%" alt="MoE Heat Pulse"/>

These visuals emphasize sparse, dynamic activation: only a subset of experts fire per token, but global capacity remains high.

---

## Attention Stack
<img src="./assets/attention-stack.svg" width="100%" alt="Attention Stack"/>

The attention pipeline combines **QKV projection**, **RoPE rotations**, **Grouped Query Attention**, **sliding-window masking**, and **attention sinks**.

### GQA Head Grouping
<img src="./assets/gqa-heads.svg" width="100%" alt="GQA Heads"/>

Grouped Query Attention shares key/value heads across multiple query heads, reducing memory bandwidth while preserving query diversity.

### Attention Matrix
<img src="./assets/attention-matrix-anim.svg" width="100%" alt="Attention Matrix"/>

A moving diagonal band reflects sliding-window attention while preserving causal structure. Full-context layers expand the band toward the full matrix.

---

## RoPE + YaRN Math
<img src="./assets/rope-rotation-anim.svg" width="100%" alt="RoPE Rotation"/>

RoPE base frequency schedule:

$$\theta_i = b^{-2i/d}$$

Wavelength:

$$\lambda_i = \frac{2\pi}{\theta_i}$$

YaRN introduces a length scaling factor and a concentration term that prevents softmax sharpening when contexts grow. NTK-by-parts then selectively blends between base RoPE and interpolated RoPE for different frequency bands.

SwiGLU core:

$$\mathrm{SwiGLU}(x) = \sigma(\alpha x_{\mathrm{glu}}) \cdot (x_{\mathrm{linear}} + 1)$$

### NTK-by-parts Scaling Zones
<img src="./assets/scaling-zones.svg" width="100%" alt="Scaling Zones"/>

Define:

$$r(i) = \frac{L \cdot \theta_i}{2\pi}$$

Fast clocks preserve local structure, slow clocks preserve global structure, and the mid band blends the two regimes.

---

## KV Cache
<img src="./assets/kv-cache-anim.svg" width="100%" alt="KV Cache"/>

Prefill builds the cache once; decode appends a single token per step, keeping latency steady for long generations.

### KV Cache Memory Map
<img src="./assets/kv-cache-map.svg" width="100%" alt="KV Cache Map"/>

The cache is a 4D tensor indexed by batch, context, KV head, and head dimension. A monotonically increasing offset points to the next write location.

---

## Sampling Controls
<img src="./assets/sampling-controls.svg" width="100%" alt="Sampling Controls"/>

MoE-Xtend exposes temperature, top-k, top-p, min-p, repetition penalties, and logit biasing to control creativity and determinism.

---

## Structured Outputs
- Harmony parsing can recover structured messages from completion tokens.
- JSON extraction + schema validation enforce downstream formats.
- Logprobs and JSONL outputs make analysis and regression testing stable.

---

## Evaluation
<img src="./assets/evals.svg" width="100%" alt="Evaluation"/>

Included evals focus on long-context retrieval and stability:
- Needle-in-haystack retrieval
- Passkey retrieval
- TTFT and throughput reporting

---

## Pseudocode: MoE Routing
```text
# x_t: token hidden state
s = W_r @ x_t
I = topk(s, k)
w = softmax(s[I])

out = 0
for i in I:
  out += w[i] * Expert_i(x_t)
```

## Pseudocode: Attention + GQA
```text
Q = X @ W_q
K = X @ W_k
V = X @ W_v

# group queries to KV heads
Q = reshape(Q, heads_q, groups, d)
K = reshape(K, heads_kv, d)
V = reshape(V, heads_kv, d)

A = softmax((Q K^T)/sqrt(d) + sinks + mask)
Y = A @ V
```

## Pseudocode: KV Cache Update
```text
# prefill stage
cache_k[offset:offset+t] = K
cache_v[offset:offset+t] = V
offset += t

# decode stage
cache_k[offset] = k_t
cache_v[offset] = v_t
offset += 1
```

---

## Quick Start
### 1. Install Dependencies
```bash
pip install -r requirments.txt
```

### 2. Run Inference
```bash
python3 inference.py --format harmony --prompt "Design a scheduling agent."
```

### 3. Multi-sample with shared prefill
```bash
python3 inference.py --prompt "Summarize MoE routing" --num_samples 3 --logprobs --output_json outputs.jsonl
```

### 4. Start Local Responses Server
```bash
python3 server.py --checkpoint /path/to/ckpt --port 8000
```

---

## CLI Highlights
- `--format harmony` uses Harmony format for optimal GPT-OSS behavior.
- `--num_samples` reuses prefill cache for faster multi-sample decoding.
- `--logprobs` and `--output_json` enable research-grade analysis.
- `--extract_json` and `--json_schema` enforce structured outputs.
- `--prefill_chunk` helps with very long prompts by chunking prefill.

---

## Repository Layout
```
assets/                       # Visual assets (SVG/PNG)
evals/                        # Long-context eval scripts
inference.py                  # Harmony-native inference + sampling
model.py                      # Transformer + MoE + RoPE
weights.py                    # MXFP4 decoding utilities
server.py                     # Local Responses-style server
requirments.txt               # Dependencies
LICENSE
README.md
```

---

## Roadmap
- MXFP4-aware MoE kernels for full quantized inference speed.
- Triton attention and CUDA graph optimizations.
- Long-context calibration (LongRoPE2 / Q-ROAR style).
- Expanded eval suite for instruction following and tool use.

---

## License
MIT. See `LICENSE` for details.
