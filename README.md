<div align="center">
  <img src="./assets/hero.svg" alt="MoE-Xtend" width="100%" />
</div>

<h1 align="center" style="font-size: 2.8em;">MoE-Xtend</h1>

<p align="center" style="font-size: 1.35em;"><strong>Context Unbound. Intelligence Unleashed.</strong></p>

<p align="center">
  <img src="./assets/badges/badge-python.svg" height="28" />
  <img src="./assets/badges/badge-torch.svg" height="28" />
  <img src="./assets/badges/badge-harmony.svg" height="28" />
  <img src="./assets/badges/badge-status.svg" height="28" />
  <img src="./assets/badges/badge-license.svg" height="28" />
</p>

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">OVERVIEW</div>

MoE-Xtend is a research-grade Mixture-of-Experts transformer stack built for **long-context stability**, **Harmony-native prompting**, and **precision-controlled inference**. The system is designed to be readable, rigorous, and production-aware, with explicit routing, transparent sampling, and deterministic metrics.

**Key Benefits**
- **Sparse compute, dense capacity**: top-k routing activates only the most relevant experts per token.
- **Long-context alignment**: YaRN + NTK-by-parts RoPE scaling stabilizes distant attention.
- **Deterministic decoding**: temperature, top-k, top-p, min-p, penalties, and logit bias.
- **Production-ready I/O**: Harmony formatting, structured outputs, JSONL logs, TTFT metrics.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">SYSTEM FLOW</div>

<img src="./assets/system-flow.png" width="100%" alt="System Flow"/>

Harmony formatting creates structured inputs, tokenization converts them to indices, the transformer stack alternates attention and MoE layers, and sampling enforces deterministic decoding. This makes failures easy to diagnose and performance measurable.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">ARCHITECTURE</div>

<img src="./assets/architecture.svg" width="100%" alt="Architecture"/>

The stack alternates **Attention → MoE** blocks. Attention layers blend full-context and sliding-window patterns, while MoE layers keep compute sparse without sacrificing capacity.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">TOKEN STREAM (DECODE)</div>

<img src="./assets/token-stream-anim.svg" width="100%" alt="Token Stream"/>

Decode is a single-token loop: read KV cache → apply attention → route through MoE → compute logits → sample next token. The cache offset advances per token to preserve absolute RoPE positions.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">MOE ROUTING</div>

<img src="./assets/moe-routing-anim.svg" width="100%" alt="MoE Routing"/>

Routing is token-local: a router projection computes expert scores, the top-k experts are selected, and their outputs are merged by normalized weights.

<img src="./assets/router-math.svg" width="100%" alt="Router Math"/>

**Router Math**

Let $x_t$ be the token hidden state:

$$s = W_r x_t$$
$$I = \text{topk}(s, k)$$
$$w = \text{softmax}(s[I])$$
$$y = \sum_{i \in I} w_i \cdot E_i(x_t)$$

<img src="./assets/moe-heatmap.png" width="100%" alt="MoE Heatmap"/>
<img src="./assets/moe-heat-pulse.svg" width="100%" alt="MoE Heat Pulse"/>

These visuals emphasize sparse, dynamic activation: only a subset of experts fire per token, but global capacity remains high.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">ATTENTION STACK</div>

<img src="./assets/attention-stack.svg" width="100%" alt="Attention Stack"/>

The attention pipeline combines **QKV projection**, **RoPE rotations**, **Grouped Query Attention**, **sliding-window masking**, and **attention sinks**.

<img src="./assets/gqa-heads.svg" width="100%" alt="GQA Heads"/>

Grouped Query Attention shares key/value heads across multiple query heads, reducing memory bandwidth while preserving query diversity.

<img src="./assets/attention-matrix-anim.svg" width="100%" alt="Attention Matrix"/>

A moving diagonal band reflects sliding-window attention while preserving causal structure. Full-context layers expand the band toward the full matrix.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">ROPE + YARN MATH</div>

<img src="./assets/rope-rotation-anim.svg" width="100%" alt="RoPE Rotation"/>

RoPE base frequency schedule:

$$\theta_i = b^{-2i/d}$$

Wavelength:

$$\lambda_i = \frac{2\pi}{\theta_i}$$

YaRN introduces a length scaling factor and a concentration term that prevents softmax sharpening when contexts grow. NTK-by-parts selectively blends between base RoPE and interpolated RoPE across frequency bands.

SwiGLU core:

$$\mathrm{SwiGLU}(x) = \sigma(\alpha x_{\mathrm{glu}}) \cdot (x_{\mathrm{linear}} + 1)$$

<img src="./assets/scaling-zones.svg" width="100%" alt="Scaling Zones"/>

Define:

$$r(i) = \frac{L \cdot \theta_i}{2\pi}$$

Fast clocks preserve local structure, slow clocks preserve global structure, and the mid band blends the two regimes.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">KV CACHE</div>

<img src="./assets/kv-cache-anim.svg" width="100%" alt="KV Cache"/>

Prefill builds the cache once; decode appends a single token per step, keeping latency steady for long generations.

<img src="./assets/kv-cache-map.svg" width="100%" alt="KV Cache Map"/>

The cache is a 4D tensor indexed by batch, context, KV head, and head dimension. A monotonically increasing offset points to the next write location.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">SAMPLING CONTROLS</div>

<img src="./assets/sampling-controls.svg" width="100%" alt="Sampling Controls"/>

Sampling parameters allow strict control over creativity and determinism: temperature, top-k, top-p, min-p, repetition penalties, and logit biasing.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">CORE FORMULAS</div>

<img src="./assets/formulas.svg" width="100%" alt="Formulas"/>

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">STRUCTURED OUTPUTS</div>

- Harmony parsing can recover structured messages from completion tokens.
- JSON extraction + schema validation enforce downstream formats.
- Logprobs and JSONL outputs make analysis and regression testing stable.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">EVALUATION</div>

<img src="./assets/evals.svg" width="100%" alt="Evaluation"/>

Included evals focus on long-context retrieval and stability.
- Needle-in-haystack retrieval
- Passkey retrieval
- TTFT and throughput reporting

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">PSEUDOCODE</div>

**MoE Routing**
```text
# x_t: token hidden state
s = W_r @ x_t
I = topk(s, k)
w = softmax(s[I])

out = 0
for i in I:
  out += w[i] * Expert_i(x_t)
```

**Attention + GQA**
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

**KV Cache Update**
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

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">QUICK START</div>

```bash
pip install -r requirments.txt
python3 inference.py --format harmony --prompt "Design a scheduling agent."
```

Multi-sample with shared prefill:
```bash
python3 inference.py --prompt "Summarize MoE routing" --num_samples 3 --logprobs --output_json outputs.jsonl
```

Local Responses-style server:
```bash
python3 server.py --checkpoint /path/to/ckpt --port 8000
```

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">CLI HIGHLIGHTS</div>

- `--format harmony` uses Harmony format for optimal GPT-OSS behavior.
- `--num_samples` reuses prefill cache for faster multi-sample decoding.
- `--logprobs` and `--output_json` enable research-grade analysis.
- `--extract_json` and `--json_schema` enforce structured outputs.
- `--prefill_chunk` helps with very long prompts by chunking prefill.

---

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">REPOSITORY LAYOUT</div>

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

<div align="center" style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.08em;">ROADMAP</div>

- MXFP4-aware MoE kernels for full quantized inference speed.
- Triton attention and CUDA graph optimizations.
- Long-context calibration (LongRoPE2 / Q-ROAR style).
- Expanded eval suite for instruction following and tool use.

---

## License
MIT. See `LICENSE` for details.
