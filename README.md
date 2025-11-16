
# <PROJECT NAME> — Triton FlashAttention-Style Scaled Dot Product Attention

Custom [Triton](https://github.com/triton-lang/triton) kernels for Scaled Dot Product Attention (SDPA — Scaled Dot Product Attention), including both forward and backward passes, designed for Vision Transformer (ViT — Vision Transformer)–style workloads and small to medium sequence lengths.

> **Status:** WIP template — TODOs are marked throughout this file.  
> Replace placeholders like `<TODO: ...>` with your actual details.

---

## 1. Overview

This repository contains a custom FlashAttention-style implementation of Scaled Dot Product Attention (SDPA) in Triton:

- Fully custom **forward** and **backward** passes (gradients for query (Q), key (K), and value (V) are computed in Triton).
- Online softmax with log-sum-exp for stability in half precision.
- Tiling and autotuning tuned for ViT-like shapes (for example, sequence length around 197).

The goal is to serve both as:

- A **practical kernel** you can plug into PyTorch, and  
- A **readable reference** for learning how to implement attention in Triton.

---

## 2. Features

- **Full SDPA pipeline**
  - Forward: attention scores, online softmax, and output.
  - Backward: gradients for Q, K, V (no fallback to PyTorch autograd for core math).

- **Numerical stability**
  - Online softmax with running max and log-sum-exp in FP32.
  - Accumulation in FP32, cast back to original dtype (for example, `torch.bfloat16`).

- **Triton-specific optimizations**
  - Blocked tiling over queries and keys/values (for example, `BLOCK_Q` × `BLOCK_KV`).
  - Swizzled program IDs (for example, `GROUP_M` or `GROUP_N`) to improve load balancing.
  - Autotuning over:
    - Block sizes (for example, `BLOCK_Q`, `BLOCK_KV`)
    - Number of warps
    - Number of stages

- **PyTorch integration**
  - Simple functional API: `sdpa_triton_fa(q, k, v, ...)`.
  - Optional `torch.autograd.Function` wrapper for drop-in use in PyTorch modules.

## 3. Repository Structure

```text
.
├─ triton_fa_full.py        # Main Triton kernels and PyTorch wrapper
├─ bench.py                 # Microbenchmarks vs torch SDPA / FlashAttention
├─ tests/
│   ├─ test_correctness.py  # Output/gradient comparisons vs torch
│   └─ test_shapes.py       # Edge cases, dtypes, etc.
├─ examples/
│   └─ vit_demo.py          # Minimal ViT / transformer block using this kernel
├─ README.md                # This file
└─ pyproject.toml / setup.py (optional)  # Packaging (optional)
````


## 4. Installation

### 4.1. Requirements

* Python: `<TODO: version, e.g. 3.10+>`
* PyTorch: `<TODO: version, e.g. 2.2+>`
* Triton: `<TODO: version, e.g. 3.x>`
* GPU: NVIDIA GPU with CUDA `<TODO: version>` (tested on `<TODO: e.g. A100 80GB>`)

### 4.2. Quick start

```bash
git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
cd <REPO_NAME>

# (Optional) create and activate virtual environment here

pip install -r requirements.txt  # TODO: add file or list deps below
```

If you do not use `requirements.txt`, list core dependencies here:

```bash
pip install torch triton
# plus any extras you use in examples/tests, e.g.
pip install einops pytest
```

---

## 5. Usage

### 5.1. Basic example (drop-in SDPA — Scaled Dot Product Attention)

```python
import torch
from triton_FA_full import sdpa_triton_fa  # TODO: update import to your module path

B, H, N, D = 2, 8, 197, 64
dtype = torch.bfloat16
device = "cuda"

q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

# Triton FlashAttention-style SDPA
o = sdpa_triton_fa(q, k, v)  # TODO: update signature if different

loss = o.sum()
loss.backward()

print("Output shape:", o.shape)
print("Grad q mean:", q.grad.float().abs().mean().item())
```

### 5.2. Using the autograd Function class

If you expose a `torch.autograd.Function`, show how to wrap it in a module:

```python
import torch
import torch.nn as nn

from triton_FA_full import TritonAttention  # TODO: update import

class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: add projections / config as needed

    def forward(self, q, k, v):
        return TritonAttention.apply(q, k, v)

# Example
attn = AttentionLayer().cuda()
out = attn(q, k, v)
```

### 5.3. Example: plugging into a ViT block

> TODO: Add a short example in `examples/vit_demo.py` and reference it here.

---

## 6. Benchmarks

> This section is meant to be filled with your actual measurements.
> Below is a template you can populate.

### 6.1. Setup

* **Hardware:**

  * GPU: `<TODO: e.g. NVIDIA A100 80GB>`
  * CPU: `<TODO: model>`
* **Software:**

  * OS: `<TODO: e.g. Ubuntu 22.04>`
  * CUDA: `<TODO>`
  * PyTorch: `<TODO>`
  * Triton: `<TODO>`

### 6.2. Shapes and dtypes

We benchmark:

* Shapes: `(B, H, N, D)` in `<TODO: list shapes>`

  * For example: `(2, 6, 197, 64)`, `(4, 8, 512, 64)`, …
* Dtypes:

  * `torch.bfloat16`
  * `torch.float16`
  * (Optional) `torch.float32` for reference

### 6.3. Latency comparison

> **Note:** replace `X.XX` / `Y.YY` / `ZZ%` with your measured values.
> You can measure forward and backward separately in `bench.py`.

#### Forward + backward (single iteration, warm-cache)

| Shape (B,H,N,D) | Dtype | Torch SDPA time (ms) | Triton FA time (ms) | Speedup |
| --------------- | ----- | -------------------- | ------------------- | ------- |
| (2, 6, 197, 64) | bf16  | X.XX                 | Y.YY                | ~ZZ %   |
| (2, 6, 197, 96) | bf16  | X.XX                 | Y.YY                | ~ZZ %   |
| (4, 8, 512, 64) | fp16  | X.XX                 | Y.YY                | ~ZZ %   |
| …               | …     | …                    | …                   | …       |

You can optionally split this into:

* Forward-only latency
* Backward-only latency
* First call (compilation + autotuning) vs warm call

---

## 7. Correctness

We validate both **outputs** and **gradients** against `torch.nn.functional.scaled_dot_product_attention`.

### 7.1. Output comparison

For randomly initialized tensors:

* `max_abs_diff(O_triton, O_torch) = <TODO>`
* `max_rel_diff(O_triton, O_torch) = <TODO>`

over shapes `<TODO>` and dtypes `<TODO>`.

### 7.2. Gradient comparison

We compute a reference output using PyTorch SDPA, then backpropagate a simple scalar loss and compare:

* `max_abs_diff(dQ_triton, dQ_torch) = <TODO>`
* `max_abs_diff(dK_triton, dK_torch) = <TODO>`
* `max_abs_diff(dV_triton, dV_torch) = <TODO>`

See `tests/test_correctness.py` for details.

### 7.3. Optional: gradient check

For small shapes in `float32`, we can use `torch.autograd.gradcheck` to verify analytical vs numerical gradients.

> TODO: add small script or test snippet if you want this.

---

## 8. Implementation details and intuition

This section gives a high-level picture of how the Triton kernels are structured.
It is meant to be readable first, then you can dive into the code.

### 8.1. Intuition: forward pass (`_attn_fwd`)

The forward kernel `_attn_fwd` computes the standard attention operation

* “scores” = Q·Kᵀ (query times key transpose)
* apply softmax over keys
* multiply by V (values) to get the output

but does it in a memory- and cache-friendly way.

Roughly, each Triton program instance:

1. **Owns a small block of queries.**
   For example, a tile of shape `(BLOCK_Q, HEAD_DIM)` for a fixed batch and head.

2. **Streams over all keys and values in blocks.**
   Instead of materializing the full score matrix S (of shape `N × N`), it reads a `(BLOCK_KV, HEAD_DIM)` tile of K and V at a time, computes the partial scores between the local Q block and this K block, and immediately folds that into an **online softmax**.

3. **Uses online softmax with running statistics.**
   For each query row, the kernel keeps track of:

   * A running maximum of logits (for numerical stability).
   * A running normalized sum for the softmax denominator.
   * A running accumulator for the output row.

   When a new block of scores arrives, the kernel updates these running quantities, so it never needs to store the full scores S explicitly.

4. **Accumulates the output in FP32.**
   The partial contributions from each K/V block are accumulated into an output buffer in FP32. At the end, the result is cast back to the requested dtype (for example `bfloat16`).

5. **Writes out per-token statistics for backward.**
   Along the way, the kernel stores compact information (for example, per-query max/normalizer) into `M` (and optionally `D` or `L`, depending on your implementation).
   These saved tensors allow the backward pass to reconstruct the softmax probabilities without recomputing everything from scratch.

The result is a forward pass that:

* Never forms the full attention matrix in memory.
* Keeps most data in on-chip memory and registers.
* Is numerically stable even in half precision.

### 8.2. Intuition: backward pass

`_attn_bwd_preprocess`, `_attn_bwd_dk_dv`, `_attn_bwd_dq`

The backward pass is split into three kernels for clarity and performance:

#### 8.2.1. Preprocess: softmax “delta” (`_attn_bwd_preprocess`)

The kernel `_attn_bwd_preprocess` computes a per-token scalar that is needed for the softmax gradient. Intuitively, for each query position it:

* Looks at the upstream gradient `dO` (gradient of the loss with respect to the output) and the forward output `O`.
* Computes a summary term like “how much did this row’s softmax contribute overall”, which is used to form the softmax Jacobian efficiently in later kernels.

This is done once and stored in a compact tensor `D` that the next kernels can reuse.

#### 8.2.2. Gradients with respect to K and V (`_attn_bwd_dk_dv`)

The kernel `_attn_bwd_dk_dv` computes gradients for keys and values:

* Each Triton program instance **fixes a block of K/V positions** (for example, a range of key indices).
* For that fixed K/V block, it **loops over all query blocks**.

Inside the loop:

1. It **rebuilds the local part of the score matrix** S (or equivalently the logits) for this Q×K tile by recomputing Q·Kᵀ. This is similar to the forward pass, but limited to local tiles.
2. Using the saved forward statistics (for example, `M`) and the precomputed `D`, it reconstructs the needed softmax gradients for that tile.
3. It uses those to accumulate:

   * `dV` by mixing `dO` with the softmax probabilities.
   * `dK` by mixing the softmax gradient with Q.

Because each program “owns” a unique K/V block and only writes to its own rows of `dK` and `dV`, there is **no need for atomic operations**. This keeps the kernel simple and fast, even though it recomputes the local scores S one more time.

#### 8.2.3. Gradients with respect to Q (`_attn_bwd_dq`)

The kernel `_attn_bwd_dq` is symmetric to `_attn_bwd_dk_dv`, but with roles reversed:

* Each Triton program instance **fixes a block of Q positions**.
* For that block of queries, it **loops over all K/V blocks**.

Inside the loop:

1. It recomputes the same Q·Kᵀ tiles as in the other kernels.
2. Uses the saved forward statistics and `D` to reconstruct softmax gradients.
3. Accumulates contributions to `dQ` by mixing:

   * The softmax gradient.
   * The values V and upstream gradient `dO`.

Again, each program “owns” its unique slice of `dQ`, so there is no need for atomic additions.

Putting it all together:

* `_attn_bwd_preprocess` computes the softmax-related “delta” once.
* `_attn_bwd_dk_dv` and `_attn_bwd_dq` both **rebuild S locally from Q and K**, but in complementary directions (fixed K/V vs fixed Q).
* This design trades a small amount of extra compute (recomputing S) to avoid atomics and large intermediate storage, which tends to give **better throughput and simpler code** on modern GPUs.

### 8.3. More low-level notes (optional)

> TODO: If you want, add details here about:
>
> * Exact block sizes you found to work well.
> * Memory layouts and assumptions about strides.
> * How you use `tl.make_block_ptr`, `tl.dot`, etc.

---

## 9. Limitations and future work

Current limitations (fill with what actually applies):

* No support yet for:

  * Causal masking
  * Arbitrary attention masks
  * Dropout
  * Attention bias (for example, relative position bias or Continuous Position Bias (CPB — Continuous Position Bias))
  * Rotary Positional Embedding (RoPE — Rotary Positional Embedding)
* Kernel is tuned primarily for:

  * Sequence lengths around `<TODO: e.g. 197>`
  * Head dimensions in `<TODO: e.g. 64–128>`

Potential future extensions:

* Causal and windowed attention variants.
* Fused support for:

  * RoPE
  * Continuous position bias (CPB)
  * Two-dimensional RoPE for ViT patch grids.
* Better autotuning across a wider range of shapes and GPUs.
* Integration with `torch.compile` and custom backends.

---

## 10. Development

### 10.1. Running tests

```bash
pytest tests
```

> TODO: Add any extra test commands or environment variables.

### 10.2. Running benchmarks

```bash
python bench.py
```

> TODO: describe relevant CLI arguments (for example, shapes, dtypes).

---

## 11. What this project demonstrates (optional section)

You can keep this section to showcase skills to recruiters/clients, or move it to a separate document.

This project demonstrates experience with:

* Implementing a full **FlashAttention-style** SDPA pipeline in Triton:

  * Custom forward and backward kernels without delegating core work to PyTorch.
* Designing **online softmax** with log-sum-exp for numerical stability in half precision.
* Using Triton effectively:

  * Block tiling, `tl.dot`, `make_block_ptr` (if used), and swizzled program IDs.
  * Autotuning kernel configurations for specific workload regimes.
* Integrating custom kernels into PyTorch:

  * `torch.autograd.Function`
  * Correctness testing vs strong baselines (PyTorch SDPA).

Feel free to edit or delete this section if you prefer a more neutral README.

---

## 12. Acknowledgements

> TODO: Add references and thanks.

* PyTorch scaled dot product attention.
* FlashAttention and related work.
* Triton examples and documentation.

---

## 13. License

> TODO: Choose a license and link to it.

Common choices:

* MIT
* Apache-2.0

Example:

```text
This project is licensed under the MIT License. See LICENSE for details.
```

```
```
