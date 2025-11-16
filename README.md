# <PROJECT NAME> — Triton FlashAttention-Style Scaled Dot Product Attention

Custom [Triton](https://github.com/triton-lang/triton) kernels for Scaled Dot Product Attention (SDPA), including both forward and backward passes, designed for Vision Transformer (ViT)–style workloads and small to medium sequence lengths.

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

---

## 3. Repository Structure

> **Note:** This is a suggested layout. Adapt to your actual file names.

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
