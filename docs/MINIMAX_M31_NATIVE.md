# Native MiniMax M3.1 development engine

This is an experimental text engine, built only with `STRATA_MINIMAX=ON`.
It is not a selectable serving family yet. Real M3.1 generation, throughput,
API serving, tool calls, vision, DSpark and concurrent conversations have not
been validated. Do not replace an existing serving engine with this prototype.

It uses Strata's GGUF mapping and GPU dequantizers with its own MiniMax forward
pass. It does not start or link the llama.cpp runtime. The initial products
are FP32 cuBLAS GEMVs over dequantized tiles, with a bounded cache of compressed
weight tiles per GPU. Weight mappings can be paged from disk; the engine does
not duplicate all experts into a host arena. This is a correctness baseline,
not an optimized prefill or expert scheduler.

## Build and run

CUDA 12 is required for P100. The small developer build used for validation is:

```sh
CUDA_ARCH=60 bash tools/minimax/build_cuda.sh build-minimax
build-minimax/strata-minimax --model /models/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf \
  --tokens 1,2,3 --context 4096 --n 8 --gpu 0,1,2 --cache-mib 512
```

All five shards must be present. `--tokens` takes model token IDs, not text.
`--gpu` preserves the specified order; consecutive layer ranges are divided
evenly across those devices. `--cache-mib` is a ceiling per GPU, in addition
to KV, indexer and working buffers. KV is FP32. Context allocation may fail
when there is insufficient free VRAM. There is no automatic context reduction.

For a normal CUDA source build, add `-DSTRATA_MINIMAX=ON` and build the
`strata-minimax` target. The full CMake build has not been exercised for this
change; the two-translation-unit developer build above has been compiled and
run. Build for each selected GPU architecture to avoid driver JIT compilation.

An experimental one-prompt text frontend uses the GGUF's own vocabulary and
chat template. It needs Strata's Python requirements (`regex`, `jinja2`):

```sh
python tools/minimax/chat.py --engine build-minimax/strata-minimax \
  --model /models/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf \
  --prompt 'Explain binary search briefly.' --max-tokens 64 --gpu 0,1,2
```

Generation is greedy. `--dump-all-logits FILE` on the engine writes little-endian
FP32 logits at every prompt position for differential checks. Prompt processing
is serial; no existing Qwen scheduling or serving defaults are changed.

## Model semantics

The GGUF contract checks dimensions, required tensors and payload bounds before
GPU allocation. Leading dense blocks refer to the FFN, not to attention:
M3.1 has a learned sparse indexer in every layer, including the first three.
The older M3 graph in llama.cpp at the reference revision below skips these
three indexers and must not be copied unchanged.

The forward pass implements per-head Q/K RMSNorm, partial NeoX RoPE, grouped-query
attention, causal block-max index scores, local-block forcing within the top-k
budget, dense leading FFNs, sigmoid routed experts with selection-only correction
bias, normalization and routed scaling, and an ungated shared expert. SwiGLU-OAI
uses alpha 1.702, gate upper clamp 7, up clamp [-7,7] and the up + 1 term.

The requested UD-IQ3_S is mixed precision: its directories contain Q6_K, IQ3_XXS,
IQ4_XS, Q8_0, F32, and NVFP4. The three NVFP4 down tensors have separate F32
per-expert `.scale` vectors, which are applied to their products. Missing or
invalid scales are refused. The model has 1191 tensors; the core contract has
1188, plus these three scale vectors.

MiniMax's `minimax-m2` tokenizer uses GPT4O splitting and NFC normalization.
The existing Qwen branch retains its expression and does not normalize text.

## Validation so far

`tools/minimax/test_native.py` writes tiny three-layer GGUFs and computes an
independent NumPy forward pass. The tests cover all-layer sparse attention,
partial rotary, more position blocks than the top-k budget, routed and shared
experts, and F32/BF16/Q8_0/NVFP4 weights with per-expert scales. On a P100,
19 positions in each format agreed within 1e-6 maximum absolute logit error.
NVFP4 also passed across two P100s and a mixed P100/2080 Ti/3060 split.
This does not establish correctness of the real 428B model.

```sh
python tools/minimax/test_native.py --engine build-minimax/strata-minimax --gpu 0
python tools/minimax/test_tokenizer.py METADATA_SHARD official-tokenizer.json
```

The tokenizer passed 509 exact token-ID comparisons against the official
tokenizer.json, including decomposed Unicode, digits, case boundaries, special
tokens and seeded random text. Existing tokenizer tests passed with three
real-Qwen-vocabulary cases skipped because their fixture was unavailable.

References: [MiniMax M3.1 model/config/tokenizer](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/262ae6830934f3512211081b8a3f16425b93a239),
[GGUF revision](https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF/tree/ce4d8620574874556f280f2bfd2f3445718d59d6),
[GGUF names and M3 graph reference](https://github.com/ggml-org/llama.cpp/blob/74b104c9ba5717116b7e65e0cd35397778653304/src/models/minimax-m3.cpp).
