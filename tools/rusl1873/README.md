# RUSL-1873: concurrent decode, 16/24/32 clients

The current target is aggregate throughput and full warm turn latency at 16,
24 and 32 concurrent sessions, with 8,000 and 33,000 input tokens. The earlier
single-session fast-lane proposal was withdrawn. No new GPU performance result
is claimed by this change.

## Evidence already available

The preceding RUSL-1873 run measured the production `.4` wheel on rented TP2
V100-SXM2 32 GB GPUs. Its reported 16 × 8k decode step was 33.3 ms:
NVFP4 GEMM 17.19 ms, lm_head 0.59, KVarN attention 7.37, GDN 2.36,
all-reduce 2.37, normalization/activation/quantization 2.14, b/a FP16 0.88,
copies/sampling 0.49, graph gaps 0.61, outside-graph gaps 0.12.
These profiler kernel sums and the unprofiled step are different measurements;
they must not be summed as an exact decomposition. The b/a prototype's reported
0.40 ms saving was below the predeclared 2% step gate. It was not deployed.
There are no 24/32-session measurements yet, nor 16/24/32 × 33k profiles.

## Source audit and actual SM70 compilation

Pinned sources inspected on 15.10.2026:

- [lucebox 6f2b179](https://github.com/Luce-Org/lucebox/tree/6f2b17958ebb49f2b7060ac0a40ce6509aa90039/optimizations/megakernel)
- [OpenJet 72ba4a7](https://github.com/L-Forster/open-jet/tree/72ba4a7e587ed0cda1312e3384e415962eaa0db3/megakernel)

Compilation used a CPU cluster job, `nvidia/cuda:12.8.1-devel-ubuntu22.04`,
2 CPU / 4 GiB; no GPU or model weights were used. This is compilation of CUDA
translation units, **not** a full PyTorch extension build, correctness test, or
`bench.py` run on V100.

| Source | Actual result with `-arch=sm_70` | Meaning |
|---|---|---|
| lucebox `kernel.cu`, `TARGET_SM=70`, 80 CTAs, 512 threads | compiled, exit 0; decode: 122 registers/thread, 18,320 bytes shared, zero spills | FP16 Volta path already exists; no need to invent a Turing-port blocker |
| lucebox `prefill.cu`, same flags | compiled, exit 0 | standalone CUDA prefill source compiles; linking/running remains |
| lucebox `kernel_gb10_nvfp4.cu` | exit 1, explicit `requires CUDA arch >= sm_120` | this NVFP4 path is Blackwell-specific, not a ready V100 implementation |
| OpenJet `cuda_compat.cuh`, small caller of `mk_cuda::mma_f32`, `QWEN_MK_SM75=1` | ptxas exit 255: `.m16n8k8` requires SM75+ | synchronous copy fallback alone is insufficient; MMA fragments need a Volta implementation |

The initial OpenJet probe omitted the `mk_cuda` namespace and failed in our
harness. That was corrected before the ptxas result above. It is not an upstream
failure. Full OpenJet build/benchmark is not claimed. Its single-session engine
experiment remains withdrawn; the architecture is still useful for analysis.

| Component | Reusable part | Work for 27B / NVFP4 / TP2 / batch |
|---|---|---|
| lucebox FP16 persistent loop | alternating DeltaNet/attention phases, scalar FP32 accumulation, SM70 fences/loads | fixed 0.8B constants (1024 hidden, 24 layers, 16 GDN heads) and scalar token/position API must become per-request/page metadata; cannot just raise layer count |
| grid synchronization | resident-CTA grid barrier and occupancy-based launch cap | 512 × 122 registers limits residency to one such CTA/SM on V100; batch tiles must fit that resident grid rather than create waiting CTAs |
| DeltaNet | fuse conv, gates, recurrence, gated normalization in the head phase | map separate request states, padding masks, 48 layers, TP heads and production state dtype; verify against the existing chain |
| attention | phase ordering and split of projections from cache reads | replace contiguous FP16 single-sequence cache access with paged KVarN and per-request sequence lengths; strongest long-context target remains attention |
| NVFP4 | group-scaled weight layout and fused projection/epilogue ideas | Blackwell file uses FP4/BF16-specific conversions and cublasLt path; retain/adapt our SM70 QPN2/TurboMind device work, not that file unchanged |
| TP2 | layer dependency ordering | neither source provides our cross-GPU collectives; a local grid barrier is not an all-reduce. Start with fused local phases bounded by existing collectives |
| OpenJet | 27B phase schedule; lazy replay of accepted GDN drafts | own implementation only (AGPL source not copied). MTP replay and DFlash comparison are already owned in RUSL-1764; avoid a second implementation here |

The compile/shape limitations are category **(Р)** (implementation needs
adaptation), not **(П)** (method cannot help). A 0.61 ms graph-gap observation
does not disprove a batched megakernel that also eliminates intermediate traffic
and combines small operators. Conversely, replicating a GEMV once per request
would lose weight reuse: the batched prototype needs tiled GEMM and measured
state/cache traffic. Expected whole-step gains remain estimates until measured.

Priority: (1) measure 24/32-seat configurations; (2) profile actual 16/24/32-row
decode at both lengths; (3) select the dominant local chain and integrate one
batched fused phase. GDN pre-processing remains a 1–2 day candidate, previously
estimated at 0.4–0.5 ms/step at 16 sessions, not a measured batch-32 result.
Full persistent-layer cost cannot be bounded responsibly from compile results.

## Launcher qualification trap

The production launcher hard-codes `SEQS=16`. Merely exporting `SEQS=32` does
not change it. Appending `--max-num-seqs` disables QPN2 by the existing guard.
The explicit experiment switch `VLLM_RUSL1873_SEQS=16|24|32` in the private
deployment branch exposes the limit, defaults to 16, and **retains that guard**.
It is not a production recommendation. Do not silently remove the QPN2 guard:
it records an earlier correctness qualification, not a tuning preference.

Record actual server argv, QPN2 state, utilization, wheel/model hashes, cache
capacity/dtype, graph sizes, GPU topology and clock state in each arm manifest.
Two comparisons are necessary:

1. Production 16/QPN2-on versus candidate 24/32/QPN2-off: actual deployable
   configurations, with all resulting configuration differences explicit.
2. Matched QPN2-off 16/24/32: isolates the seat-limit effect. This is diagnostic
   and cannot replace comparison to production.

Keeping QPN2 on at 24/32 is a separate candidate requiring numerical and quality
qualification. Comments in the linear layer mention M≤32 support; that is not
proof of end-to-end correctness for the production configuration.

## Client and predeclared gates

`concurrency.py` runs against an already allocated, isolated HTTP server. It
does not start/stop servers or modify installed files. Token prompts are
synthetic, generated deterministically and tokenized once per context. Prompts
are identical across arms and offered concurrency; each stage warms all prompts
then repeats full turns. This is a synthetic repeat-prompt measurement, not the
agent conversation quality suite. Client concurrency and server seat limit are
different axes.

Example (on the benchmark host):

```sh
python tools/rusl1873/concurrency.py --url http://127.0.0.1:8103 \
  --model MODEL_ALIAS --manifest verified-arm.json --out results/arm-a1.json
```

Manifest required keys: `arm`, `wheel_sha256`, `model_revision`, `server_argv`,
`seats`, `qpn2`, `gpu`. The operator must populate these from actual server
evidence; the client does not attest them. Add topology/utilization/cache fields.

Metrics: total completion tokens per wall second; median/p90 full turn and TTFT;
preemption counter delta; individual usage/cache counts and output hashes.
Missing/reset metrics and missing/short usage invalidate a round. SSE chunks
are never interpreted as tokens or engine steps. Output hashes detect changes
but are not a quality verdict. Missing prefix-cache usage remains unknown.

For each candidate use fresh server arms A/B/B/A, identical prompts and settings,
three rounds/stage, 128 output tokens, 16/24/32 offered clients × 8k/33k context.
Store every arm separately. Do not pool away restart variance. Before enabling:

- ≥5% aggregate throughput improvement at 24 and 32 offered clients;
- median and p90 full warm turn no worse than +5% in any profile;
- zero request errors/preemptions; mixbench quick 18/18;
- numerical correctness and review of the selected configuration;
- acceptance by Mika, tagged wheel/launcher and rollback bundle.

These are predeclared proposed gates, not a pass. Collect profiling separately
on a trace-confirmed steady decode plateau with 16/24/32 **active decode rows**,
each ≤ its server limit. HTTP client count/stream chunks do not prove batch size.
Use both ranks, label profiler overhead, report milliseconds and percentages,
and distinguish overlapped kernel sums from critical-path time. Warm full-turn
latency from this client cannot substitute for that kernel profile.

Local client validation: `python -m unittest discover -s tools/rusl1873`.
Five tests exercise multithreaded real HTTP streaming, multi-token chunks,
missing/short usage, absent/reset counters, and tail-latency accounting.
