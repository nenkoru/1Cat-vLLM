# RUSL-1873: concurrent decode, 16/24/32 clients

The current target is aggregate throughput and full warm turn latency at 16,
24 and 32 concurrent sessions, with 8,000 and 33,000 input tokens. The earlier
single-session fast-lane proposal was withdrawn. The first production baseline
is measured below; no candidate improvement is claimed.

## Evidence already available

The preceding RUSL-1873 run measured the production `.4` wheel on rented TP2
V100-SXM2 32 GB GPUs. Its reported 16 × 8k decode step was 33.3 ms:
NVFP4 GEMM 17.19 ms, lm_head 0.59, KVarN attention 7.37, GDN 2.36,
all-reduce 2.37, normalization/activation/quantization 2.14, b/a FP16 0.88,
copies/sampling 0.49, graph gaps 0.61, outside-graph gaps 0.12.
These profiler kernel sums and the unprofiled step are different measurements;
they must not be summed as an exact decomposition. The b/a prototype's reported
0.40 ms saving was below the predeclared 2% step gate. It was not deployed.
There are no complete 24/32-seat candidate matrices yet. New profiles are
reported below; the full 16/24/32 × 8k/33k profile set remains incomplete.

The new `.5` production baseline (16 seats, QPN2 enabled, TP2 V100-SXM2 32 GB)
completed on 15.10.2026 around 19:26 MSK. Three rounds per stage, 128 completion
tokens; throughput is the median of the rounds. Every round was valid, with
zero request errors/preemptions.

| Input tokens | Offered clients | Completion tok/s | Full-turn p90 range, s |
|---|---:|---:|---:|
| 8,000 | 16 | 284.14 | 7.19–7.20 |
| 8,000 | 24 | 256.10 | 11.95–11.99 |
| 8,000 | 32 | 284.91 | 14.33–14.34 |
| 33,000 | 16 | 133.28 | 15.34–15.41 |
| 33,000 | 24 | 126.87 | 24.15–24.27 |
| 33,000 | 32 | 133.76 | 30.53–30.56 |

Cached input tokens were 7,680/8,000 and 32,256/33,000. These full turns include
the recomputed prompt tail and queueing; they are not the earlier pure-decode
rate. HTTP concurrency is not proof of the number of active kernel rows. The
remaining candidate/restart comparisons, separate profiles and quality gates
are still required; this single baseline is not an acceptance result.

The matched 16-seat/QPN2-off control completed around 19:53 MSK; all 18 rounds
were valid with zero errors/preemptions. Median round throughput and p90:

| Input tokens | Clients | QPN2 on, tok/s | QPN2 off, tok/s | Off vs on | Off p90, s |
|---|---:|---:|---:|---:|---:|
| 8,000 | 16 | 284.14 | 277.14 | −2.46% | 7.37 |
| 8,000 | 24 | 256.10 | 245.62 | −4.09% | 12.49 |
| 8,000 | 32 | 284.91 | 276.64 | −2.90% | 14.77 |
| 33,000 | 16 | 133.28 | 128.34 | −3.70% | 15.95 |
| 33,000 | 24 | 126.87 | 121.72 | −4.06% | 25.19 |
| 33,000 | 32 | 133.76 | 128.79 | −3.72% | 31.73 |

These are two single server arms, not A/B/B/A and not a seat-limit result. Both
servers have 16 seats. They quantify why the QPN2 guard must be accounted for
when testing 24/32 seats. The corrected comparator independently revalidated
all 48 warmup/measurement records without changing the raw data. Candidate configurations remain separate pending work; profiler evidence is
reported below.

## Verified 32-row profile, 15.10.2026

The `.5` wheel, 32 seats and QPN2 disabled completed the 8k profile on both
ranks. The observation window had 32 running requests, zero waiting requests,
32 streams with first output and none finished before profiling stopped. Both
rank traces contain 1,920 GDN calls (40 steps × 48 layers), all with grid
`[24, 32, 8]` and block `[32, 2, 1]`. The kernel maps `blockIdx.y` to the token
row. Geometry and stream lifetime evidence together support actual 32-row decode.
The completed request round had zero errors/preemptions and 1,024 output tokens
per request. Profiling overhead means these client metrics are not an acceptance
benchmark.

| Per-step quantity, ms | Rank 0 | Rank 1 |
|---|---:|---:|
| Median graph-to-graph period | 46.553 | 46.550 |
| Mean NVFP4 GEMM kernel durations | 23.020 | 22.973 |
| Mean attention kernel durations | 12.212 | 12.210 |
| Mean GDN + convolution kernel durations | 4.067 | 4.079 |
| Mean all-reduce kernel durations | 2.505 | 2.556 |
| Mean normalization / activation / quantization | 1.955 | 1.958 |
| Mean b/a FP16 projection kernel durations | 0.689 | 0.684 |
| Mean graph idle intervals | 0.549 | 0.545 |
| Mean outside-graph idle intervals | 0.118 | 0.114 |

The analyzer uses 39 adjacent graph intervals from 40 captured steps (it omits
the final incomplete interval, not both boundaries). Kernel duration sums can
double-count overlap and are not an additive critical-path decomposition.
This QPN2-off profile cannot establish improvement over the 16-seat/QPN2-on
production configuration or rank proposed kernels by end-to-end benefit alone.

The initial 32 × 33k cold warmup exceeded the profiler client's hard-coded
600-second timeout: eight requests timed out while the server continued draining
the queue. This is category **(Б)**, a harness limit. The client now accepts a
positive `--timeout` (default 1800), saves `warmup.json` before validation, and
records the timeout in the profile window. A real delayed HTTP response test
reproduces the short-timeout failure and verifies success with sufficient time.
The outer rental controller deadline remains independent and takes precedence.

The single repaired 32 × 33k retry completed at 20:35 MSK, exit 0.
Cold warmup took 620.659 seconds, directly exceeding the old 600-second limit.
Warmup and the subsequent profile round were both valid. Both ranks
again show 1,920 GDN calls with grid `[24,32,8]`; all 32 streams were active,
none finished during the profile window, and errors/preemptions were zero.

| Per-step quantity, ms | Rank 0 | Rank 1 |
|---|---:|---:|
| Median graph-to-graph period, 33k | 78.280 | 78.278 |
| Mean graph-to-graph period, 33k | 79.367 | 79.361 |
| Mean attention kernel durations | 44.660 | 44.635 |
| Mean NVFP4 GEMM kernel durations | 23.324 | 23.285 |
| Mean GDN + convolution kernel durations | 4.075 | 4.081 |
| Mean all-reduce kernel durations | 2.529 | 2.626 |
| Mean normalization / activation / quantization | 1.993 | 1.965 |
| Mean graph idle intervals | 0.549 | 0.542 |

Long-context attention is the largest measured component at 32 rows. This
supports prioritizing batched attention/cache work alongside GEMM; it does not
prove a persistent-kernel speedup or disprove smaller fused phases. The full
24/32-seat A/B/B/A throughput matrix, 16/24-row profiles, numerical/quality
qualification and a new batched fused prototype remain outstanding.

The complete profile archive was exported and checksum-matched at 20:36 MSK
(SHA256 `560153552631409cacd846be1bd1cfd8daaacb2b3d4e140979ee0f898f0e7468`).
The allocated pair was released at 20:37 MSK after independent checks found
zero owned processes and 0 MiB on both GPUs. No additional rental runs follow.

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

A subsequent full extension build also succeeded in cluster job `kj-fc29ba18`
(exit 0): upstream `setup.py build_ext --inplace`, PyTorch `2.7.1+cu128`,
Python 3.11, CUDA 12.8, `MEGAKERNEL_CUDA_ARCH=sm_70`, 80 blocks and two build
workers. Both CUDA units and `torch_bindings.cpp` compiled and linked; importing
`qwen35_megakernel_bf16_C` and resolving its `decode` operator succeeded.
The resulting shared object SHA256 was
`ebf2932dc79456b69bb6d06f4a8a4c306f01b90bdafb9526b1486ff63917809e`.
This CPU build/import check did not execute kernels or load model weights.
The V100 `bench.py`, numerical comparison and 27B adaptation remain open;
this Python 3.11 artifact is not a production Python 3.12 wheel.

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

`profile_concurrency.py` observes each stream's first content and completion,
rejects a window if any stream finishes before recording stops, and captures
the configured 40-iteration trace separately from the latency matrix. The
trace analysis must still verify both ranks and graph shapes.

`compare_concurrency.py A1.json B1.json B2.json A2.json --out comparison.json`
requires a complete matrix, 128 output tokens, at least three rounds, matching
workload/prompt identity, wheel/model/GPU identity and repeated-arm argv. Each
manifest must have a distinct nonempty `arm` and timezone-qualified
`started_msk`; those recorded server starts must follow A/B/B/A chronology.
The operator verifies that they identify independent server runs. Reusing paths
or run identities, including copied JSON, is rejected. Missing identities are
not equality. The comparator checks every request's usage, errors and finite
timings, checks zero preemptions in every round, and recomputes saved aggregates
from rows/counters/elapsed time. It rejects partial or inconsistent
measurements. It reports per-round throughput as well as pooled latency
percentiles; preserve raw arm files to inspect restart variance. A performance
pass never implies the separate quality gate or rollout approval.

Local validation: `python -m unittest discover -s tools/rusl1873`.
Fourteen tests exercise multithreaded real HTTP streaming, multi-token chunks,
missing/short usage, absent/reset counters, tail-latency accounting, profile
stream lifetimes, and acceptance rejection for partial/mismatched/regressed data.
