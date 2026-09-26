# CNN Gesture Acceleration: HIP Conv2D with LDS Tiling + Double-Buffered DMA

From-scratch 2D convolution for a small gesture-recognition CNN: a HIP GPU
kernel with LDS tiling, a pinned double-buffered DMA transfer engine, a CPU
reference implementation, and measured benchmarks. NCHW, float32 throughout.

## Status

- CPU path is built and tested. `conv2d_cpu` matches hand-computed values,
  the DMA engine's buffer logic is unit-tested, and the gesture-CNN forward
  pass is smoke-tested. Benchmarks below are measured wall-clock time on
  this machine.
- HIP path is written but I haven't been able to compile or run it, no
  ROCm toolchain or AMD GPU here. It's written against the public HIP
  runtime API and the Makefile picks it up automatically when `hipcc` is
  on PATH.
- No speedup numbers. I had no GPU to measure against.

## Layout

```
include/
  conv2d.hpp       shared conv descriptor (NCHW) + CPU reference declaration
  conv2d_hip.hpp   HIP entry point declaration (ROCm builds only)
  dma_engine.hpp   PinnedDMAEngine: double-buffered transfer engine
  gesture_cnn.hpp  5-class gesture CNN descriptor
src/
  conv2d_cpu.cpp   straightforward CPU reference conv2d
  conv2d_hip.cpp   HIP kernel: LDS-tiled conv2d, batch-chunked, DMA-driven
  dma_engine.cpp   engine implementation (HIP async copies, or memcpy shim)
  gesture_cnn.cpp  CNN forward path on CPU (conv->ReLU->pool x2, FC, softmax)
bench/
  test_correctness.cpp  hand-computed conv tests, DMA shim test, CNN smoke test
  bench_throughput.cpp  measured wall-clock benchmark, ROCm auto-detect
Makefile               auto-detects hipcc; CPU-only fallback otherwise
```

## Design notes

**LDS-tiled kernel** (`conv2d_hip.cpp`): each thread block computes a 16x16
tile of the output for one output channel. Per input channel the block
cooperatively loads the input patch its tile needs into LDS (zero-filled
halo for padding), syncs, then each thread accumulates its R*S products
from shared memory instead of re-reading global memory. Dynamic shared
memory keeps stride a runtime parameter.

**DMA engine** (`dma_engine.*`): two slots, each with input/output buffers
and its own HIP stream. `conv2d_hip` chunks the batch to fit the engine's
buffers and per chunk `i` on slot `s = i % 2` queues
`upload -> kernel -> download` in order on that slot's stream, so chunk
`i+1`'s transfer overlaps chunk `i`'s compute. The only host-side sync is
`sync_slot(s)` before a slot gets reused. Weights go up once per call;
a production version would keep them resident. In CPU-only builds the
engine compiles to a memcpy-backed shim with identical slot/chunk
bookkeeping, so the double-buffering logic itself is tested. The shim is
not part of the measured CPU benchmark.

**Gesture CNN** (`gesture_cnn.*`): 1x64x64 in -> conv(8, 5x5) -> ReLU ->
maxpool2 -> conv(16, 3x3) -> ReLU -> maxpool2 -> FC 3136->5 -> softmax over
{fist, palm, point, peace, thumbs_up}. Weights are deterministic
pseudo-random and untrained, so predictions are meaningless. What's tested
is the forward data path. No camera input; tests use synthetic frames.

## Build and test

```sh
make            # auto-detects hipcc; CPU-only if absent
make test       # correctness tests
make bench      # throughput benchmark
make clean
```

On a ROCm machine with `hipcc` on PATH, the same `make` compiles
`conv2d_hip.cpp` with `-DUSE_ROCM` and links with `hipcc`; the benchmark
then measures the GPU path too (or reports it skipped if no AMD GPU is
usable at runtime).

## Measured results

CPU-only build, `g++ 13.3.0 -O2`, 2 vCPUs (AMD EPYC 9D64), Ubuntu 24.04.
Wall-clock mean per iteration, warmup excluded. Values carry a `~` because
this is a shared cloud VM and runs vary a few tens of percent run to run.

| Workload                        | ~ms/iter | ~iters/sec |
|---------------------------------|----------|------------|
| conv1 5x5, batch 1 (8x60x60)    | 1.9      | 540        |
| conv1 5x5, batch 8              | 15       | 67         |
| conv1 5x5, batch 32             | 73       | 14         |
| conv2 3x3, batch 1 (16x28x28)   | 3.9      | 256        |
| conv2 3x3, batch 8              | 19       | 53         |
| conv2 3x3, batch 32             | 77       | 13         |
| full gesture-CNN forward, b=1   | 4.4      | 228        |

The CPU reference is intentionally naive (seven nested loops, no SIMD or
blocking). It's the correctness baseline, not a performance target.

## TODO

- Compile and run the HIP path on a ROCm machine with an AMD GPU. Check
  numerical parity against `conv2d_cpu` and measure real transfer/compute
  overlap.
- Keep weights resident on-device across calls instead of re-uploading.
- Train the CNN (or load trained weights) before claiming anything about
  recognition accuracy.

## License

Apache 2.0, see [LICENSE](LICENSE).
