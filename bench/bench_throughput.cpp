// Throughput benchmark. All timing is MEASURED wall-clock time
// (std::chrono::steady_clock); nothing is hardcoded.
//
// CPU path: always available.
// GPU path: compiled only when hipcc/ROCm is present at build time
// (Makefile auto-detects). Even then, if no AMD GPU is usable at
// runtime the GPU section reports "skipped" instead of failing.
//
// Run: ./build/bench_throughput

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "conv2d.hpp"
#include "dma_engine.hpp"
#include "gesture_cnn.hpp"

#ifdef USE_ROCM
#include "conv2d_hip.hpp"
#endif

namespace {

double bench_ms(const std::function<void()>& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::steady_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return total_ms / iters;
}

std::vector<float> deterministic_fill(std::size_t n, unsigned seed) {
    std::vector<float> v(n);
    unsigned s = seed ? seed : 1u;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = ((s >> 9) & 0xFFFFu) * (1.0f / 65536.0f) - 0.5f;
    }
    return v;
}

void report(const char* name, double ms_per_iter) {
    std::printf("%-42s %10.3f ms/iter   %10.1f iters/sec\n", name,
                ms_per_iter, 1000.0 / ms_per_iter);
}

}  // namespace

int main() {
    std::printf("== CPU path (measured, wall clock) ==\n");

    // conv1 of the gesture CNN: 1x64x64 -> 8x60x60, 5x5 filters
    for (int batch : {1, 8, 32}) {
        Conv2DParams p{batch, 1, 64, 64, 8, 5, 5, 1, 0};
        auto in = deterministic_fill(conv_in_elems(p), 11u);
        auto w = deterministic_fill(conv_weight_elems(p), 22u);
        std::vector<float> b(8, 0.0f);
        std::vector<float> out(conv_out_elems(p));
        int iters = (batch <= 1) ? 200 : (batch <= 8 ? 40 : 10);
        double ms = bench_ms(
            [&] { conv2d_cpu(in.data(), w.data(), b.data(), out.data(), p); },
            5, iters);
        // Guard against dead-code elimination changing results.
        volatile float sink = out[out.size() / 2];
        (void)sink;
        char name[64];
        std::snprintf(name, sizeof(name), "conv1 5x5 (batch %d)", batch);
        report(name, ms);
    }

    // conv2 of the gesture CNN: 8x30x30 -> 16x28x28, 3x3 filters
    for (int batch : {1, 8, 32}) {
        Conv2DParams p{batch, 8, 30, 30, 16, 3, 3, 1, 0};
        auto in = deterministic_fill(conv_in_elems(p), 33u);
        auto w = deterministic_fill(conv_weight_elems(p), 44u);
        std::vector<float> b(16, 0.0f);
        std::vector<float> out(conv_out_elems(p));
        int iters = (batch <= 1) ? 200 : (batch <= 8 ? 40 : 10);
        double ms = bench_ms(
            [&] { conv2d_cpu(in.data(), w.data(), b.data(), out.data(), p); },
            5, iters);
        volatile float sink = out[out.size() / 2];
        (void)sink;
        char name[64];
        std::snprintf(name, sizeof(name), "conv2 3x3 (batch %d)", batch);
        report(name, ms);
    }

    // Full gesture-CNN forward pass, batch 1.
    {
        GestureCNN net;
        gesture_cnn_init(net, 0xC10Cu);
        auto img = deterministic_fill(64 * 64, 55u);
        float probs[GESTURE_NUM_CLASSES];
        double ms = bench_ms(
            [&] { gesture_cnn_predict(net, img.data(), probs); }, 5, 100);
        volatile float sink = probs[0];
        (void)sink;
        report("gesture_cnn_predict (batch 1)", ms);
    }

#ifdef USE_ROCM
    std::printf("\n== GPU path (measured, wall clock) ==\n");
    try {
        PinnedDMAEngine dma(4 * 1024 * 1024);  // 4 MB chunk buffers
        if (!dma.init() || !dma.gpu_backed())
            throw std::runtime_error("DMA engine has no GPU backend");
        for (int batch : {1, 8, 32}) {
            Conv2DParams p{batch, 1, 64, 64, 8, 5, 5, 1, 0};
            auto in = deterministic_fill(conv_in_elems(p), 11u);
            auto w = deterministic_fill(conv_weight_elems(p), 22u);
            std::vector<float> b(8, 0.0f);
            std::vector<float> out(conv_out_elems(p));
            int iters = (batch <= 1) ? 200 : (batch <= 8 ? 40 : 10);
            double ms = bench_ms(
                [&] {
                    conv2d_hip(dma, in.data(), w.data(), b.data(),
                               out.data(), p);
                },
                3, iters);
            volatile float sink = out[out.size() / 2];
            (void)sink;
            char name[64];
            std::snprintf(name, sizeof(name), "conv1 5x5 HIP (batch %d)",
                          batch);
            report(name, ms);
        }
        dma.shutdown();
    } catch (const std::exception& e) {
        std::printf("GPU benchmark skipped: %s\n", e.what());
    }
#else
    std::printf("\n== GPU path ==\n");
    std::printf("skipped: built without ROCm (hipcc not found at build time)\n");
#endif

    std::printf("\nNote: no speedup ratio is reported. A speedup claim\n"
                "requires a measured GPU baseline on AMD hardware, which\n"
                "was not available for this build.\n");
    return 0;
}
