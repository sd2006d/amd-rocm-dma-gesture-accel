// Correctness tests. Everything here is checked against hand-computed
// values or structural invariants; nothing is compared against itself.
//
// Run: ./build/test_correctness

#include <cmath>
#include <cstdio>
#include <vector>

#include "conv2d.hpp"
#include "dma_engine.hpp"
#include "gesture_cnn.hpp"

static int failures = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            ++failures;                                                  \
        }                                                                \
    } while (0)

static bool near(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

// input 4x4 = 1..16, weight [[1,0],[0,-1]], bias 0, stride 1, pad 0.
// out[i][j] = in[i][j] - in[i+1][j+1] = -5 everywhere (3x3).
static void test_conv2d_hand_computed_stride1() {
    Conv2DParams p{1, 1, 4, 4, 1, 2, 2, 1, 0};
    std::vector<float> in(16);
    for (int i = 0; i < 16; ++i) in[i] = (float)(i + 1);
    float w[4] = {1, 0, 0, -1};
    float b[1] = {0};
    std::vector<float> out(9, 12345.0f);
    conv2d_cpu(in.data(), w, b, out.data(), p);
    for (float v : out) CHECK(near(v, -5.0f));
    std::printf("test_conv2d_hand_computed_stride1: %s\n",
                failures == 0 ? "ok" : "FAILED");
}

// Same tensors, stride 2 -> 2x2 output, also all -5.
static void test_conv2d_hand_computed_stride2() {
    Conv2DParams p{1, 1, 4, 4, 1, 2, 2, 2, 0};
    std::vector<float> in(16);
    for (int i = 0; i < 16; ++i) in[i] = (float)(i + 1);
    float w[4] = {1, 0, 0, -1};
    float b[1] = {0};
    std::vector<float> out(4, 12345.0f);
    conv2d_cpu(in.data(), w, b, out.data(), p);
    CHECK(conv_out_h(p) == 2 && conv_out_w(p) == 2);
    for (float v : out) CHECK(near(v, -5.0f));
    std::printf("test_conv2d_hand_computed_stride2: %s\n",
                failures == 0 ? "ok" : "FAILED");
}

// Multi-channel, multi-filter, bias.
// ch0 = all 1.0 (3x3), ch1 = all 2.0 (3x3); R=S=2, stride 1, pad 0 -> 2x2.
// k0: all weights 1.0, bias 0 -> 4*1 + 4*2 = 12 everywhere.
// k1: all weights 0.0, bias 1 -> 1.0 everywhere.
static void test_conv2d_multichannel_bias() {
    Conv2DParams p{1, 2, 3, 3, 2, 2, 2, 1, 0};
    std::vector<float> in(18);
    for (int i = 0; i < 9; ++i) in[i] = 1.0f;
    for (int i = 9; i < 18; ++i) in[i] = 2.0f;
    std::vector<float> w(2 * 2 * 2 * 2, 0.0f);
    for (int i = 0; i < 2 * 2 * 2; ++i) w[i] = 1.0f;  // k0 weights
    float b[2] = {0.0f, 1.0f};
    std::vector<float> out(8, 12345.0f);
    conv2d_cpu(in.data(), w.data(), b, out.data(), p);
    for (int i = 0; i < 4; ++i) CHECK(near(out[i], 12.0f));
    for (int i = 4; i < 8; ++i) CHECK(near(out[i], 1.0f));
    std::printf("test_conv2d_multichannel_bias: %s\n",
                failures == 0 ? "ok" : "FAILED");
}

// DMA engine shim (CPU-only build): round-trip data through both
// slots, alternating as the double-buffered pipeline does.
static void test_dma_engine_shim() {
    PinnedDMAEngine dma(4096);
    CHECK(dma.init());
    CHECK(!dma.gpu_backed());  // CPU-only build here
    CHECK(dma.chunk_floats() == 1024);

    std::vector<float> src(1024), dst(1024, 0.0f);
    for (int i = 0; i < 1024; ++i) src[i] = (float)i * 0.5f;

    // Simulate the pipeline's slot alternation: even chunks on
    // slot 0, odd chunks on slot 1.
    for (int chunk = 0; chunk < 4; ++chunk) {
        int slot = chunk % 2;
        dma.upload_chunk(src.data(), 1024, slot);
        // shim "kernel": copy in-buffer to out-buffer through the
        // engine's device pointers, like conv2d_hip would
        float* din = dma.device_in_ptr(slot);
        float* dout = dma.device_out_ptr(slot);
        CHECK(din != nullptr && dout != nullptr && din != dout);
        for (int i = 0; i < 1024; ++i) dout[i] = din[i] * 2.0f;
        dma.download_chunk(dst.data(), 1024, slot);
        for (int i = 0; i < 1024; ++i) CHECK(near(dst[i], src[i] * 2.0f));
    }
    dma.sync_slot(0);
    dma.sync_all();
    dma.shutdown();
    std::printf("test_dma_engine_shim: %s\n", failures == 0 ? "ok" : "FAILED");
}

// CNN forward smoke test: probabilities sum to 1, deterministic,
// class index in range.
static void test_gesture_cnn_forward() {
    GestureCNN net;
    gesture_cnn_init(net, 0xC10Cu);

    std::vector<float> img(64 * 64);
    for (int i = 0; i < 64 * 64; ++i)
        img[i] = ((i / 64) % 2 == 0) ? 1.0f : 0.0f;  // horizontal stripes

    float probs1[GESTURE_NUM_CLASSES], probs2[GESTURE_NUM_CLASSES];
    int c1 = gesture_cnn_predict(net, img.data(), probs1);
    int c2 = gesture_cnn_predict(net, img.data(), probs2);

    float sum = 0.0f;
    for (int k = 0; k < GESTURE_NUM_CLASSES; ++k) {
        sum += probs1[k];
        CHECK(probs1[k] >= 0.0f && probs1[k] <= 1.0f);
        CHECK(near(probs1[k], probs2[k], 1e-6f));  // deterministic
    }
    CHECK(near(sum, 1.0f, 1e-5f));
    CHECK(c1 == c2);
    CHECK(c1 >= 0 && c1 < GESTURE_NUM_CLASSES);
    std::printf("test_gesture_cnn_forward: predicted '%s', %s\n",
                gesture_class_name(c1), failures == 0 ? "ok" : "FAILED");
}

int main() {
    test_conv2d_hand_computed_stride1();
    test_conv2d_hand_computed_stride2();
    test_conv2d_multichannel_bias();
    test_dma_engine_shim();
    test_gesture_cnn_forward();
    if (failures == 0) {
        std::printf("\nALL TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d CHECK(S) FAILED\n", failures);
    return 1;
}
