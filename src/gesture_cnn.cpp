// Gesture CNN forward path (CPU), built on conv2d_cpu.
//
// conv1: 1x64x64 -> conv(8,5x5,s1,p0) -> 8x60x60 -> ReLU -> maxpool2 -> 8x30x30
// conv2: 8x30x30 -> conv(16,3x3,s1,p0) -> 16x28x28 -> ReLU -> maxpool2 -> 16x14x14
// fc:    3136 -> 5, softmax
//
// Weights are deterministically pseudo-random (untrained): predictions
// are meaningless on real images. This file implements and tests the
// forward data path, which is what the HIP kernel accelerates.

#include "gesture_cnn.hpp"

#include "conv2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {

// splitmix32: tiny deterministic PRNG for reproducible init.
std::uint32_t splitmix32(std::uint32_t& state) {
    std::uint32_t z = (state += 0x9E3779B9u);
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

float rand_uniform(std::uint32_t& state, float lo, float hi) {
    float u = (splitmix32(state) >> 8) * (1.0f / 16777216.0f);
    return lo + u * (hi - lo);
}

void relu_inplace(float* data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        if (data[i] < 0.0f) data[i] = 0.0f;
}

// 2x2 max-pool, stride 2. Input [C][H][W] with even H, W.
void maxpool2x2(const float* in, float* out, int C, int H, int W) {
    const int OH = H / 2, OW = W / 2;
    for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                float m = in[((std::size_t)c * H + 2 * oh) * W + 2 * ow];
                m = std::max(m, in[((std::size_t)c * H + 2 * oh) * W + 2 * ow + 1]);
                m = std::max(m, in[((std::size_t)c * H + 2 * oh + 1) * W + 2 * ow]);
                m = std::max(m, in[((std::size_t)c * H + 2 * oh + 1) * W + 2 * ow + 1]);
                out[((std::size_t)c * OH + oh) * OW + ow] = m;
            }
        }
    }
}

}  // namespace

const char* gesture_class_name(int cls) {
    static const char* names[GESTURE_NUM_CLASSES] = {
        "fist", "palm", "point", "peace", "thumbs_up"};
    if (cls < 0 || cls >= GESTURE_NUM_CLASSES) return "unknown";
    return names[cls];
}

void gesture_cnn_init(GestureCNN& net, std::uint32_t seed) {
    std::uint32_t st = seed ? seed : 1u;
    auto fill = [&](std::vector<float>& v, float scale) {
        for (float& x : v) x = rand_uniform(st, -scale, scale);
    };
    net.w1.assign(8 * 1 * 5 * 5, 0.0f);
    net.b1.assign(8, 0.0f);
    net.w2.assign(16 * 8 * 3 * 3, 0.0f);
    net.b2.assign(16, 0.0f);
    net.wfc.assign(GESTURE_NUM_CLASSES * 3136, 0.0f);
    net.bfc.assign(GESTURE_NUM_CLASSES, 0.0f);
    fill(net.w1, 0.2f);
    fill(net.w2, 0.2f);
    fill(net.wfc, 0.05f);
    // biases stay zero
}

int gesture_cnn_predict(const GestureCNN& net, const float* img,
                        float* probs_out) {
    // conv1: N=1,C=1,H=W=64,K=8,R=S=5 -> 8x60x60
    Conv2DParams p1{1, 1, GESTURE_IMG_H, GESTURE_IMG_W, 8, 5, 5, 1, 0};
    std::vector<float> c1(conv_out_elems(p1));
    conv2d_cpu(img, net.w1.data(), net.b1.data(), c1.data(), p1);
    relu_inplace(c1.data(), c1.size());

    std::vector<float> m1(8 * 30 * 30);
    maxpool2x2(c1.data(), m1.data(), 8, 60, 60);

    // conv2: N=1,C=8,H=W=30,K=16,R=S=3 -> 16x28x28
    Conv2DParams p2{1, 8, 30, 30, 16, 3, 3, 1, 0};
    std::vector<float> c2(conv_out_elems(p2));
    conv2d_cpu(m1.data(), net.w2.data(), net.b2.data(), c2.data(), p2);
    relu_inplace(c2.data(), c2.size());

    std::vector<float> m2(16 * 14 * 14);
    maxpool2x2(c2.data(), m2.data(), 16, 28, 28);

    // fc: 3136 -> 5, softmax
    float logits[GESTURE_NUM_CLASSES];
    for (int k = 0; k < GESTURE_NUM_CLASSES; ++k) {
        float acc = net.bfc[k];
        const float* w = net.wfc.data() + (std::size_t)k * 3136;
        for (int i = 0; i < 3136; ++i) acc += w[i] * m2[i];
        logits[k] = acc;
    }
    float maxl = logits[0];
    for (int k = 1; k < GESTURE_NUM_CLASSES; ++k)
        maxl = std::max(maxl, logits[k]);
    float sum = 0.0f;
    int best = 0;
    for (int k = 0; k < GESTURE_NUM_CLASSES; ++k) {
        probs_out[k] = std::exp(logits[k] - maxl);
        sum += probs_out[k];
        if (probs_out[k] > probs_out[best]) best = k;
    }
    for (int k = 0; k < GESTURE_NUM_CLASSES; ++k) probs_out[k] /= sum;
    return best;
}
