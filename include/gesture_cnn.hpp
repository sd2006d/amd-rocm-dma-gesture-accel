#ifndef AMD_GESTURE_CNN_HPP
#define AMD_GESTURE_CNN_HPP

// Small gesture-classification CNN forward path.
//
// Architecture (input: 1x64x64 grayscale frame):
//   conv1:  8 filters 5x5, stride 1, pad 0  ->  8x60x60, ReLU, maxpool 2 ->  8x30x30
//   conv2: 16 filters 3x3, stride 1, pad 0  -> 16x28x28, ReLU, maxpool 2 -> 16x14x14
//   fc:    3136 -> 5 gesture classes, softmax
//
// Classes: fist, palm, point, peace, thumbs_up.
//
// Honesty notes:
//   - The weights are deterministically pseudo-random; the network is
//     NOT trained, so predictions on real images are meaningless.
//     What is implemented and tested here is the forward data path.
//   - There is no camera/sensor input in this repo; tests feed
//     synthetic frames.

#include <cstdint>
#include <vector>

constexpr int GESTURE_IMG_H = 64;
constexpr int GESTURE_IMG_W = 64;
constexpr int GESTURE_NUM_CLASSES = 5;

const char* gesture_class_name(int cls);

struct GestureCNN {
    // conv1: [8][1][5][5], bias [8]
    std::vector<float> w1, b1;
    // conv2: [16][8][3][3], bias [16]
    std::vector<float> w2, b2;
    // fc: [5][3136], bias [5]
    std::vector<float> wfc, bfc;
};

// Deterministic pseudo-random initialisation (fixed seed) so tests
// and benchmarks are reproducible.
void gesture_cnn_init(GestureCNN& net, std::uint32_t seed = 0xC10Cu);

// Forward pass on one 64x64 frame. Writes class probabilities into
// probs_out[5] and returns the argmax class index.
int gesture_cnn_predict(const GestureCNN& net, const float* img, float* probs_out);

#endif  // AMD_GESTURE_CNN_HPP
