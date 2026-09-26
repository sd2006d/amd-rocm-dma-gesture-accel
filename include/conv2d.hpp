#ifndef AMD_GESTURE_CONV2D_HPP
#define AMD_GESTURE_CONV2D_HPP

// Shared convolution descriptor and CPU reference interface.
// Data layout everywhere in this project is NCHW, float32.

#include <cstddef>

struct Conv2DParams {
    int N;  // batch size
    int C;  // input channels
    int H;  // input height
    int W;  // input width
    int K;  // output channels (filters)
    int R;  // filter height
    int S;  // filter width
    int stride = 1;
    int pad = 0;
};

inline int conv_out_h(const Conv2DParams& p) {
    return (p.H + 2 * p.pad - p.R) / p.stride + 1;
}

inline int conv_out_w(const Conv2DParams& p) {
    return (p.W + 2 * p.pad - p.S) / p.stride + 1;
}

inline std::size_t conv_in_elems(const Conv2DParams& p) {
    return (std::size_t)p.N * p.C * p.H * p.W;
}

inline std::size_t conv_weight_elems(const Conv2DParams& p) {
    return (std::size_t)p.K * p.C * p.R * p.S;
}

inline std::size_t conv_out_elems(const Conv2DParams& p) {
    return (std::size_t)p.N * p.K * conv_out_h(p) * conv_out_w(p);
}

// Straightforward, obviously-correct CPU reference implementation.
// Used as the ground truth for correctness tests and as the CPU
// benchmark baseline.
void conv2d_cpu(const float* input,   // [N][C][H][W]
                const float* weight,  // [K][C][R][S]
                const float* bias,     // [K]
                float* output,         // [N][K][OH][OW]
                const Conv2DParams& p);

#endif  // AMD_GESTURE_CONV2D_HPP
