// CPU reference convolution (NCHW, float32).
//
// Written for clarity over speed: it is the ground truth that the HIP
// kernel is checked against, and the baseline the benchmark measures.

#include "conv2d.hpp"

void conv2d_cpu(const float* input, const float* weight, const float* bias,
                float* output, const Conv2DParams& p) {
    const int OH = conv_out_h(p);
    const int OW = conv_out_w(p);

    for (int n = 0; n < p.N; ++n) {
        for (int k = 0; k < p.K; ++k) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float acc = bias[k];
                    for (int c = 0; c < p.C; ++c) {
                        for (int r = 0; r < p.R; ++r) {
                            for (int s = 0; s < p.S; ++s) {
                                const int ih = oh * p.stride - p.pad + r;
                                const int iw = ow * p.stride - p.pad + s;
                                float v = 0.0f;
                                if (ih >= 0 && ih < p.H && iw >= 0 && iw < p.W) {
                                    v = input[((std::size_t)n * p.C + c) * p.H * p.W +
                                              (std::size_t)ih * p.W + iw];
                                }
                                v *= weight[((std::size_t)k * p.C + c) * p.R * p.S +
                                            (std::size_t)r * p.S + s];
                                acc += v;
                            }
                        }
                    }
                    output[((std::size_t)n * p.K + k) * OH * OW +
                           (std::size_t)oh * OW + ow] = acc;
                }
            }
        }
    }
}
