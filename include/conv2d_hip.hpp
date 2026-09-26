#ifndef AMD_GESTURE_CONV2D_HIP_HPP
#define AMD_GESTURE_CONV2D_HIP_HPP

// HIP convolution entry point.
//
// NOTE: this translation unit is only compiled when ROCm/hipcc is
// available (see Makefile). It has never been compiled or run in this
// environment because no ROCm toolchain or AMD GPU was available.
// It is written against the public HIP runtime API and structured so
// that `hipcc` can build it unchanged.
//
// Data movement goes through PinnedDMAEngine with double buffering:
// the batch is processed in chunks; while chunk i is being convolved
// on the GPU, chunk i+1 is transferred on the other stream/slot.
// Weights are transferred once per call (a documented simplification:
// a production version would keep them resident across calls).

#include "conv2d.hpp"

class PinnedDMAEngine;

void conv2d_hip(PinnedDMAEngine& dma,
                const float* h_input,   // [N][C][H][W], host
                const float* h_weight,  // [K][C][R][S], host
                const float* h_bias,     // [K], host
                float* h_output,        // [N][K][OH][OW], host
                const Conv2DParams& p);

#endif  // AMD_GESTURE_CONV2D_HIP_HPP
