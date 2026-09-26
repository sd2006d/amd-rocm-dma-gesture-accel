// HIP convolution with LDS (shared memory) tiling, driven through
// PinnedDMAEngine with double-buffered transfers.
//
// Each thread block computes a TP x TQ tile of the output for one
// output channel. For every input channel, the block cooperatively
// loads the input patch its tile needs into LDS (zero-filling the
// padding halo), synchronises, and each thread accumulates its
// R x S products from LDS instead of re-reading global memory.
//
// The batch is processed in chunks sized to the DMA engine's buffers.
// For chunk i on slot s = i % 2 the engine queues, in order on the
// slot's stream:
//
//     upload(input chunk) -> kernel -> download(output chunk)
//
// so transfer of chunk i+1 (other stream) overlaps compute and
// download of chunk i. Before reusing slot s for chunk i >= 2 we
// sync_slot(s), which is the only host-side synchronisation in the
// loop.
//
// NOTE: never compiled or executed -- no ROCm toolchain or AMD GPU
// was available in this environment. Written against the public HIP
// runtime API; build with hipcc on a ROCm machine (see Makefile).

#include "conv2d_hip.hpp"

#include "dma_engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#define HIP_CHECK_THROW(cmd)                                      \
    do {                                                          \
        hipError_t _e = (cmd);                                    \
        if (_e != hipSuccess)                                     \
            throw std::runtime_error(std::string("HIP error: ") + \
                                     hipGetErrorString(_e));      \
    } while (0)

// TP x TQ output tile per block; threads are (TQ, TP).
template <int TP, int TQ>
__global__ void conv2d_lds_kernel(const float* __restrict__ in,   // [N][C][H][W] device
                                 const float* __restrict__ wt,   // [K][C][R][S] device
                                 const float* __restrict__ bias, // [K] device
                                 float* __restrict__ out,        // [N][K][OH][OW] device
                                 int C, int H, int W, int K, int R, int S,
                                 int OH, int OW, int stride, int pad,
                                 int n0,        // first batch index of this chunk
                                 int oh_tiles)  // output-height tiles per image
{
    extern __shared__ float smem[];  // PH x PW input patch, dynamic size

    const int tx = threadIdx.x;  // 0..TQ-1 -> ow within tile
    const int ty = threadIdx.y;  // 0..TP-1 -> oh within tile
    const int k = blockIdx.x;    // output channel
    const int ow_tile = blockIdx.y;
    const int z = blockIdx.z;  // ln * oh_tiles + oh_tile
    const int ln = z / oh_tiles;
    const int oh_tile = z % oh_tiles;
    const int n = n0 + ln;

    const int oh = oh_tile * TP + ty;
    const int ow = ow_tile * TQ + tx;
    const bool active = (oh < OH && ow < OW);

    // Input rows/cols needed by this tile for one channel:
    // oh in [oh_tile*TP, oh_tile*TP + TP), ih = oh*stride - pad + r.
    const int PH = (TP - 1) * stride + R;
    const int PW = (TQ - 1) * stride + S;
    const int ih_base = oh_tile * TP * stride - pad;
    const int iw_base = ow_tile * TQ * stride - pad;

    const int tid = ty * TQ + tx;
    const int nthreads = TP * TQ;

    float acc = 0.0f;
    const float* in_n = in + (std::size_t)n * C * H * W;

    for (int c = 0; c < C; ++c) {
        const float* in_nc = in_n + (std::size_t)c * H * W;

        // Cooperative load of the PH x PW patch (halo zero-filled).
        for (int i = tid; i < PH * PW; i += nthreads) {
            const int ph = i / PW;
            const int pw = i % PW;
            const int ih = ih_base + ph;
            const int iw = iw_base + pw;
            smem[i] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                          ? in_nc[(std::size_t)ih * W + iw]
                          : 0.0f;
        }
        __syncthreads();

        if (active) {
            const float* w_kc = wt + ((std::size_t)k * C + c) * R * S;
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    acc += smem[(ty * stride + r) * PW + (tx * stride + s)] *
                           w_kc[r * S + s];
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        out[((std::size_t)n * K + k) * OH * OW + (std::size_t)oh * OW + ow] =
            acc + bias[k];
    }
}

void conv2d_hip(PinnedDMAEngine& dma,
                const float* h_input,
                const float* h_weight,
                const float* h_bias,
                float* h_output,
                const Conv2DParams& p) {
    if (!dma.gpu_backed())
        throw std::runtime_error(
            "conv2d_hip requires a ROCm-backed PinnedDMAEngine");

    const int OH = conv_out_h(p);
    const int OW = conv_out_w(p);
    const std::size_t in_img = (std::size_t)p.C * p.H * p.W;
    const std::size_t out_img = (std::size_t)p.K * OH * OW;

    // Weights are small and reused by every chunk: transfer once per
    // call. (A production version would keep them resident on the
    // device across calls; this is a documented simplification.)
    float* d_w = nullptr;
    float* d_b = nullptr;
    HIP_CHECK_THROW(hipMalloc(&d_w, conv_weight_elems(p) * sizeof(float)));
    try {
        HIP_CHECK_THROW(hipMalloc(&d_b, (std::size_t)p.K * sizeof(float)));
    } catch (...) {
        hipFree(d_w);
        throw;
    }
    HIP_CHECK_THROW(hipMemcpy(d_w, h_weight, conv_weight_elems(p) * sizeof(float),
                             hipMemcpyHostToDevice));
    HIP_CHECK_THROW(hipMemcpy(d_b, h_bias, (std::size_t)p.K * sizeof(float),
                             hipMemcpyHostToDevice));

    try {
        constexpr int TP = 16;
        constexpr int TQ = 16;
        const int oh_tiles = (OH + TP - 1) / TP;
        const int ow_tiles = (OW + TQ - 1) / TQ;

        // Batch chunking sized to the DMA engine's chunk buffers.
        const std::size_t floats_per_chunk = dma.chunk_floats();
        std::size_t imgs_per_chunk =
            std::min(floats_per_chunk / in_img, floats_per_chunk / out_img);
        if (imgs_per_chunk == 0) imgs_per_chunk = 1;

        const std::size_t shmem_bytes =
            (std::size_t)((TP - 1) * p.stride + p.R) *
            ((TQ - 1) * p.stride + p.S) * sizeof(float);

        for (std::size_t off = 0, i = 0; off < (std::size_t)p.N;
             off += imgs_per_chunk, ++i) {
            const int slot = static_cast<int>(i % 2);
            const std::size_t cn =
                std::min(imgs_per_chunk, (std::size_t)p.N - off);

            // Slot s still holds chunk i-2's download until it has
            // completed; wait before reusing it.
            if (i >= 2) dma.sync_slot(slot);

            dma.upload_chunk(h_input + off * in_img, cn * in_img, slot);

            hipStream_t stream =
                static_cast<hipStream_t>(dma.stream_handle(slot));
            dim3 block(TQ, TP, 1);
            dim3 grid((unsigned)p.K, (unsigned)ow_tiles,
                      (unsigned)(cn * oh_tiles));
            hipLaunchKernelGGL(conv2d_lds_kernel<TP, TQ>, grid, block,
                               shmem_bytes, stream, dma.device_in_ptr(slot),
                               d_w, d_b, dma.device_out_ptr(slot), p.C, p.H,
                               p.W, p.K, p.R, p.S, OH, OW, p.stride, p.pad,
                               (int)off, oh_tiles);

            dma.download_chunk(h_output + off * out_img, cn * out_img, slot);
        }
        dma.sync_all();
    } catch (...) {
        hipFree(d_w);
        hipFree(d_b);
        throw;
    }
    hipFree(d_w);
    hipFree(d_b);
}
