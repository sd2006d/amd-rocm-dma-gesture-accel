// PinnedDMAEngine implementation.
//
// ROCm build: device buffers + hipMemcpyAsync on per-slot streams.
// CPU-only build: memcpy-backed shim with identical slot/chunk
// bookkeeping so the double-buffering logic is testable without a GPU.

#include "dma_engine.hpp"

#include <cassert>
#include <cstring>
#include <new>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>

#define DMA_HIP_CHECK(cmd)                 \
    do {                                   \
        hipError_t _e = (cmd);             \
        if (_e != hipSuccess) return false; \
    } while (0)

struct PinnedDMAEngine::HipState {
    float* dev_in[2] = {nullptr, nullptr};
    float* dev_out[2] = {nullptr, nullptr};
    hipStream_t stream[2] = {nullptr, nullptr};
};
#endif

PinnedDMAEngine::PinnedDMAEngine(std::size_t chunk_bytes)
    : chunk_bytes_(chunk_bytes) {}

PinnedDMAEngine::~PinnedDMAEngine() { shutdown(); }

bool PinnedDMAEngine::init() {
    if (initialised_) return true;
    assert(chunk_bytes_ > 0);

#ifdef USE_ROCM
    hip_ = new (std::nothrow) HipState();
    if (!hip_) return false;
    for (int i = 0; i < 2; ++i) {
        DMA_HIP_CHECK(hipMalloc(&hip_->dev_in[i], chunk_bytes_));
        DMA_HIP_CHECK(hipMalloc(&hip_->dev_out[i], chunk_bytes_));
        DMA_HIP_CHECK(hipStreamCreate(&hip_->stream[i]));
    }
    gpu_backed_ = true;
#else
    for (int i = 0; i < 2; ++i) {
        host_in_[i] = new (std::nothrow) float[chunk_floats()];
        host_out_[i] = new (std::nothrow) float[chunk_floats()];
        if (!host_in_[i] || !host_out_[i]) {
            shutdown();
            return false;
        }
    }
    gpu_backed_ = false;
#endif

    initialised_ = true;
    return true;
}

void PinnedDMAEngine::shutdown() {
#ifdef USE_ROCM
    if (hip_) {
        for (int i = 0; i < 2; ++i) {
            if (hip_->stream[i]) hipStreamDestroy(hip_->stream[i]);
            if (hip_->dev_in[i]) hipFree(hip_->dev_in[i]);
            if (hip_->dev_out[i]) hipFree(hip_->dev_out[i]);
        }
        delete hip_;
        hip_ = nullptr;
    }
#else
    for (int i = 0; i < 2; ++i) {
        delete[] host_in_[i];
        host_in_[i] = nullptr;
        delete[] host_out_[i];
        host_out_[i] = nullptr;
    }
#endif
    gpu_backed_ = false;
    initialised_ = false;
}

float* PinnedDMAEngine::device_in_ptr(int slot) {
    assert(slot == 0 || slot == 1);
#ifdef USE_ROCM
    return hip_ ? hip_->dev_in[slot] : nullptr;
#else
    return host_in_[slot];
#endif
}

float* PinnedDMAEngine::device_out_ptr(int slot) {
    assert(slot == 0 || slot == 1);
#ifdef USE_ROCM
    return hip_ ? hip_->dev_out[slot] : nullptr;
#else
    return host_out_[slot];
#endif
}

void* PinnedDMAEngine::stream_handle(int slot) {
    assert(slot == 0 || slot == 1);
#ifdef USE_ROCM
    return hip_ ? static_cast<void*>(hip_->stream[slot]) : nullptr;
#else
    (void)slot;
    return nullptr;
#endif
}

void PinnedDMAEngine::upload_chunk(const float* host_src, std::size_t n_floats,
                                  int slot) {
    assert(slot == 0 || slot == 1);
    assert(n_floats <= chunk_floats());
#ifdef USE_ROCM
    hipMemcpyAsync(hip_->dev_in[slot], host_src, n_floats * sizeof(float),
                   hipMemcpyHostToDevice, hip_->stream[slot]);
#else
    std::memcpy(host_in_[slot], host_src, n_floats * sizeof(float));
#endif
}

void PinnedDMAEngine::download_chunk(float* host_dst, std::size_t n_floats,
                                    int slot) {
    assert(slot == 0 || slot == 1);
    assert(n_floats <= chunk_floats());
#ifdef USE_ROCM
    hipMemcpyAsync(host_dst, hip_->dev_out[slot], n_floats * sizeof(float),
                   hipMemcpyDeviceToHost, hip_->stream[slot]);
#else
    std::memcpy(host_dst, host_out_[slot], n_floats * sizeof(float));
#endif
}

void PinnedDMAEngine::sync_slot(int slot) {
    assert(slot == 0 || slot == 1);
#ifdef USE_ROCM
    hipStreamSynchronize(hip_->stream[slot]);
#else
    (void)slot;  // memcpy shim: nothing asynchronous to wait for
#endif
}

void PinnedDMAEngine::sync_all() {
#ifdef USE_ROCM
    hipDeviceSynchronize();
#endif
}
