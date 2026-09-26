#ifndef AMD_GESTURE_DMA_ENGINE_HPP
#define AMD_GESTURE_DMA_ENGINE_HPP

// Pinned host memory + double-buffered DMA transfer engine.
//
// Each of the two slots owns an input buffer and an output buffer, so
// a kernel can read chunk i from slot s while the next chunk streams
// in on the other slot:
//
//   slot s: upload(in) -> kernel(in -> out) -> download(out),
//           all queued in order on the slot's stream
//
// ROCm build (compiled with hipcc, USE_ROCM defined):
//   - buffers are device memory, transfers are hipMemcpyAsync on the
//     slot's stream; alternating slots lets transfer of chunk i+1
//     overlap compute of chunk i (see conv2d_hip)
// CPU-only build (no ROCm):
//   - buffers are host memory, transfers are memcpy, sync is a no-op;
//     the slot/chunk bookkeeping is identical so the logic is
//     unit-testable without a GPU
//
// All buffers are allocated once in init() and reused; nothing is
// allocated in the per-chunk hot path.

#include <cstddef>

class PinnedDMAEngine {
public:
    explicit PinnedDMAEngine(std::size_t chunk_bytes);
    ~PinnedDMAEngine();

    PinnedDMAEngine(const PinnedDMAEngine&) = delete;
    PinnedDMAEngine& operator=(const PinnedDMAEngine&) = delete;

    // Allocate buffers and streams. Returns false on failure.
    bool init();
    void shutdown();

    // True only in ROCm builds where HIP initialised successfully.
    bool gpu_backed() const { return gpu_backed_; }

    std::size_t chunk_bytes() const { return chunk_bytes_; }
    std::size_t chunk_floats() const { return chunk_bytes_ / sizeof(float); }

    // Kernel input buffer for a slot (H2D target). Slot must be 0/1.
    float* device_in_ptr(int slot);
    // Kernel output buffer for a slot (D2H source). Slot must be 0/1.
    float* device_out_ptr(int slot);

    // Opaque stream handle for a slot (hipStream_t in ROCm builds,
    // nullptr in CPU-only builds). Kernels for a slot are launched on
    // its stream so compute and transfers stay ordered per slot while
    // the two slots run concurrently.
    void* stream_handle(int slot);

    // Enqueue host->device copy of n_floats into the slot's input
    // buffer. n_floats must be <= chunk_floats().
    void upload_chunk(const float* host_src, std::size_t n_floats, int slot);

    // Enqueue device->host copy of n_floats from the slot's output
    // buffer. n_floats must be <= chunk_floats().
    void download_chunk(float* host_dst, std::size_t n_floats, int slot);

    // Block until all work queued on the slot's stream is complete.
    void sync_slot(int slot);

    // Block until all queued work on every stream is complete.
    void sync_all();

private:
    struct HipState;  // defined in dma_engine.cpp (ROCm builds only)

    std::size_t chunk_bytes_;
    bool gpu_backed_ = false;
    bool initialised_ = false;
    float* host_in_[2] = {nullptr, nullptr};   // CPU shim buffers
    float* host_out_[2] = {nullptr, nullptr};  // CPU shim buffers
    HipState* hip_ = nullptr;                  // ROCm state (pimpl)
};

#endif  // AMD_GESTURE_DMA_ENGINE_HPP
