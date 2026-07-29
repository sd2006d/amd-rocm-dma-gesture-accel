#ifndef DMA_KERNEL_HPP
#define DMA_KERNEL_HPP

#include <cstddef>

struct DMAPoolStats {
    size_t totalAllocatedBytes;
    size_t pinnedHostBytes;
    float peakBandwidthGBs;
};

class AMDHostDeviceDMAEngine {
public:
    AMDHostDeviceDMAEngine(size_t bufferSizeBytes);
    ~AMDHostDeviceDMAEngine();
    bool initialize();
    DMAPoolStats getStats() const;

private:
    size_t bufferSize;
    DMAPoolStats stats;
};

#endif // DMA_KERNEL_HPP
