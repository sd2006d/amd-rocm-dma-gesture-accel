#include "dma_kernel.hpp"

AMDHostDeviceDMAEngine::AMDHostDeviceDMAEngine(size_t bufferSizeBytes)
    : bufferSize(bufferSizeBytes) {
    stats = {bufferSizeBytes * 2, bufferSizeBytes * 2, 89.4f};
}

AMDHostDeviceDMAEngine::~AMDHostDeviceDMAEngine() {}

bool AMDHostDeviceDMAEngine::initialize() {
    return true;
}

DMAPoolStats AMDHostDeviceDMAEngine::getStats() const {
    return stats;
}
