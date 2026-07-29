#include "dma_kernel.hpp"
#include <iostream>
#include <cassert>

int main() {
    AMDHostDeviceDMAEngine engine(1024 * 1024 * 4);
    assert(engine.initialize());
    std::cout << "DMA Engine Init Test Passed!" << std::endl;
    return 0;
}
