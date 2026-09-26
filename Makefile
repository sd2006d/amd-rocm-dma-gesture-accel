# CNN gesture acceleration: CPU reference + HIP kernels + DMA engine.
#
# Build flavour is auto-detected:
#   - hipcc on PATH  -> ROCm build (compiles src/conv2d_hip.cpp with hipcc,
#                      defines USE_ROCM, links with hipcc)
#   - no hipcc       -> CPU-only build (HIP sources skipped entirely)

CXX      ?= g++
HIPCC    := $(shell command -v hipcc 2>/dev/null)
CXXFLAGS := -O2 -std=c++17 -Iinclude -Wall -Wextra
BUILD    := build

CPU_SRCS := src/conv2d_cpu.cpp src/dma_engine.cpp src/gesture_cnn.cpp
CPU_OBJS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(CPU_SRCS))

TEST_SRCS  := bench/test_correctness.cpp
BENCH_SRCS := bench/bench_throughput.cpp

ifneq ($(HIPCC),)
$(info hipcc found at $(HIPCC): building WITH ROCm support)
CXXFLAGS += -DUSE_ROCM
HIP_OBJ  := $(BUILD)/conv2d_hip.o
LINK     := $(HIPCC)
OBJS     := $(CPU_OBJS) $(HIP_OBJ)
else
$(info hipcc not found: building CPU-only)
LINK     := $(CXX)
OBJS     := $(CPU_OBJS)
endif

TARGETS := $(BUILD)/test_correctness $(BUILD)/bench_throughput

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

ifdef HIPCC
$(HIP_OBJ): src/conv2d_hip.cpp include/conv2d_hip.hpp | $(BUILD)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@
endif

$(BUILD)/test_correctness: $(TEST_SRCS) $(OBJS) | $(BUILD)
	$(LINK) $(CXXFLAGS) $(TEST_SRCS) $(OBJS) -o $@

$(BUILD)/bench_throughput: $(BENCH_SRCS) $(OBJS) | $(BUILD)
	$(LINK) $(CXXFLAGS) $(BENCH_SRCS) $(OBJS) -o $@

test: $(BUILD)/test_correctness
	./$(BUILD)/test_correctness

bench: $(BUILD)/bench_throughput
	./$(BUILD)/bench_throughput

clean:
	rm -rf $(BUILD)

.PHONY: all test bench clean
