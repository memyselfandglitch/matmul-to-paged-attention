CXX ?= c++

CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic

ifeq ($(shell uname -m),arm64)
NATIVE_FLAGS ?= -mcpu=native
else
NATIVE_FLAGS ?= -march=native
endif

BUILD_DIR := build
STUDY := $(BUILD_DIR)/loop_order_study
STAGES := $(BUILD_DIR)/matmul_optimization_stages

.PHONY: all check clean

all: $(STUDY) $(STAGES)

$(BUILD_DIR):
	mkdir -p $@

$(STUDY): loop_order_study.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(NATIVE_FLAGS) $< -o $@

$(STAGES): matmul_optimization_stages.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(NATIVE_FLAGS) $< -o $@

check: all
	$(STUDY) 384 1 384
	$(STAGES) 48 1 48

clean:
	rm -f $(STUDY) $(STAGES)
