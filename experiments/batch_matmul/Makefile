CXX ?= c++

CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic

ifeq ($(shell uname -m),arm64)
NATIVE_FLAGS ?= -mcpu=native
else
NATIVE_FLAGS ?= -march=native
endif

BUILD_DIR := build
TARGET := $(BUILD_DIR)/bmm_kv_study

.PHONY: all check clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(TARGET): bmm_kv_study.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(NATIVE_FLAGS) $< -o $@

check: $(TARGET)
	$(TARGET) 9

clean:
	rm -f $(TARGET)
