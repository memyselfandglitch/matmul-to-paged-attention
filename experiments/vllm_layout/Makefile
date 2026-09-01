CXX ?= c++

CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic

ifeq ($(shell uname -m),arm64)
NATIVE_FLAGS ?= -mcpu=native
else
NATIVE_FLAGS ?= -march=native
endif

BUILD_DIR := build
TARGET := $(BUILD_DIR)/kv_layout_study
TRAVERSAL_TARGET := $(BUILD_DIR)/block_traversal_study

.PHONY: all check clean

all: $(TARGET) $(TRAVERSAL_TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(TARGET): kv_layout_study.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(NATIVE_FLAGS) $< -o $@

$(TRAVERSAL_TARGET): block_traversal_study.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(NATIVE_FLAGS) $< -o $@

check: $(TARGET) $(TRAVERSAL_TARGET)
	$(TARGET) 7
	$(TRAVERSAL_TARGET) 7

clean:
	rm -f $(TARGET) $(TRAVERSAL_TARGET)
