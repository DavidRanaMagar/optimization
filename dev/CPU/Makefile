# Compiler and flags
CXX := clang++
CXXFLAGS := -std=c++17 -Xpreprocessor -fopenmp -O3 -Wall -Wextra -Wsign-compare
CPPFLAGS := $(shell pkg-config --cflags opencv4)
# Updated LDFLAGS to include correct OpenMP path
LDFLAGS := $(shell pkg-config --libs opencv4) -L/opt/homebrew/opt/libomp/lib -lomp

# For Intel Macs, use this instead:
# LDFLAGS := $(shell pkg-config --libs opencv4) -L/usr/local/opt/libomp/lib -lomp

# Target executable
TARGET := CNN_CPU

# Source files
SRCS := CNN_CPU.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)
	rm -rf ../../data/dev/cpu

# [Rest of your Makefile remains the same...]

# Run with all images
run:
	./$(TARGET)

# Handle 'make run #' where # is a number
ifneq ($(filter run,$(MAKECMDGOALS)),)
  # Get the number argument (the first non-run argument)
  NUM_ARG := $(firstword $(filter-out run,$(MAKECMDGOALS)))
  ifneq ($(NUM_ARG),)
    # If we got a number, use it
    .PHONY: run$(NUM_ARG)
    run$(NUM_ARG):
		@:
    run: run$(NUM_ARG)
		./$(TARGET) -n $(NUM_ARG)
  endif
endif

# Quick test with 2 images
test:
	./$(TARGET) -n 2

# Create output directories
create-dirs:
	mkdir -p ../../data/dev/cpu/Final
	mkdir -p ../../data/dev/cpu/Conv

# Install required dependencies (for macOS using Homebrew)
install-deps:
	# Install Homebrew if not already installed
	@if ! command -v brew &> /dev/null; then \
		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		echo 'eval "$$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc; \
		source ~/.zshrc; \
	fi

	# Install required packages
	brew update
	brew install llvm libomp opencv pkg-config

	# Set environment variables for llvm
	@echo "You may need to add these to your shell configuration:"
	@echo 'export PATH="/opt/homebrew/opt/llvm/bin:$$PATH"'
	@echo 'export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"'
	@echo 'export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"'

# Show help
help:
	@echo "Available targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  clean        - Remove executable and object files"
	@echo "  run          - Run with all images"
	@echo "  run 4        - Run with 4 images (replace 4 with any number)"
	@echo "  test         - Quick test with 2 images"
	@echo "  create-dirs  - Create output directories"
	@echo "  install-deps - Install required dependencies (macOS only)"
	@echo "  help         - Show this help message"

.PHONY: all clean run test create-dirs install-deps help