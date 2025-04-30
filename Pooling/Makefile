# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -fopenmp -O3 -Wall -Wextra -Wsign-compare
CPPFLAGS := $(shell pkg-config --cflags opencv4)
LDFLAGS := $(shell pkg-config --libs opencv4) -fopenmp

# Target executable
TARGET := Pooling_CPU

# Source files
SRCS := Pooling_CPU.cpp

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
	rm  -rf ../data/Pooling/cats_output ../data/Pooling/dogs_output ../data/Final/cats_final ../data/Final/dogs_final

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
	mkdir -p ../data/Pooling/cats_output
	mkdir -p ../data/Pooling/dogs_output
	mkdir -p ../data/Final/dogs_final
	mkdir -p ../data/Final/cats_final

# Install required dependencies (for Ubuntu)
install-deps:
	sudo apt-get update
	sudo apt-get install -y g++ libomp-dev libopencv-dev pkg-config

# Show help
help:
	@echo "Available targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  clean        - Remove executable and object files"
	@echo "  run          - Run with all images"
	@echo "  run 4        - Run with 4 images (replace 4 with any number)"
	@echo "  test         - Quick test with 2 images"
	@echo "  create-dirs  - Create output directories"
	@echo "  install-deps - Install required dependencies (Ubuntu only)"
	@echo "  help         - Show this help message"

.PHONY: all clean run test create-dirs install-deps help