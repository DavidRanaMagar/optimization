CCXX = g++
NVCC = nvcc
CFLAGS = -std=c++17 -O3 -fopenmp
NVCCFLAGS = -std=c++17 -O3 --compiler-options -fopenmp

# Use pkg-config to get OpenCV flags
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

# The executable
TARGET = CNN_GPU

# Source files
SRCS = CNN_GPU.cu

# Object files
OBJS = $(SRCS:.cu=.o)

# Default target
all: $(TARGET)

# Compile source files to object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(TARGET) $(OPENCV_LIBS)

clean:
	rm -f $(OBJS) $(TARGET)
	rm  -rf ../../data/dev/gpu

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

# Create output directories
create-dirs:
	mkdir -p ../../data/dev/gpu/Final 
	mkdir -p ../../data/dev/gpu/Conv

.PHONY: all clean run

