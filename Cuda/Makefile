CXX = nvcc
CXXFLAGS = -std=c++14
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`
CUDA_FLAGS = -arch=sm_75

# Update this if your compute capability is different
# Common values:
# - sm_50 for Maxwell
# - sm_60 for Pascal (GTX 1060, 1070, 1080)
# - sm_70 for Volta (Tesla V100)
# - sm_75 for Turing (RTX 2060, 2070, 2080)
# - sm_80 for Ampere (RTX 3060, 3070, 3080)

TARGET = Convolution_GPU

# Source files
SRCS = Convolution_GPU.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(CUDA_FLAGS) -o $@ $^ $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

# If you want to run with a specific number of images
run_limit:
	./$(TARGET) -n 10