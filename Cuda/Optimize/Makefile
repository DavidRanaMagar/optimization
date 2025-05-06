# Makefile for cross-platform CUDA CNN (simplified)

# Platform detection
ifeq ($(OS),Windows_NT)
    PLATFORM = WINDOWS
    SHELL = cmd
    RM = del /Q /F
    RM_DIR = rmdir /S /Q
    EXECUTABLE = CNN_Optimize_GPU.exe
    PATHSEP = \\
    CUDA_PATH ?= C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
else
    PLATFORM = LINUX
    RM = rm -f
    RM_DIR = rm -rf
    EXECUTABLE = CNN_Optimize_GPU
    PATHSEP = /
    CUDA_PATH ?= /usr/local/cuda
endif

# Windows-specific settings
ifeq ($(PLATFORM),WINDOWS)
    # OpenCV
    OPENCV_DIR ?= C:$(PATHSEP)opencv$(PATHSEP)build
    OPENCV_INCLUDE = -I"$(OPENCV_DIR)$(PATHSEP)include"
    OPENCV_LIBS = -lopencv_world460

    # CUDA
    NVCC = "$(CUDA_PATH)$(PATHSEP)bin$(PATHSEP)nvcc.exe"
    CUDA_INCLUDE = -I"$(CUDA_PATH)$(PATHSEP)include"
    CUDA_LIB = -L"$(CUDA_PATH)$(PATHSEP)lib$(PATHSEP)x64"
    NVCC_FLAGS = -g -G --std=c++17 -Xcompiler "/EHsc /W3 /nologo /O2 /Zi" -ccbin "$(shell where cl 2>NUL | head -n 1)"
else
    # Linux-specific settings
    OPENCV_INCLUDE = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
    OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)
    ifeq ($(OPENCV_INCLUDE),)
        OPENCV_INCLUDE = -I/usr/include/opencv4
        OPENCV_LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc
    endif

    NVCC = $(CUDA_PATH)/bin/nvcc
    CUDA_INCLUDE = -I$(CUDA_PATH)/include
    CUDA_LIB = -L$(CUDA_PATH)/lib64
    NVCC_FLAGS = -g -G --std=c++17 -Xcompiler "-Wall -Wextra -O2" -arch=sm_75 --expt-relaxed-constexpr
endif

# Common settings
SRC = CNN_Optimize_GPU.cu
OBJ = $(SRC:.cu=.o)
CUDA_LIBS = -lcudart
INCLUDES = -I. $(CUDA_INCLUDE) $(OPENCV_INCLUDE)
OUTPUT_DIR = ..$(PATHSEP)..$(PATHSEP)data$(PATHSEP)gpu_optimal_output

# Targets
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(CUDA_LIB) $(CUDA_LIBS) $(OPENCV_LIBS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

run: $(EXECUTABLE)
	@echo "Running $(EXECUTABLE)..."
	@./$(EXECUTABLE)

clean:
	@echo "Cleaning build artifacts..."
	$(RM) $(OBJ) $(EXECUTABLE)
	@echo "Cleaning output directory..."
ifeq ($(PLATFORM),WINDOWS)
	@if exist "$(subst /,\,$(OUTPUT_DIR))" $(RM_DIR) "$(subst /,\,$(OUTPUT_DIR))"
else
	@$(RM_DIR) $(OUTPUT_DIR)
endif
	@echo "Clean complete."

config:
	@echo "Platform: $(PLATFORM)"
	@echo "NVCC: $(NVCC)"
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "OpenCV Includes: $(OPENCV_INCLUDE)"
	@echo "OpenCV Libraries: $(OPENCV_LIBS)"

.PHONY: all clean run config