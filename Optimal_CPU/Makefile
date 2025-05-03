# Cross-platform Makefile for CNN_CPU

# Detect operating system
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    EXE_EXT := .exe
    RM_CMD := del /Q
    RM_DIR_CMD := rmdir /S /Q
    MKDIR_CMD := mkdir
    
    # Check for MinGW or MSYS2
    ifneq (,$(findstring MINGW,$(shell uname -s)))
        CC := g++
        RM_CMD := rm -f
        RM_DIR_CMD := rm -rf
        MKDIR_CMD := mkdir -p
    else
        CC := g++
    endif
else
    DETECTED_OS := $(shell uname -s)
    EXE_EXT :=
    RM_CMD := rm -f
    RM_DIR_CMD := rm -rf
    MKDIR_CMD := mkdir -p
    CC := g++
endif

# Output binary name
TARGET := CNN_CPU$(EXE_EXT)

# Source files
SOURCES := CNN_CPU.cpp

# OpenCV configuration
ifeq ($(DETECTED_OS),Windows)
    OPENCV_DIR := C:/opencv/build
    OPENCV_INCLUDE := -I$(OPENCV_DIR)/include
    OPENCV_LIBS := -L$(OPENCV_DIR)/x64/mingw/lib \
                   -lopencv_core455 -lopencv_imgproc455 -lopencv_imgcodecs455 -lopencv_highgui455
else
    OPENCV_INCLUDE := $(shell pkg-config --cflags opencv4 2>/dev/null)
    OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null)
    
    ifeq ($(OPENCV_INCLUDE),)
        OPENCV_INCLUDE := $(shell pkg-config --cflags opencv 2>/dev/null)
        OPENCV_LIBS := $(shell pkg-config --libs opencv 2>/dev/null)
    endif
    
    ifeq ($(OPENCV_INCLUDE),)
        OPENCV_INCLUDE := -I/usr/include/opencv4
        OPENCV_LIBS := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
    endif
endif

# Compiler flags
CXXFLAGS := -std=c++11 -Wall -Wextra -O2 $(OPENCV_INCLUDE)
LDFLAGS := $(OPENCV_LIBS)

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

# Output directory to clean
# Output directory to clean - platform specific
ifeq ($(DETECTED_OS),Windows)
    OUTPUT_DIR := ..\data\cpu_output
    # For Windows rmdir command, we need to add quotes for paths with spaces
    CLEAN_OUTPUT_CMD := if exist "$(OUTPUT_DIR)" $(RM_DIR_CMD) "$(OUTPUT_DIR)"
else
    OUTPUT_DIR := ../data/cpu_output
    CLEAN_OUTPUT_CMD := $(RM_DIR_CMD) $(OUTPUT_DIR)
endif


# Default target
all: $(TARGET)

# Pattern rule for object files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CC) $(CXXFLAGS) -c $< -o $@

# Linking
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CC) $^ -o $@ $(LDFLAGS)
	@echo "Build successful!"

# Run target
run: $(TARGET)
	@echo "Running $(TARGET)..."
	@./$(TARGET)

clean:
	@echo "Cleaning build files..."
	$(RM_CMD) $(OBJECTS) $(TARGET)
	@echo "Cleaning output directory..."
	$(RM_DIR_CMD) $(OUTPUT_DIR)
	@echo "Clean complete."

config:
	@echo "Detected OS: $(DETECTED_OS)"
	@echo "Compiler: $(CC)"
	@echo "OpenCV Include: $(OPENCV_INCLUDE)"
	@echo "OpenCV Libraries: $(OPENCV_LIBS)"

help:
	@echo "Available targets:"
	@echo "  all      - Build the application (default)"
	@echo "  run      - Build and run the application"
	@echo "  clean    - Remove build files and output directory"
	@echo "  config   - Show build configuration"
	@echo "  help     - Show this help message"

.PHONY: all clean config help run