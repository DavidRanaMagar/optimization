#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// Cross-platform includes and definitions
#ifdef _WIN32
#include <direct.h>
    #define MKDIR(dir) _mkdir(dir)
    #define PATH_SEPARATOR "\\"
#else
#include <sys/stat.h>
#include <sys/types.h>
#define MKDIR(dir) mkdir(dir, 0755)
#define PATH_SEPARATOR "/"
#endif

// OpenCV includes with minimal dependencies
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// Use C++17 filesystem if available, otherwise fallback
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#include <filesystem>
    namespace fs = std::filesystem;
    #define HAS_FILESYSTEM 1
#else
#define HAS_FILESYSTEM 0
// Simple directory entry for fallback implementation
struct DirectoryEntry {
    std::string path;
    bool is_regular_file() const { return true; } // Simplified
    std::string extension() const {
        size_t pos = path.find_last_of('.');
        return (pos != std::string::npos) ? path.substr(pos) : "";
    }
};
#endif

using namespace cv;
using namespace std;
using namespace std::chrono;

// Configurations
const int NUM_FILTERS = 6;
const int FILTER_SIZE = 3;
const int POOLING_SIZE = 3;

// CUDA error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        cerr << "CUDA Runtime Error at: " << file << ":" << line << endl;
        cerr << cudaGetErrorString(err) << " " << func << endl;
        exit(1);
    }
}

// Store filter in constant memory for faster access
__constant__ int filter_round_line[3][3] = {{0, 1, 0}, {1, 0, 1}, {0, 1, 0}};

// Kernel for convolution operation
__global__ void convolutionKernel(const unsigned char* input, unsigned char* output,
                                  int width, int height, int filterSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width - filterSize + 1 && y < height - filterSize + 1) {
        int sum = 0;
        for (int i = 0; i < filterSize; i++) {
            for (int j = 0; j < filterSize; j++) {
                int image_value = input[(y + i) * width + (x + j)];
                int filter_value = filter_round_line[i][j];
                sum += image_value * filter_value;
            }
        }
        output[y * (width - filterSize + 1) + x] = static_cast<unsigned char>(sum);
    }
}

// Kernel for max pooling operation
__global__ void maxPoolingKernel(const unsigned char* input, unsigned char* output,
                                 int width, int height, int poolingSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pooledWidth = width / poolingSize;
    int pooledHeight = height / poolingSize;

    if (x < pooledWidth && y < pooledHeight) {
        unsigned char maxVal = 0;
        int startX = x * poolingSize;
        int startY = y * poolingSize;

        for (int i = startY; i < startY + poolingSize && i < height; i++) {
            for (int j = startX; j < startX + poolingSize && j < width; j++) {
                maxVal = (input[i * width + j] > maxVal) ? input[i * width + j] : maxVal;
            }
        }
        output[y * pooledWidth + x] = maxVal;
    }
}

// Cross-platform directory creation function
bool createDirectory(const string& path) {
    if (path.empty()) return false;

#if HAS_FILESYSTEM
    try {
            return fs::create_directories(path);
        } catch (const fs::filesystem_error& e) {
            cerr << "Error creating directory: " << e.what() << endl;
            return false;
        }
#else
    return MKDIR(path.c_str()) == 0;
#endif
}

// Cross-platform directory existence check
bool directoryExists(const string& path) {
#if HAS_FILESYSTEM
    return fs::exists(path) && fs::is_directory(path);
#else
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
#endif
}

// Cross-platform function to get file extension in lowercase
string getFileExtension(const string& filename) {
    size_t dotPos = filename.find_last_of('.');
    if (dotPos == string::npos) return "";

    string ext = filename.substr(dotPos);
    for (char& c : ext) c = tolower(c);
    return ext;
}

// Cross-platform function to get filename from path
string getFilenameFromPath(const string& filepath) {
    size_t lastSeparator = filepath.find_last_of("/\\");
    return (lastSeparator == string::npos) ? filepath : filepath.substr(lastSeparator + 1);
}

// Get image files from a directory
vector<string> getImageFiles(const string& directory) {
    vector<string> imageFiles;

#if HAS_FILESYSTEM
    try {
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (fs::is_regular_file(entry.path())) {
                    string ext = entry.path().extension().string();
                    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                        imageFiles.push_back(entry.path().string());
                    }
                }
            }
        } catch (const fs::filesystem_error& e) {
            cerr << "Error accessing directory: " << e.what() << endl;
        }
#else
    // Simplified implementation - would need platform-specific code for actual use
    cerr << "Warning: C++17 filesystem not supported. Please implement a directory listing function for your platform." << endl;
    // Here you would use platform-specific code (e.g., FindFirstFile/FindNextFile on Windows,
    // opendir/readdir on POSIX systems) to list directory contents
#endif

    return imageFiles;
}

Mat applyConvAndPoolCUDA(const Mat &image) {
    if (image.empty()) {
        return Mat();
    }

    // Allocate device memory
    unsigned char *d_input, *d_convolved, *d_pooled;
    int width = image.cols;
    int height = image.rows;

    size_t inputSize = width * height * sizeof(unsigned char);
    size_t convOutputSize = (width - FILTER_SIZE + 1) * (height - FILTER_SIZE + 1) * sizeof(unsigned char);

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_convolved, convOutputSize));

    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, image.data, inputSize, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters for convolution
    dim3 blockDim(32, 32);
    dim3 gridDim((width - FILTER_SIZE + 1 + blockDim.x - 1) / blockDim.x,
                 (height - FILTER_SIZE + 1 + blockDim.y - 1) / blockDim.y);

    // Launch convolution kernel
    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_convolved, width, height, FILTER_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Configure kernel launch parameters for pooling
    int pooledWidth = (width - FILTER_SIZE + 1) / POOLING_SIZE;
    int pooledHeight = (height - FILTER_SIZE + 1) / POOLING_SIZE;
    size_t pooledSize = pooledWidth * pooledHeight * sizeof(unsigned char);

    CHECK_CUDA_ERROR(cudaMalloc(&d_pooled, pooledSize));

    dim3 poolBlockDim(32, 32);
    dim3 poolGridDim((pooledWidth + poolBlockDim.x - 1) / poolBlockDim.x,
                     (pooledHeight + poolBlockDim.y - 1) / poolBlockDim.y);

    // Launch pooling kernel
    maxPoolingKernel<<<poolGridDim, poolBlockDim>>>(d_convolved, d_pooled,
                                                    width - FILTER_SIZE + 1,
                                                    height - FILTER_SIZE + 1,
                                                    POOLING_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result back to host
    Mat pooledResult(pooledHeight, pooledWidth, CV_8UC1);
    CHECK_CUDA_ERROR(cudaMemcpy(pooledResult.data, d_pooled, pooledSize, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_convolved));
    CHECK_CUDA_ERROR(cudaFree(d_pooled));

    return pooledResult;
}

// Print GPU information
void printCudaDeviceInfo() {
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));

    cout << "CUDA Device Information:" << endl;
    cout << "-----------------------" << endl;
    cout << "Number of CUDA devices: " << deviceCount << endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, i));

        cout << "\nDevice " << i << ": " << deviceProp.name << endl;
        cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << endl;
        cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << endl;
        cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << endl;
        cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << endl;
        cout << "  Max threads dimensions: ("
             << deviceProp.maxThreadsDim[0] << ", "
             << deviceProp.maxThreadsDim[1] << ", "
             << deviceProp.maxThreadsDim[2] << ")" << endl;
    }
    cout << "-----------------------" << endl;
}

int main(int argc, char* argv[]) {
    // Verify CUDA is available
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        cerr << "Error: No CUDA-capable devices found or CUDA driver not installed!" << endl;
        cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << endl;
        return -1;
    }

    // Print CUDA device information
    printCudaDeviceInfo();

    // Set default paths based on platform
#ifdef _WIN32
    string defaultInputDir = "..\\..\\data\\images";
        string defaultOutputDir = "..\\..\\data\\gpu_output";
#else
    string defaultInputDir = "../../data/images";
    string defaultOutputDir = "../../data/gpu_output";
#endif

    // Parse command line arguments for input/output directories
    string inputDir = defaultInputDir;
    string outputDir = defaultOutputDir;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            inputDir = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  -i, --input DIR   Input directory containing images (default: " << defaultInputDir << ")" << endl;
            cout << "  -o, --output DIR  Output directory for processed images (default: " << defaultOutputDir << ")" << endl;
            cout << "  -h, --help        Display this help message" << endl;
            return 0;
        }
    }

    cout << "Input directory: " << inputDir << endl;
    cout << "Output directory: " << outputDir << endl;

    // Start total processing timer
    auto total_start = high_resolution_clock::now();

    // Create output directory if it doesn't exist
    if (!directoryExists(outputDir)) {
        if (!createDirectory(outputDir)) {
            cerr << "Error: Could not create output directory '" << outputDir << "'!" << endl;
            return -1;
        }
        cout << "Created output directory: " << outputDir << endl;
    }

    // Get all image files in the input directory
    vector<string> imageFiles = getImageFiles(inputDir);

    if (imageFiles.empty()) {
        cerr << "Error: No image files found in " << inputDir << endl;
        return -1;
    }

    // Initialize counters and timers
    int processed_count = 0;
    int failed_count = 0;
    long long total_processing_time = 0;

    cout << "Starting processing of " << imageFiles.size() << " images..." << endl;

    // Process each image
    for (const auto &imagePath : imageFiles) {
        auto image_start = high_resolution_clock::now();

        // Load image
        Mat image = imread(imagePath, IMREAD_GRAYSCALE);
        if (image.empty()) {
            cerr << "Warning: Could not load image " << imagePath << " - skipping" << endl;
            failed_count++;
            continue;
        }

        // Process image using CUDA
        Mat result = applyConvAndPoolCUDA(image);
        if (result.empty()) {
            cerr << "Warning: Processing failed for " << imagePath << " - skipping" << endl;
            failed_count++;
            continue;
        }

        // Create output filename
        string filename = getFilenameFromPath(imagePath);
        string outputPath = outputDir + PATH_SEPARATOR + "final_" + filename;

        // Save result
        if (!imwrite(outputPath, result)) {
            cerr << "Warning: Failed to save " << outputPath << endl;
            failed_count++;
            continue;
        }

        auto image_end = high_resolution_clock::now();
        auto image_duration = duration_cast<milliseconds>(image_end - image_start);
        total_processing_time += image_duration.count();
        processed_count++;
    }

    auto total_end = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(total_end - total_start);

    // Print summary
    cout << "\nProcessing complete!" << endl;
    cout << "=================================" << endl;
    cout << "Total images processed: " << processed_count << endl;
    cout << "Failed to process: " << failed_count << endl;
    cout << "Total processing time: " << total_duration.count() << " ms" << endl;
    cout << "=================================" << endl;

    return 0;
}