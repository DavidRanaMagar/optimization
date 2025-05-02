#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;
namespace fs = filesystem;

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
    dim3 blockDim(16, 16);
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

    dim3 poolBlockDim(16, 16);
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

int main() {
    // Verify CUDA is available
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "Error: No CUDA-capable devices found!" << endl;
        return -1;
    }

    // Start total processing timer
    auto total_start = high_resolution_clock::now();

    string inputDir = "./../../data/cats";
    string outputDir = "./../../testcuda";

    // Create output directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        if (!fs::create_directory(outputDir)) {
            cerr << "Error: Could not create output directory!" << endl;
            return -1;
        }
    }

    // Get all image files in the input directory
    vector<string> imageFiles;
    try {
        for (const auto &entry : fs::directory_iterator(inputDir)) {
            if (entry.is_regular_file()) {
                string ext = entry.path().extension().string();
                transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                    imageFiles.push_back(entry.path().string());
                }
            }
        }
    }
    catch (const fs::filesystem_error &e) {
        cerr << "Error accessing directory: " << e.what() << endl;
        return -1;
    }

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
        fs::path inputPath(imagePath);
        string outputPath = outputDir + "/final_" + inputPath.filename().string();

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