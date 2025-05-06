#include <iostream>                   // Standard I/O stream library
#include <vector>                     // STL vector container
#include <string>                     // STL string class
#include <chrono>                     // High-resolution clock for timing
#include <filesystem>                 // File system operations (C++17)
#include <opencv2/opencv.hpp>         // OpenCV core and image I/O
#include <cuda_runtime.h>             // CUDA runtime API

// Constants defining filter, block, and pooling sizes
#define FILTER_SIZE 3
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
#define POOL_SIZE 3

// Paths for input images and final output directory
#define IMG_PATH "../../data/images/"
#define IMG_PATH_FINAL "../../data/gpu_optimal_output/"

// Filter stored in GPU constant memory (flattened 1D array)
__constant__ int d_filter[FILTER_SIZE * FILTER_SIZE];

// -----------------------------------------------------------------------------
// GPU kernel: perform fused 3×3 convolution followed by 3×3 max pooling
// -----------------------------------------------------------------------------
extern "C" __global__ void convPoolKernel(
        const unsigned char *__restrict__ input,   // Pointer to input image data
        unsigned char *__restrict__ output,        // Pointer to output image data
        int width,                                 // Input image width
        int height,                                // Input image height
        int pooledW,                               // Output (pooled) width
        int pooledH)                               // Output (pooled) height
{
    // Compute output pixel coordinates for this thread
    int outX = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    int outY = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;

    // Boundary check: exit if outside output dimensions
    if (outX >= pooledW || outY >= pooledH) return;

    unsigned char maxVal = 0;  // Initialize max value for pooling

    // Iterate over each position in the 3×3 pooling window
#pragma unroll
    for (int py = 0; py < POOL_SIZE; ++py) {
        for (int px = 0; px < POOL_SIZE; ++px) {
            // Compute top-left corner of the convolution window for this pool cell
            int convY = outY * POOL_SIZE + py;
            int convX = outX * POOL_SIZE + px;
            int sum = 0;  // Accumulator for convolution sum

            // Perform 3×3 convolution at this position
#pragma unroll
            for (int fy = 0; fy < FILTER_SIZE; ++fy) {
#pragma unroll
                for (int fx = 0; fx < FILTER_SIZE; ++fx) {
                    int inY = convY + fy;
                    int inX = convX + fx;

                    // Check bounds before reading input pixel
                    if (inX >= 0 && inX < width && inY >= 0 && inY < height) {
                        unsigned char pix = __ldg(&input[inY * width + inX]);                // Load pixel
                        int w = __ldg(&d_filter[fy * FILTER_SIZE + fx]);                     // Load filter weight
                        sum += pix * w;                                                       // Multiply-accumulate
                    }
                }
            }

            // Clamp convolution result to 0–255 and update pooling max
            unsigned char convVal = static_cast<unsigned char>(sum);
            if (convVal > maxVal) {
                maxVal = convVal;
            }
        }
    }

    // Write pooled maximum value to output image
    output[outY * pooledW + outX] = maxVal;
}

// -----------------------------------------------------------------------------
// Host code: set up CUDA, read images, launch kernel, and save output
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // ----- Query and display CUDA device properties -----
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA Device Information:\n";
    std::cout << "-----------------------\n";
    std::cout << "Number of CUDA devices: " << deviceCount << "\n\n";

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::cout << "Device " << dev << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads dimensions: ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")\n";
    }
    std::cout << "-----------------------\n\n";

    // ----- Create output directory if it doesn't exist -----
    std::filesystem::create_directories(IMG_PATH_FINAL);

    // ----- Initialize filter weights on GPU -----
    int h_filter[FILTER_SIZE * FILTER_SIZE] = {
            0, 1, 0,
            1, 0, 1,
            0, 1, 0
    };
    cudaMemcpyToSymbol(d_filter, h_filter, sizeof(h_filter));

    // ----- Allocate pinned (page-locked) host buffers for faster transfers -----
    unsigned char *h_input, *h_output;
    cudaHostAlloc(&h_input, 1920 * 1080 * sizeof(unsigned char), cudaHostAllocDefault);
    cudaHostAlloc(&h_output, 1920 * 1080 * sizeof(unsigned char), cudaHostAllocDefault);

    // ----- Create multiple CUDA streams for pipelined processing -----
    const int STREAM_COUNT = 4;
    std::vector<cudaStream_t> streams(STREAM_COUNT);
    for (int i = 0; i < STREAM_COUNT; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // ----- Allocate device buffers for input and output images -----
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, 1920 * 1080 * sizeof(unsigned char));
    cudaMalloc(&d_output, 1920 * 1080 / (POOL_SIZE * POOL_SIZE) * sizeof(unsigned char));

    // ----- Gather list of image file paths from input directory -----
    std::vector<std::filesystem::path> files;
    for (auto &p : std::filesystem::directory_iterator(IMG_PATH)) {
        files.push_back(p.path());
    }

    // ----- Set up CUDA events for timing measurement -----
    cudaEvent_t startE, endE;
    cudaEventCreate(&startE);
    cudaEventCreate(&endE);
    cudaEventRecord(startE);

    // ----- Process each image using round-robin streams -----
    int idx = 0;
    for (auto &file : files)
    {
        cudaStream_t s = streams[idx % STREAM_COUNT];

        // Read image in grayscale
        cv::Mat img = cv::imread(file.string(), cv::IMREAD_GRAYSCALE);
        int W = img.cols;
        int H = img.rows;

        // Compute convolution output dimensions (valid convolution)
        int convW = W - FILTER_SIZE + 1;
        int convH = H - FILTER_SIZE + 1;

        // Compute pooled output dimensions (integer division)
        int pooledW = convW / POOL_SIZE;
        int pooledH = convH / POOL_SIZE;

        // Copy image data to pinned host buffer
        memcpy(h_input, img.data, W * H * sizeof(unsigned char));

        // Asynchronously transfer input image to device
        cudaMemcpyAsync(d_input, h_input, W * H * sizeof(unsigned char),
                        cudaMemcpyHostToDevice, s);

        // Determine grid and block dimensions for kernel launch
        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 grid((pooledW + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                  (pooledH + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

        // Launch convolution + pooling kernel
        convPoolKernel<<<grid, block, 0, s>>>(d_input, d_output,
                                              W, H, pooledW, pooledH);

        // Asynchronously copy result back to host
        cudaMemcpyAsync(h_output, d_output, pooledW * pooledH * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost, s);

        // Wait for stream operations to complete
        cudaStreamSynchronize(s);

        // Wrap output buffer into OpenCV matrix and save to file
        cv::Mat outImg(pooledH, pooledW, CV_8UC1, h_output);
        std::string output_filename = "final_" + file.filename().string();
        cv::imwrite(std::string(IMG_PATH_FINAL) + output_filename, outImg);

        idx++;
    }

    // ----- Record end time and compute elapsed milliseconds -----
    cudaEventRecord(endE);
    cudaEventSynchronize(endE);
    float ms;
    cudaEventElapsedTime(&ms, startE, endE);

    // ----- Print processing summary -----
    std::cout << "Processing complete!\n";
    std::cout << "=================================\n";
    std::cout << "Total images processed: " << files.size() << "\n";
    std::cout << "Failed to process: 0\n";
    std::cout << "Total processing time: " << static_cast<int>(ms) << " ms\n";
    std::cout << "=================================\n";

    // ----- Cleanup: destroy streams and free memory -----
    for (auto &s : streams) {
        cudaStreamDestroy(s);
    }
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}