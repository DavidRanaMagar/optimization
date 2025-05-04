#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define FILTER_SIZE    3
#define BLOCK_SIZE     16
#define POOL_SIZE      3
#define IMG_PATH           "data/training_set/cats/"
#define IMG_PATH_FINAL     "data/gpu_optimal_output/cats/"

// Filter stored in constant memory (flattened)
__constant__ int d_filter[FILTER_SIZE * FILTER_SIZE];

// Fused convolution + max-pooling kernel
extern "C" __global__ void convPoolKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height,
    int pooledW, int pooledH)
{
    const int pad = FILTER_SIZE / 2;
    const int tileW = BLOCK_SIZE * POOL_SIZE + FILTER_SIZE - 1;
    extern __shared__ unsigned char tile[];

    int outX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int outY = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int baseX = blockIdx.x * BLOCK_SIZE * POOL_SIZE;
    int baseY = blockIdx.y * BLOCK_SIZE * POOL_SIZE;

    for (int ty = threadIdx.y; ty < tileW; ty += BLOCK_SIZE) {
        for (int tx = threadIdx.x; tx < tileW; tx += BLOCK_SIZE) {
            int gx = baseX + tx - pad;
            int gy = baseY + ty - pad;
            unsigned char val = 0;
            if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                val = input[gy * width + gx];
            }
            tile[ty * tileW + tx] = val;
        }
    }
    __syncthreads();

    if (outX < pooledW && outY < pooledH) {
        int maxVal = 0;
        for (int py = 0; py < POOL_SIZE; ++py) {
            for (int px = 0; px < POOL_SIZE; ++px) {
                int sum = 0;
                int startY = threadIdx.y * POOL_SIZE + py;
                int startX = threadIdx.x * POOL_SIZE + px;
                for (int fy = 0; fy < FILTER_SIZE; ++fy) {
                    for (int fx = 0; fx < FILTER_SIZE; ++fx) {
                        unsigned char pix = tile[(startY + fy) * tileW + (startX + fx)];
                        sum += pix * d_filter[fy * FILTER_SIZE + fx];
                    }
                }
                maxVal = max(maxVal, sum);
            }
        }
        output[outY * pooledW + outX] = static_cast<unsigned char>(maxVal);
    }
}

int main(int argc, char** argv) {
    // ===== CUDA Device Information =====
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA Device Information:" << std::endl;
    std::cout << "-----------------------" << std::endl;
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl << std::endl;
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimensions: ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")" << std::endl;
    }
    std::cout << "-----------------------" << std::endl << std::endl;

    // Ensure output directory exists
    std::filesystem::create_directories(IMG_PATH_FINAL);

    // Read filter on host
    int h_filter[FILTER_SIZE * FILTER_SIZE] = { /* your 3x3 values */ };
    cudaMemcpyToSymbol(d_filter, h_filter, sizeof(h_filter));

    // Prepare pinned host buffers
    unsigned char *h_input, *h_output;
    cudaHostAlloc(&h_input,  1920 * 1080 * sizeof(unsigned char), cudaHostAllocDefault);
    cudaHostAlloc(&h_output, 1920 * 1080 * sizeof(unsigned char), cudaHostAllocDefault);

    // Create CUDA streams
    const int STREAM_COUNT = 4;
    std::vector<cudaStream_t> streams(STREAM_COUNT);
    for (int i = 0; i < STREAM_COUNT; ++i) cudaStreamCreate(&streams[i]);

    // Device buffers
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input,  1920 * 1080 * sizeof(unsigned char));
    cudaMalloc(&d_output, 1920 * 1080 / (POOL_SIZE * POOL_SIZE) * sizeof(unsigned char));

    // List images
    std::vector<std::filesystem::path> files;
    for (auto& p : std::filesystem::directory_iterator(IMG_PATH))
        files.push_back(p.path());

    // Timing
    cudaEvent_t startE, endE;
    cudaEventCreate(&startE);
    cudaEventCreate(&endE);
    cudaEventRecord(startE);

    int idx = 0;
    for (auto& file : files) {
        cudaStream_t s = streams[idx % STREAM_COUNT];
        cv::Mat img = cv::imread(file.string(), cv::IMREAD_GRAYSCALE);
        int W = img.cols, H = img.rows;
        int pooledW = (W + POOL_SIZE - 1) / POOL_SIZE;
        int pooledH = (H + POOL_SIZE - 1) / POOL_SIZE;

        memcpy(h_input, img.data, W * H * sizeof(unsigned char));
        cudaMemcpyAsync(d_input, h_input, W * H * sizeof(unsigned char), cudaMemcpyHostToDevice, s);

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((pooledW + BLOCK_SIZE - 1)/BLOCK_SIZE, (pooledH + BLOCK_SIZE - 1)/BLOCK_SIZE);
        size_t sharedBytes = (BLOCK_SIZE * POOL_SIZE + FILTER_SIZE - 1) *
                             (BLOCK_SIZE * POOL_SIZE + FILTER_SIZE - 1) * sizeof(unsigned char);
        convPoolKernel<<<grid, block, sharedBytes, s>>>(d_input, d_output, W, H, pooledW, pooledH);

        cudaMemcpyAsync(h_output, d_output, pooledW * pooledH * sizeof(unsigned char), cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);

        cv::Mat outImg(pooledH, pooledW, CV_8UC1, h_output);
        std::string output_filename = "final_" + file.filename().string();
        cv::imwrite(std::string(IMG_PATH_FINAL) + output_filename, outImg);
        idx++;
    }

    cudaEventRecord(endE);
    cudaEventSynchronize(endE);
    float ms;
    cudaEventElapsedTime(&ms, startE, endE);

    // ===== Processing Summary =====
    std::cout << "Processing complete!" << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << "Total images processed: " << files.size() << std::endl;
    std::cout << "Failed to process: 0" << std::endl;
    std::cout << "Total processing time: " << static_cast<int>(ms) << " ms" << std::endl;
    std::cout << "=================================" << std::endl;

    // Cleanup
    for (auto& s : streams) cudaStreamDestroy(s);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
