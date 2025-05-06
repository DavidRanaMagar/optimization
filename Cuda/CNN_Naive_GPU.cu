/*
 * Naive GPU implementation of a simple convolutional neural network on CUDA:
 *  - Applies a fixed 3x3 round filter to grayscale images.
 *  - Performs 3x3 max-pooling on the convolved output.
 *  - Processes all images in the input directory defined by IMG_PATH and writes results to GPU_OUTPUT_PATH.
 *  - Measures and reports total processing time.
 *
 * Usage: No arguments needed. Simply compile and run.
 *
 * Paths:
 *   - Input directory:  IMG_PATH (e.g., ".\\data\\training_set\\cats\\")
 *   - Output directory: GPU_OUTPUT_PATH (e.g., ".\\data\\gpu_naive_output")
 *
 * Dependencies:
 *   - C++17 <filesystem> for directory operations
 *   - CUDA toolkit for GPU kernels
 *   - OpenCV for image I/O (grayscale)
 */

 #include <iostream>
 #include <vector>
 #include <chrono>
 #include <filesystem>
 #include <cuda_runtime.h>
 #include <opencv2/opencv.hpp>
 
 using namespace std;
 using namespace chrono;
 namespace fs = filesystem;
 
 // Input/output paths
 #define IMG_PATH           "data/training_set/1000_images/"
 #define GPU_OUTPUT_PATH    "data/gpu_naive_output/"
 
 // Filter and pooling dimensions
 constexpr int FILTER_SIZE = 3;    // 3x3 convolution filter
 constexpr int POOL_SIZE   = 3;    // 3x3 pooling window
 enum { BLK_X = 32, BLK_Y = 32 };  // CUDA block dimensions
 
 // Macro to check CUDA API calls for errors
 #define CHECK_CUDA(call) \
     do { \
         cudaError_t err = call; \
         if (err != cudaSuccess) { \
             cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << endl; \
             exit(EXIT_FAILURE); \
         } \
     } while (0)
 
 // 3x3 filter stored in constant memory for fast access
 __constant__ int d_filter[FILTER_SIZE][FILTER_SIZE] = {
     {0, 1, 0},
     {1, 0, 1},
     {0, 1, 0}
 };
 
 // Convolution kernel: apply 3x3 filter to each pixel
 __global__ void convK(const unsigned char* in, unsigned char* out, int w, int h) {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     int outW = w - FILTER_SIZE + 1;
     if (x < outW && y < h - FILTER_SIZE + 1) {
         int sum = 0;
         for (int i = 0; i < FILTER_SIZE; ++i) {
             for (int j = 0; j < FILTER_SIZE; ++j) {
                 sum += in[(y + i) * w + (x + j)] * d_filter[i][j];
             }
         }
         out[y * outW + x] = static_cast<unsigned char>(sum);
     }
 }
 
 // Pooling kernel: perform max-pooling with a 3x3 window
 __global__ void poolK(const unsigned char* in, unsigned char* out, int w, int h) {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     int outW = w / POOL_SIZE;
     if (x < outW && y < h / POOL_SIZE) {
         int mval = 0;
         for (int i = 0; i < POOL_SIZE; ++i) {
             for (int j = 0; j < POOL_SIZE; ++j) {
                 int v = in[(y * POOL_SIZE + i) * w + (x * POOL_SIZE + j)];
                 mval = max(mval, v);
             }
         }
         out[y * outW + x] = static_cast<unsigned char>(mval);
     }
 }
 
 // Process a single image: allocate GPU buffers, copy data, execute kernels, retrieve result
 cv::Mat processImage(const cv::Mat& img) {
     int w = img.cols, h = img.rows;
     int convW = w - FILTER_SIZE + 1, convH = h - FILTER_SIZE + 1;
     int poolW = convW / POOL_SIZE, poolH = convH / POOL_SIZE;
     size_t sizeIn = w * h;
     size_t sizeConv = convW * convH;
     size_t sizePool = poolW * poolH;
     
     unsigned char *d_in, *d_conv, *d_pool;
     CHECK_CUDA(cudaMalloc(&d_in,   sizeIn));
     CHECK_CUDA(cudaMalloc(&d_conv, sizeConv));
     CHECK_CUDA(cudaMalloc(&d_pool, sizePool));
 
     // Copy input image to GPU memory
     CHECK_CUDA(cudaMemcpy(d_in, img.data, sizeIn, cudaMemcpyHostToDevice));
 
     // Launch convolution kernel
     dim3 block(BLK_X, BLK_Y);
     dim3 gridConv((convW + BLK_X - 1) / BLK_X, (convH + BLK_Y - 1) / BLK_Y);
     convK<<<gridConv, block>>>(d_in, d_conv, w, h);
     CHECK_CUDA(cudaDeviceSynchronize());
 
     // Launch pooling kernel
     dim3 gridPool((poolW + BLK_X - 1) / BLK_X, (poolH + BLK_Y - 1) / BLK_Y);
     poolK<<<gridPool, block>>>(d_conv, d_pool, convW, convH);
     CHECK_CUDA(cudaDeviceSynchronize());
 
     // Retrieve result to CPU and free GPU memory
     cv::Mat output(poolH, poolW, CV_8UC1);
     CHECK_CUDA(cudaMemcpy(output.data, d_pool, sizePool, cudaMemcpyDeviceToHost));
     cudaFree(d_in);
     cudaFree(d_conv);
     cudaFree(d_pool);
     return output;
 }
 
 // Print available CUDA device information in specified format
 void printDeviceInfo() {
     int deviceCount;
     CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
     cout << "CUDA Device Information:" << endl;
     cout << "-----------------------" << endl;
     cout << "Number of CUDA devices: " << deviceCount << endl << endl;
     cudaDeviceProp prop;
     for (int i = 0; i < deviceCount; ++i) {
         CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
         cout << "Device " << i << ": " << prop.name << endl;
         cout << "  Compute capability: " << prop.major << "." << prop.minor << endl;
         cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
         cout << "  Multiprocessors: " << prop.multiProcessorCount << endl;
         cout << "  Max threads per block: " << prop.maxThreadsPerBlock << endl;
         cout << "  Max threads dimensions: ("
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << ")" << endl;
     }
     cout << "-----------------------" << endl;
 }
 
 int main() {
     // Ensure the output directory exists
     fs::create_directories(GPU_OUTPUT_PATH);
 
     // Display GPU information
     printDeviceInfo();
 
     // Collect image file paths from input directory
     vector<fs::path> imageFiles;
     for (const auto& entry : fs::directory_iterator(IMG_PATH)) {
         if (entry.is_regular_file()) {
             imageFiles.push_back(entry.path());
         }
     }
 
     // Print input/output directories and start message
     string inputDir = string(".\\") + IMG_PATH;
     string outputDir = string(".\\") + string(GPU_OUTPUT_PATH).substr(0, string(GPU_OUTPUT_PATH).length()-1);
     cout << "Input directory: " << inputDir << endl;
     cout << "Output directory: " << outputDir << endl;
     cout << "Starting processing of " << imageFiles.size() << " images..." << endl << endl;
 
     int successCount = 0, failCount = 0;
     auto startTime = high_resolution_clock::now();
     // Process each image
     for (const auto& path : imageFiles) {
         cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
         if (img.empty()) {
             cerr << "Skipping unreadable file: " << path << endl;
             ++failCount;
             continue;
         }
         cv::Mat result = processImage(img);
         cv::imwrite(string(GPU_OUTPUT_PATH) + "final_" + path.filename().string(), result);
         ++successCount;
     }
     auto endTime = high_resolution_clock::now();
     long long elapsed = duration_cast<milliseconds>(endTime - startTime).count();
 
     // Print summary
     cout << "Processing complete!" << endl;
     cout << "=================================" << endl;
     cout << "=================================" << endl;
     cout << "Total images processed: " << successCount << endl;
     cout << "Failed to process: " << failCount << endl;
     cout << "Total processing time: " << elapsed << " ms" << endl;
     cout << "=================================" << endl;
 
     return 0;
 }
 