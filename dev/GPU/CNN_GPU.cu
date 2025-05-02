#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <chrono>
#include <cstdio>
#include <unistd.h>     // For getcwd() on Linux
#include <filesystem>   // C++17 filesystem
#include <getopt.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;
using namespace chrono;

#define IMG_PATH "./../../data/images/"
#define IMG_CONV_OUTPUT "./../../data/dev/gpu/Conv/"
#define IMG_PATH_FINAL "./../../data/dev/gpu/Final/"
#define NUM_FILTERS 6
#define FILTER_SIZE 3
#define POOLING_SIZE 3

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Forward declarations
void getCurrDir();
vector<filesystem::path> getFiles(const string &path);
vector<vector<vector<int>>> createFilters();
bool createDirectory(const string &path);

// CUDA kernel for convolution
__global__ void convKernel(
        const unsigned char* image,
        unsigned char* output,
        const int* filter,
        int image_width,
        int image_height,
        int new_width,
        int new_height,
        int filter_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int filter_idx = blockIdx.z;

    // Make sure we're within bounds
    if (row < new_height && col < new_width) {
        int sum = 0;

        // Apply filter at this position
        for (int i = 0; i < filter_size; i++) {
            for (int j = 0; j < filter_size; j++) {
                int image_value = image[(row + i) * image_width + (col + j)];
                int filter_value = filter[filter_idx * filter_size * filter_size + i * filter_size + j];
                sum += image_value * filter_value;
            }
        }

        // Write output
        output[filter_idx * new_width * new_height + row * new_width + col] = sum;
    }
}

// CUDA kernel for max pooling
__global__ void maxPoolKernel(
        const unsigned char* image,
        unsigned char* output,
        int image_width,
        int image_height,
        int new_width,
        int new_height,
        int pooling_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Make sure we're within bounds
    if (row < new_height && col < new_width) {
        unsigned char maximum = 0;

        int start_row = row * pooling_size;
        int start_col = col * pooling_size;

        // Find maximum in this pool region
        for (int i = 0; i < pooling_size; i++) {
            for (int j = 0; j < pooling_size; j++) {
                int r = start_row + i;
                int c = start_col + j;

                if (r < image_height && c < image_width) {
                    unsigned char pixel = image[r * image_width + c];
                    if (pixel > maximum) {
                        maximum = pixel;
                    }
                }
            }
        }

        // Write output
        output[row * new_width + col] = maximum;
    }
}

// Function to perform convolution using CUDA
vector<Mat> conv2D_cuda(
        const string& image_path,
        const vector<vector<vector<int>>>& filters,
float& memcpy_time,
float& kernel_time)
{
// Read image
Mat image = imread(image_path, IMREAD_GRAYSCALE);
vector<Mat> result_images;

if (image.empty()) {
return result_images;
}

// Start timing
cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));
float elapsed_memcpy = 0;
float elapsed_kernel = 0;

// Image dimensions
int image_width = image.cols;
int image_height = image.rows;
int new_width = image_width - FILTER_SIZE + 1;
int new_height = image_height - FILTER_SIZE + 1;

// Allocate host memory for filters
int* h_filters = new int[NUM_FILTERS * FILTER_SIZE * FILTER_SIZE];
for (int f = 0; f < NUM_FILTERS; f++) {
for (int i = 0; i < FILTER_SIZE; i++) {
for (int j = 0; j < FILTER_SIZE; j++) {
h_filters[f * FILTER_SIZE * FILTER_SIZE + i * FILTER_SIZE + j] = filters[f][i][j];
}
}
}

// Allocate device memory
unsigned char* d_image;
unsigned char* d_output;
int* d_filters;

CUDA_CHECK(cudaEventRecord(start));
CUDA_CHECK(cudaMalloc(&d_image, image_width * image_height));
CUDA_CHECK(cudaMalloc(&d_output, NUM_FILTERS * new_width * new_height));
CUDA_CHECK(cudaMalloc(&d_filters, NUM_FILTERS * FILTER_SIZE * FILTER_SIZE * sizeof(int)));

// Copy data to device
CUDA_CHECK(cudaMemcpy(d_image, image.data, image_width * image_height, cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_filters, h_filters, NUM_FILTERS * FILTER_SIZE * FILTER_SIZE * sizeof(int), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaEventRecord(stop));
CUDA_CHECK(cudaEventSynchronize(stop));
CUDA_CHECK(cudaEventElapsedTime(&elapsed_memcpy, start, stop));
memcpy_time += elapsed_memcpy;

// Define block and grid dimensions
dim3 blockDim(16, 16);
dim3 gridDim((new_width + blockDim.x - 1) / blockDim.x,
             (new_height + blockDim.y - 1) / blockDim.y,
             NUM_FILTERS);

// Launch kernel
CUDA_CHECK(cudaEventRecord(start));
convKernel<<<gridDim, blockDim>>>(d_image, d_output, d_filters, image_width, image_height, new_width, new_height, FILTER_SIZE);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaEventRecord(stop));
CUDA_CHECK(cudaEventSynchronize(stop));
CUDA_CHECK(cudaEventElapsedTime(&elapsed_kernel, start, stop));
kernel_time += elapsed_kernel;

// Allocate host memory for results
unsigned char* h_output = new unsigned char[NUM_FILTERS * new_width * new_height];

// Copy results back to host
CUDA_CHECK(cudaEventRecord(start));
CUDA_CHECK(cudaMemcpy(h_output, d_output, NUM_FILTERS * new_width * new_height, cudaMemcpyDeviceToHost));
CUDA_CHECK(cudaEventRecord(stop));
CUDA_CHECK(cudaEventSynchronize(stop));
float elapsed_temp;
CUDA_CHECK(cudaEventElapsedTime(&elapsed_temp, start, stop));
memcpy_time += elapsed_temp;

// Convert back to OpenCV Mat format
for (int f = 0; f < NUM_FILTERS; f++) {
Mat new_image(new_height, new_width, CV_8UC1);
for (int i = 0; i < new_height; i++) {
for (int j = 0; j < new_width; j++) {
new_image.at<uchar>(i, j) = h_output[f * new_width * new_height + i * new_width + j];
}
}
result_images.push_back(new_image);
}

// Free memory
delete[] h_filters;
delete[] h_output;
CUDA_CHECK(cudaFree(d_image));
CUDA_CHECK(cudaFree(d_output));
CUDA_CHECK(cudaFree(d_filters));

return result_images;
}

// Function to perform max pooling using CUDA
Mat pool2D_max_cuda(const string& image_path, float& memcpy_time, float& kernel_time) {
    // Read image
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    Mat result_image;

    if (image.empty()) {
        return result_image;
    }

    // Start timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed_memcpy = 0;
    float elapsed_kernel = 0;

    // Image dimensions
    int image_width = image.cols;
    int image_height = image.rows;
    int new_width = image_width / POOLING_SIZE;
    int new_height = image_height / POOLING_SIZE;

    // Allocate device memory
    unsigned char* d_image;
    unsigned char* d_output;

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMalloc(&d_image, image_width * image_height));
    CUDA_CHECK(cudaMalloc(&d_output, new_width * new_height));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_image, image.data, image_width * image_height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_memcpy, start, stop));
    memcpy_time += elapsed_memcpy;

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((new_width + blockDim.x - 1) / blockDim.x,
                 (new_height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    CUDA_CHECK(cudaEventRecord(start));
    maxPoolKernel<<<gridDim, blockDim>>>(d_image, d_output, image_width, image_height, new_width, new_height, POOLING_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_kernel, start, stop));
    kernel_time += elapsed_kernel;

    // Allocate host memory for results
    unsigned char* h_output = new unsigned char[new_width * new_height];

    // Copy results back to host
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, new_width * new_height, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_temp;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_temp, start, stop));
    memcpy_time += elapsed_temp;

    // Convert back to OpenCV Mat format
    result_image = Mat(new_height, new_width, CV_8UC1);
    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            result_image.at<uchar>(i, j) = h_output[i * new_width + j];
        }
    }

    // Free memory
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_output));

    return result_image;
}

void print_usage(const char *prog_name)
{
    cerr << "Usage: " << prog_name << " [-n NUM_IMAGES]" << endl;
    cerr << "Options:" << endl;
    cerr << "  -n NUM_IMAGES  Number of images to process from each category (default: all)" << endl;
}

// Function to create directory if it doesn't exist
bool createDirectory(const string &path)
{
    try
    {
        filesystem::create_directories(path);
        return true;
    }
    catch (const filesystem::filesystem_error &e)
    {
        cerr << "Error creating directory: " << e.what() << endl;
        return false;
    }
}

int main(int argc, char *argv[])
{
    // Parse command line arguments
    int num_images = -1; // Default: process all images
    int opt;

    while ((opt = getopt(argc, argv, "n:")) != -1)
    {
        switch (opt)
        {
            case 'n':
                num_images = atoi(optarg);
                if (num_images <= 0)
                {
                    cerr << "Error: Number of images must be positive" << endl;
                    print_usage(argv[0]);
                    return 1;
                }
                break;
            default: /* '?' */
                print_usage(argv[0]);
                return 1;
        }
    }

    // Print current working directory
    getCurrDir();

    // Ensure output directories exist
    if (!createDirectory(IMG_CONV_OUTPUT) ||
        !createDirectory(IMG_PATH_FINAL))
    {
        cerr << "Error: Could not create output directories" << endl;
        return 1;
    }

    // Get all images
    vector<filesystem::path> images = getFiles(IMG_PATH);

    cout << "Found " << images.size() << " images" << endl;

    // Limit the number of images if specified
    if (num_images > 0)
    {
        if (static_cast<size_t>(num_images) < images.size())
        {
            images.resize(num_images);
        }
        cout << "Processing " << images.size() << " images and " << endl;
    }

    // Create filters
    vector<vector<vector<int>>> filters = createFilters();

    // Store the output filenames to track which ones to pool later
    vector<string> processed_filenames;

    // Timing variables
    float total_memcpy_time = 0.0f;
    float total_kernel_time = 0.0f;

    // Start overall timing
    auto start = high_resolution_clock::now();

    // Process images
    cout << "Processing images with CUDA..." << endl;
    for (size_t i = 0; i < images.size(); i++) {
        // Extract filename from path
        string filename = images[i].filename().string();

        // Perform convolution
        vector<Mat> new_images = conv2D_cuda(images[i].string(), filters, total_memcpy_time, total_kernel_time);

        // Write convolved images to output folder
        int index = 0;
        for (auto& image : new_images) {
            string output_filename = "filter_" + to_string(index++) + "_" + filename;
            string output_path = string(IMG_CONV_OUTPUT) + output_filename;
            bool success = imwrite(output_path, image);

            // Store the output filename for later pooling
            if (success) {
                processed_filenames.push_back(output_filename);
            }
        }
    }


    // Pooling for images
    cout << "Pooling images with CUDA..." << endl;
    for (size_t i = 0; i < processed_filenames.size(); i++) {
        string input_path = string(IMG_CONV_OUTPUT) + processed_filenames[i];
        Mat new_image = pool2D_max_cuda(input_path, total_memcpy_time, total_kernel_time);

        // Write final images to output folder
        imwrite(string(IMG_PATH_FINAL) + processed_filenames[i], new_image);
    }

    // End timing
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    // Output timing information
    float total_time = duration.count() / 1000000.0;
    cout << "Total time: " << total_time << ", memcopy: " << total_memcpy_time / 1000.0
         << ", kernel: " << total_kernel_time / 1000.0 << endl;

    return 0;
}

void getCurrDir()
{
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr)
    {
        std::cout << "Current working directory: " << cwd << std::endl;
    }
    else
    {
        std::cerr << "Failed to get current working directory." << std::endl;
    }
}

vector<filesystem::path> getFiles(const string &path)
{
    vector<filesystem::path> files;
    for (const auto &entry : filesystem::directory_iterator(path))
    {
        files.push_back(entry.path());
    }

    return files;
}

vector<vector<vector<int>>> createFilters()
{
    vector<vector<int>> filter_vertical_line{
            {0, 1, 0},
            {0, 1, 0},
            {0, 1, 0},
    };

    vector<vector<int>> filter_horiz_line{
            {0, 0, 0},
            {1, 1, 1},
            {0, 0, 0},
    };

    vector<vector<int>> filter_diagonal_lbru_line{
            {0, 0, 1},
            {0, 1, 0},
            {1, 0, 0},
    };

    vector<vector<int>> filter_diagonal_lurb_line{
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
    };

    vector<vector<int>> filter_diagonal_x_line{
            {1, 0, 1},
            {0, 1, 0},
            {1, 0, 1},
    };

    vector<vector<int>> filter_round_line{
            {0, 1, 0},
            {1, 0, 1},
            {0, 1, 0},
    };

    vector<vector<vector<int>>> filters{
            filter_vertical_line,
            filter_horiz_line,
            filter_diagonal_lbru_line,
            filter_diagonal_lurb_line,
            filter_diagonal_x_line,
            filter_round_line};

    return filters;
}