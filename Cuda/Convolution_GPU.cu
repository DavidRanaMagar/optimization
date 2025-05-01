//NOT COMPLETE

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <getopt.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace chrono;

#define BASE_PATH "./../data/"
#define CATS_PATH "./../data/cats/"
#define CATS_PATH_OUTPUT "./../data/Convolution/cats_output/"
#define DOGS_PATH "./../data/dogs/"
#define DOGS_PATH_OUTPUT "./../data/Convolution/dogs_output/"
#define NUM_FILTERS 6
#define FILTER_SIZE 3

// CUDA kernel for convolution
__global__ void convolutionKernel(
    const unsigned char* input,
    unsigned char* output,
    const int* filter,
    int width,
    int height,
    int filter_size)
{
    // Calculate the row and column index of the output element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the thread is within the valid output region
    if (row < height - filter_size + 1 && col < width - filter_size + 1) {
        int sum = 0;
        
        // Apply the filter
        for (int i = 0; i < filter_size; i++) {
            for (int j = 0; j < filter_size; j++) {
                int image_value = input[(row + i) * width + (col + j)];
                int filter_value = filter[i * filter_size + j];
                sum += image_value * filter_value;
            }
        }
        
        // Clamp the value to valid range [0, 255]
        sum = max(0, min(255, sum));
        
        // Store the result
        output[row * (width - filter_size + 1) + col] = static_cast<unsigned char>(sum);
    }
}

void getCurrDir();
vector<string> getFiles(const string &path);
vector<vector<vector<int>>> createFilters();
bool createDirectory(const string &path);

// CUDA implementation of image convolution
vector<Mat> conv2D_CUDA(
    const string &image_path,
    const vector<vector<vector<int>>> &filters)
{
    // Read the image in grayscale
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    vector<Mat> new_images;
    
    if (!image.empty()) {
        // Original image size
        int image_width = image.cols;
        int image_height = image.rows;
        
        // New image size (after convolution)
        int new_image_width = image_width - FILTER_SIZE + 1;
        int new_image_height = image_height - FILTER_SIZE + 1;
        
        // Allocate device memory for the input image
        unsigned char* d_image;
        cudaMalloc(&d_image, image_width * image_height * sizeof(unsigned char));
        cudaMemcpy(d_image, image.data, image_width * image_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
        
        // Define block and grid dimensions for CUDA kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((new_image_width + blockDim.x - 1) / blockDim.x, 
                     (new_image_height + blockDim.y - 1) / blockDim.y);
        
        // Process each filter
        for (const auto& filter : filters) {
            // Flatten the 2D filter to 1D for easy transfer to GPU
            vector<int> flat_filter;
            for (const auto& row : filter) {
                for (int val : row) {
                    flat_filter.push_back(val);
                }
            }
            
            // Allocate device memory for the filter
            int* d_filter;
            cudaMalloc(&d_filter, flat_filter.size() * sizeof(int));
            cudaMemcpy(d_filter, flat_filter.data(), flat_filter.size() * sizeof(int), cudaMemcpyHostToDevice);
            
            // Allocate device memory for the output
            unsigned char* d_output;
            cudaMalloc(&d_output, new_image_width * new_image_height * sizeof(unsigned char));
            
            // Launch the CUDA kernel for convolution
            convolutionKernel<<<gridDim, blockDim>>>(
                d_image,
                d_output,
                d_filter,
                image_width,
                image_height,
                FILTER_SIZE
            );
            
            // Synchronize to wait for kernel to finish
            cudaDeviceSynchronize();
            
            // Copy the output back to the host
            Mat output(new_image_height, new_image_width, CV_8UC1);
            cudaMemcpy(output.data, d_output, new_image_width * new_image_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
            
            // Add the output to the result vector
            new_images.push_back(output);
            
            // Free device memory for this filter iteration
            cudaFree(d_filter);
            cudaFree(d_output);
        }
        
        // Free device memory for the input image
        cudaFree(d_image);
    }
    
    return new_images;
}

void print_usage(const char *prog_name)
{
    cerr << "Usage: " << prog_name << " [-n NUM_IMAGES]" << endl;
    cerr << "Options:" << endl;
    cerr << "  -n NUM_IMAGES  Number of images to process from each category (default: all)" << endl;
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

    // Check if CUDA is available
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        cerr << "No CUDA devices found. Exiting..." << endl;
        return 1;
    }
    cout << "CUDA device(s) found: " << device_count << endl;
    
    // Print CUDA device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cout << "Using CUDA device: " << deviceProp.name << endl;
    cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << endl;

    // Ensure output directories exist
    if (!createDirectory(CATS_PATH_OUTPUT) || !createDirectory(DOGS_PATH_OUTPUT))
    {
        cerr << "Error: Could not create output directories" << endl;
        return 1;
    }

    // Get all images
    vector<string> cat_images = getFiles(CATS_PATH);
    vector<string> dog_images = getFiles(DOGS_PATH);

    cout << "Found " << cat_images.size() << " cat images" << endl;
    cout << "Found " << dog_images.size() << " dog images" << endl;

    // Limit the number of images if specified
    if (num_images > 0)
    {
        if (static_cast<size_t>(num_images) < cat_images.size())
        {
            cat_images.resize(num_images);
        }
        if (static_cast<size_t>(num_images) < dog_images.size())
        {
            dog_images.resize(num_images);
        }
        cout << "Processing " << cat_images.size() << " cat images and "
             << dog_images.size() << " dog images" << endl;
    }

    // Create filters
    vector<vector<vector<int>>> filters = createFilters();

    // Process cat images
    auto start = high_resolution_clock::now();
    cout << "Processing cat images with CUDA..." << endl;

    for (size_t i = 0; i < cat_images.size(); i++)
    {
        // Use CUDA implementation
        vector<Mat> new_images = conv2D_CUDA(cat_images[i], filters);

        // Extract filename from path
        size_t last_slash = cat_images[i].find_last_of("/\\");
        string filename = (last_slash == string::npos) ? cat_images[i] : cat_images[i].substr(last_slash + 1);

        // Write convolved images to output folder
        int index = 0;
        for (auto image : new_images)
        {
            bool success = imwrite(string(CATS_PATH_OUTPUT) + "filter_" + to_string(index++) + "_" + filename, image);
            cout << "Cat image " << i + 1 << "/" << cat_images.size() << ", filter " << index << ": Success: " << success << endl;
        }
    }

    // Process dog images
    cout << "Processing dog images with CUDA..." << endl;

    for (size_t i = 0; i < dog_images.size(); i++)
    {
        // Use CUDA implementation
        vector<Mat> new_images = conv2D_CUDA(dog_images[i], filters);

        // Extract filename from path
        size_t last_slash = dog_images[i].find_last_of("/\\");
        string filename = (last_slash == string::npos) ? dog_images[i] : dog_images[i].substr(last_slash + 1);

        // Write convolved images to output folder
        int index = 0;
        for (auto image : new_images)
        {
            bool success = imwrite(string(DOGS_PATH_OUTPUT) + "filter_" + to_string(index++) + "_" + filename, image);
            cout << "Dog image " << i + 1 << "/" << dog_images.size() << ", filter " << index << ": Success: " << success << endl;
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Time taken in seconds: " << duration.count() / 1000000.0 << " seconds" << endl;

    // Clean up CUDA resources
    cudaDeviceReset();

    return 0;
}

// Function to create a directory if it doesn't exist
bool createDirectory(const string &path)
{
    struct stat st = {};
    if (stat(path.c_str(), &st) == -1)
    {
        // Directory doesn't exist, create it
        if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
        {
            perror(("Failed to create directory: " + path).c_str());
            return false;
        }
    }
    return true;
}

void getCurrDir()
{
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)))
    {
        std::cout << "Current working directory: " << cwd << std::endl;
    }
    else
    {
        std::cerr << "Failed to get current working directory." << std::endl;
    }
}

vector<string> getFiles(const string &path)
{
    vector<string> files;
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(path.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            if (ent->d_type == DT_REG)
            { // Regular file
                files.push_back(path + ent->d_name);
            }
        }
        closedir(dir);
    }
    else
    {
        perror(("Could not open directory: " + path).c_str());
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