/*
 * CPU implementation of simple convolutional neural network operations:
 *  - Applies six 3x3 filters (edge detectors and pattern detectors) to grayscale images.
 *  - Performs max-pooling with a 3x3 window on the final convolved feature map.
 *  - Processes all images found in the input directory and saves the results to an output directory.
 *  - Measures and reports total processing time.
 *
 * Usage:
 *   ./program [-n NUM_IMAGES]
 *   -n NUM_IMAGES : Limit the number of images to process (default: all images).
 *
 * Input Path:
 *   Defined by IMG_PATH (e.g., "data/images/")
 * Output Path:
 *   Defined by IMG_PATH_FINAL (e.g., "data/cpu_output/")
 *
 * Dependencies:
 *   - C++17 <filesystem> for directory operations
 *   - OpenCV for image loading and saving
 *   - Cross-platform support for getcwd and getopt/_getcwd
 */

#include <iostream>  // Basic I/O operations
#include <string>    // std::string class
#include <vector>    // std::vector container
#include <chrono>    // Timing utilities
#include <filesystem> // C++17 filesystem for directory operations
#include <cstring>   // strcmp for command-line parsing
#include <cstdlib>   // atoi for string to integer conversion

#include <opencv2/opencv.hpp>  // OpenCV includes for image processing

#ifdef _WIN32
#include <direct.h>  // For _getcwd on Windows
  #define GETCWD _getcwd
#else
#include <getopt.h>  // getopt for Unix/Linux argument parsing
#include <unistd.h>  // For getcwd on Linux/Unix
#define GETCWD getcwd
#endif

using namespace std;
using namespace cv;
using namespace chrono;
namespace fs = std::filesystem;

// Cross-platform default paths
#define IMG_PATH "../data/images/"     // Input images directory
#define IMG_PATH_FINAL "./../data/cpu_output/"      // Output images directory

#define FILTER_SIZE 3     // Convolution filter size (3x3)
#define POOLING_SIZE 3    // Max-pooling window size (3x3)

// Function declarations
string getCurrDir();                              // Retrieve current working directory
vector<fs::path> getFiles(const string &path);    // List files in a directory
bool createDirectory(const string &path);          // Create a directory (including parents)
int parseArgs(int argc, char *argv[]);             // Parse command-line arguments
vector<Mat> conv2D(const string &image_path);     // Perform 2D convolution with six filters
Mat pool2D_max(const Mat &image);                 // Perform max-pooling on a single image

// Print program usage instructions
void print_usage(const char *prog_name)
{
    cerr << "Usage: " << prog_name << " [-n NUM_IMAGES]" << endl;
    cerr << "  -n NUM_IMAGES : Number of images to process (default: all)" << endl;
}

// Create directory (including parent directories) in a cross-platform way
bool createDirectory(const string &path)
{
    try {
        fs::create_directories(path);
        return true;
    } catch (const fs::filesystem_error &e) {
        cerr << "Error creating directory: " << e.what() << endl;
        return false;
    }
}

// Parse command-line arguments to get optional number of images to process
int parseArgs(int argc, char *argv[])
{
    int num_images = -1;  // Default: process all images
    int opt;
#ifdef _WIN32
    // Windows: simple manual parsing
      for (int i = 1; i < argc; i++) {
          if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
              num_images = atoi(argv[i + 1]);
              if (num_images <= 0) {
                  cerr << "Error: Number of images must be positive" << endl;
                  print_usage(argv[0]);
                  return -1;
              }
              break;
          }
      }
#else
    // Unix/Linux: use getopt for robust parsing
    while ((opt = getopt(argc, argv, "n:")) != -1) {
        switch (opt) {
            case 'n':
                num_images = atoi(optarg);
                if (num_images <= 0) {
                    cerr << "Error: Number of images must be positive" << endl;
                    print_usage(argv[0]);
                    return -1;
                }
                break;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }
#endif
    return num_images;
}

int main(int argc, char *argv[])
{
    int num_images = parseArgs(argc, argv);
    if (num_images == -1 && argc > 1) return 1;

    cout << "Current working directory: " << getCurrDir() << endl;

    if (!createDirectory(IMG_PATH_FINAL)) {
        cerr << "Error: Could not create output directory" << endl;
        return 1;
    }

    vector<fs::path> images;
    try {
        images = getFiles(IMG_PATH);
        cout << "Found " << images.size() << " images" << endl;
    } catch (const exception &e) {
        cerr << "Error accessing image directory: " << e.what() << endl;
        cerr << "Ensure directory exists: " << IMG_PATH << endl;
        return 1;
    }

    if (num_images > 0 && static_cast<size_t>(num_images) < images.size()) {
        images.resize(num_images);
        cout << "Processing " << images.size() << " images" << endl;
    }

    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < images.size(); i++) {
        vector<Mat> convolved_images = conv2D(images[i].string());
        if (!convolved_images.empty()) {
            Mat last_convolved = convolved_images.back();
            Mat pooled_image = pool2D_max(last_convolved);
            string filename = images[i].filename().string();
            string output_filename = "final_" + filename;
            imwrite(IMG_PATH_FINAL + output_filename, pooled_image);
        }
    }
    auto end = high_resolution_clock::now();
    auto duration_ms = duration_cast<milliseconds>(end - start);
    cout << "Time taken: " << duration_ms.count() << " ms" << endl;
    return 0;
}

// Get the current working directory in a cross-platform way
string getCurrDir()
{
    char cwd[1024];
    if (GETCWD(cwd, sizeof(cwd)) != nullptr) return string(cwd);
    cerr << "Failed to get current working directory." << endl;
    return string();
}

// List all files in the given directory path
vector<fs::path> getFiles(const string &path)
{
    vector<fs::path> files;
    for (const auto &entry : fs::directory_iterator(path)) files.push_back(entry.path());
    return files;
}

// Perform 2D convolution on the input image with six predefined 3x3 filters
vector<Mat> conv2D(const string &image_path)
{
    vector<vector<int>> filter_vertical{{0,1,0},{0,1,0},{0,1,0}};
    vector<vector<int>> filter_horizontal{{0,0,0},{1,1,1},{0,0,0}};
    vector<vector<int>> filter_diag1{{0,0,1},{0,1,0},{1,0,0}};
    vector<vector<int>> filter_diag2{{1,0,0},{0,1,0},{0,0,1}};
    vector<vector<int>> filter_x{{1,0,1},{0,1,0},{1,0,1}};
    vector<vector<int>> filter_round{{0,1,0},{1,0,1},{0,1,0}};

    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    if (image.empty()) { cerr << "Warning: Unable to read image " << image_path << endl; return {}; }

    int out_width  = image.cols - FILTER_SIZE + 1;
    int out_height = image.rows - FILTER_SIZE + 1;
    Mat out_vert   = Mat::zeros(out_height, out_width, CV_8UC1);
    Mat out_horiz  = Mat::zeros(out_height, out_width, CV_8UC1);
    Mat out_d1     = Mat::zeros(out_height, out_width, CV_8UC1);
    Mat out_d2     = Mat::zeros(out_height, out_width, CV_8UC1);
    Mat out_x      = Mat::zeros(out_height, out_width, CV_8UC1);
    Mat out_round  = Mat::zeros(out_height, out_width, CV_8UC1);

    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            int sv=0, sh=0, s1=0, s2=0, sx=0, sr=0;
            for (int fi=0; fi<FILTER_SIZE; fi++) {
                for (int fj=0; fj<FILTER_SIZE; fj++) {
                    int p = image.at<uchar>(i+fi, j+fj);
                    sv += p*filter_vertical[fi][fj];
                    sh += p*filter_horizontal[fi][fj];
                    s1 += p*filter_diag1[fi][fj];
                    s2 += p*filter_diag2[fi][fj];
                    sx += p*filter_x[fi][fj];
                    sr += p*filter_round[fi][fj];
                }
            }
            out_vert.at<uchar>(i,j)=sv;
            out_horiz.at<uchar>(i,j)=sh;
            out_d1.at<uchar>(i,j)=s1;
            out_d2.at<uchar>(i,j)=s2;
            out_x.at<uchar>(i,j)=sx;
            out_round.at<uchar>(i,j)=sr;
        }
    }
    return {out_vert, out_horiz, out_d1, out_d2, out_x, out_round};
}

// Perform max-pooling with a 3x3 window on a single-channel image
Mat pool2D_max(const Mat &image)
{
    if (image.empty()) { cerr << "Warning: Empty image passed to pooling." << endl; return Mat(); }
    int pooled_w = image.cols / POOLING_SIZE;
    int pooled_h = image.rows / POOLING_SIZE;
    Mat pooled = Mat::zeros(pooled_h, pooled_w, CV_8UC1);

    for (int i=0; i<pooled_h; i++) {
        for (int j=0; j<pooled_w; j++) {
            int mval = numeric_limits<int>::min();
            for (int pi=0; pi<POOLING_SIZE; pi++) {
                for (int pj=0; pj<POOLING_SIZE; pj++) {
                    int y=i*POOLING_SIZE+pi, x=j*POOLING_SIZE+pj;
                    if (y<image.rows && x<image.cols)
                        mval = max(mval, (int)image.at<uchar>(y,x));
                }
            }
            pooled.at<uchar>(i,j)=static_cast<uchar>(mval);
        }
    }
    return pooled;
}