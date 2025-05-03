#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <filesystem> // C++17 filesystem
#include <getopt.h>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

// Platform-specific includes
#ifdef _WIN32
#include <direct.h>  // For _getcwd on Windows
    #define GETCWD _getcwd
    #define PATH_SEPARATOR "\\"
#else
#include <unistd.h>  // For getcwd on Linux/Unix
#define GETCWD getcwd
#define PATH_SEPARATOR "/"
#endif

using namespace std;
using namespace cv;
using namespace chrono;
namespace fs = std::filesystem;

// Cross-platform path definitions
#define IMG_PATH ".." PATH_SEPARATOR "data" PATH_SEPARATOR "images" PATH_SEPARATOR
#define IMG_PATH_FINAL ".." PATH_SEPARATOR "data" PATH_SEPARATOR "cpu_output" PATH_SEPARATOR

#define NUM_FILTERS 6
#define FILTER_SIZE 3
#define POOLING_SIZE 3

// Function declarations
string getCurrDir();
vector<fs::path> getFiles(const string &path);
bool createDirectory(const string &path);
vector<Mat> conv2D(const string &);
Mat pool2D_max(const Mat &);

void print_usage(const char *prog_name)
{
    cerr << "Usage: " << prog_name << " [-n NUM_IMAGES]" << endl;
    cerr << "Options:" << endl;
    cerr << "  -n NUM_IMAGES  Number of images to process from each category (default: all)" << endl;
}

// Cross-platform directory creation
bool createDirectory(const string &path)
{
    try
    {
        fs::create_directories(path);
        return true;
    }
    catch (const fs::filesystem_error &e)
    {
        cerr << "Error creating directory: " << e.what() << endl;
        return false;
    }
}

// Cross-platform argument parsing
int parseArgs(int argc, char *argv[])
{
    int num_images = -1;
    int opt;

#ifdef _WIN32
    // Simple command-line parsing for Windows
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
    // Use getopt for Unix/Linux
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
    // Parse command line arguments
    int num_images = parseArgs(argc, argv);
    if (num_images == -1 && argc > 1) {
        return 1;
    }

    // Display current directory
    cout << "Current working directory: " << getCurrDir() << endl;

    // Create output directory
    if (!createDirectory(IMG_PATH_FINAL))
    {
        cerr << "Error: Could not create output directories" << endl;
        return 1;
    }

    // Get image files
    vector<fs::path> images;
    try {
        images = getFiles(IMG_PATH);
        cout << "Found " << images.size() << " images" << endl;
    }
    catch (const exception& e) {
        cerr << "Error accessing image directory: " << e.what() << endl;
        cerr << "Make sure the directory " << IMG_PATH << " exists" << endl;
        return 1;
    }

    // Limit number of images if specified
    if (num_images > 0)
    {
        if (static_cast<size_t>(num_images) < images.size())
        {
            images.resize(num_images);
        }
        cout << "Processing " << images.size() << " images" << endl;
    }

    // Process images
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < images.size(); i++)
    {
        vector<Mat> new_images = conv2D(images[i].string());
        string filename = images[i].filename().string();

        if (!new_images.empty()) {
            Mat last_convolved = new_images.back(); // Gets the last image
            Mat pooled_image = pool2D_max(last_convolved);

            string output_filename = "final_" + filename;
            imwrite(string(IMG_PATH_FINAL) + output_filename, pooled_image);
        }
    }

    auto end = high_resolution_clock::now();
    auto duration_pol = duration_cast<milliseconds>(end - start);

    cout << "Time taken: " << duration_pol.count() << " ms" << endl;
    return 0;
}

// Cross-platform get current directory
string getCurrDir()
{
    char cwd[1024];
    if (GETCWD(cwd, sizeof(cwd)) != nullptr)
    {
        return string(cwd);
    }
    else
    {
        cerr << "Failed to get current working directory." << endl;
        return "";
    }
}

// Get files from directory
vector<fs::path> getFiles(const string &path)
{
    vector<fs::path> files;
    for (const auto &entry : fs::directory_iterator(path))
    {
        files.push_back(entry.path());
    }
    return files;
}

// Image convolution
vector<Mat> conv2D(const string &image_path)
{
    vector<vector<int>> filter_vertical_line{{0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
    vector<vector<int>> filter_horizontal_line{{0, 0, 0},{1, 1, 1},{0, 0, 0}};
    vector<vector<int>> filter_diagonal_lbru_line{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}};
    vector<vector<int>> filter_diagonal_lurb_line{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    vector<vector<int>> filter_diagonal_x_line{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}};
    vector<vector<int>> filter_round_line{{0, 1, 0}, {1, 0, 1}, {0, 1, 0}};

    Mat image = imread(image_path, IMREAD_GRAYSCALE);

    if (!image.empty())
    {
        int image_width = image.cols;
        int image_height = image.rows;

        int new_image_width = image_width - FILTER_SIZE + 1;
        int new_image_height = image_height - FILTER_SIZE + 1;

        Mat new_image_extract_vertical = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
        Mat new_image_extract_horiz = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
        Mat new_image_extract_diagonal_lbru = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
        Mat new_image_extract_diagonal_lurb = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
        Mat new_image_extract_diagonal_x = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
        Mat new_image_extract_round = Mat::zeros(new_image_height, new_image_width, CV_8UC1);

        for (int i = 0; i < new_image_height; i++)
        {
            for (int j = 0; j < new_image_width; j++)
            {
                int vertical_sum = 0;
                int horiz_sum = 0;
                int diagonal_lbru_sum = 0;
                int diagonal_lurb_sum = 0;
                int diagonal_x_sum = 0;
                int round_sum = 0;

                for (int filter_i = i; filter_i < i + FILTER_SIZE; filter_i++)
                {
                    for (int filter_j = j; filter_j < j + FILTER_SIZE; filter_j++)
                    {
                        int image_value = image.at<uchar>(filter_i, filter_j);

                        vertical_sum += image_value * filter_vertical_line[filter_i - i][filter_j - j];
                        horiz_sum += image_value * filter_horizontal_line[filter_i - i][filter_j - j];
                        diagonal_lbru_sum += image_value * filter_diagonal_lbru_line[filter_i - i][filter_j - j];
                        diagonal_lurb_sum += image_value * filter_diagonal_lurb_line[filter_i - i][filter_j - j];
                        diagonal_x_sum += image_value * filter_diagonal_x_line[filter_i - i][filter_j - j];
                        round_sum += image_value * filter_round_line[filter_i - i][filter_j - j];
                    }
                }

                new_image_extract_vertical.at<uchar>(i, j) = vertical_sum;
                new_image_extract_horiz.at<uchar>(i, j) = horiz_sum;
                new_image_extract_diagonal_lbru.at<uchar>(i, j) = diagonal_lbru_sum;
                new_image_extract_diagonal_lurb.at<uchar>(i, j) = diagonal_lurb_sum;
                new_image_extract_diagonal_x.at<uchar>(i, j) = diagonal_x_sum;
                new_image_extract_round.at<uchar>(i, j) = round_sum;
            }
        }

        return {new_image_extract_vertical, new_image_extract_horiz,
                new_image_extract_diagonal_lbru, new_image_extract_diagonal_lurb,
                new_image_extract_diagonal_x, new_image_extract_round};
    }

    return {};
}

// Pooling operation
Mat pool2D_max(const Mat &image)
{
    if (!image.empty())
    {
        int image_width = image.cols;
        int image_height = image.rows;

        int new_image_width = image_width / POOLING_SIZE;
        int new_image_height = image_height / POOLING_SIZE;

        Mat new_image = Mat::zeros(new_image_height, new_image_width, CV_8UC1);

        for (int i = 0; i < new_image_height; i++)
        {
            for (int j = 0; j < new_image_width; j++)
            {
                int corner_i = i * POOLING_SIZE;
                int corner_j = j * POOLING_SIZE;

                int maximum = INT_MIN;

                for (int pool_i = corner_i; pool_i < corner_i + POOLING_SIZE && pool_i < image_height; pool_i++)
                {
                    for (int pool_j = corner_j; pool_j < corner_j + POOLING_SIZE && pool_j < image_width; pool_j++)
                    {
                        int image_value = image.at<uchar>(pool_i, pool_j);
                        if (image_value > maximum)
                        {
                            maximum = image_value;
                        }
                    }
                }

                new_image.at<uchar>(i, j) = maximum;
            }
        }

        return new_image;
    }

    return Mat();
}