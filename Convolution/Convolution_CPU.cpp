#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <getopt.h>
#include <omp.h>

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

void getCurrDir();
vector<string> getFiles(const string &path);
vector<vector<vector<int>>> createFilters();
bool createDirectory(const string &path);

vector<Mat> conv2D_static(const string &, // This function takes static filters as parameters
                          const vector<vector<int>>,
                          const vector<vector<int>>,
                          const vector<vector<int>>,
                          const vector<vector<int>>,
                          const vector<vector<int>>,
                          const vector<vector<int>>);

vector<Mat> conv2D( // The function takes a vector of filters as a parameter
        const string &,
        const vector<vector<vector<int>>> &);

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

    // Create filters (test use, for conv2D_static)
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

    // Create filters (actual use, for conv2D)
    vector<vector<vector<int>>> filters = createFilters();

    // Process cat images
    auto start = high_resolution_clock::now();
    cout << "Processing cat images..." << endl;
// commit below line for native no optimization
#pragma omp parallel for // OpenMP can run this loop 10X faster
    for (size_t i = 0; i < cat_images.size(); i++)
    {
        // The function that takes static filters as parameters
        vector<Mat> new_images = conv2D_static(cat_images[i],
                                               filter_vertical_line,
                                               filter_horiz_line,
                                               filter_diagonal_lbru_line,
                                               filter_diagonal_lurb_line,
                                               filter_diagonal_x_line,
                                               filter_round_line);

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
    cout << "Processing dog images..." << endl;
// commit below line for native no optimization
#pragma omp parallel for // OpenMP can run this loop 10X faster
    for (size_t i = 0; i < dog_images.size(); i++)
    {
        // The function that takes static filters as parameters
        vector<Mat> new_images = conv2D_static(dog_images[i],
                                               filter_vertical_line,
                                               filter_horiz_line,
                                               filter_diagonal_lbru_line,
                                               filter_diagonal_lurb_line,
                                               filter_diagonal_x_line,
                                               filter_round_line);

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

vector<Mat> conv2D_static(
        const string &image_path,
        const vector<vector<int>> filter_vertical_line,
        const vector<vector<int>> filter_horizontal_line,
        const vector<vector<int>> filter_diagonal_lbru_line,
        const vector<vector<int>> filter_diagonal_lurb_line,
        const vector<vector<int>> filter_diagonal_x_line,
        const vector<vector<int>> filter_round_line)
{
    Mat image = imread(image_path, IMREAD_GRAYSCALE);

    if (!image.empty())
    {
        int image_width = image.cols;
        int image_height = image.rows;

        int new_image_width = image_width - FILTER_SIZE;
        int new_image_height = image_height - FILTER_SIZE;

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

                        int filter_value = filter_vertical_line[filter_i - i][filter_j - j];
                        vertical_sum += image_value * filter_value;

                        filter_value = filter_horizontal_line[filter_i - i][filter_j - j];
                        horiz_sum += image_value * filter_value;

                        filter_value = filter_diagonal_lbru_line[filter_i - i][filter_j - j];
                        diagonal_lbru_sum += image_value * filter_value;

                        filter_value = filter_diagonal_lurb_line[filter_i - i][filter_j - j];
                        diagonal_lurb_sum += image_value * filter_value;

                        filter_value = filter_diagonal_x_line[filter_i - i][filter_j - j];
                        diagonal_x_sum += image_value * filter_value;

                        filter_value = filter_round_line[filter_i - i][filter_j - j];
                        round_sum += image_value * filter_value;
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

        vector<Mat> new_images{
                new_image_extract_vertical,
                new_image_extract_horiz,
                new_image_extract_diagonal_lbru,
                new_image_extract_diagonal_lurb,
                new_image_extract_diagonal_x,
                new_image_extract_round};
        return new_images;
    }

    vector<Mat> new_images;
    return new_images;
}

vector<Mat> conv2D(const string &image_path, const vector<vector<vector<int>>> &filters)
{
Mat image = imread(image_path, IMREAD_GRAYSCALE);

if (!image.empty())
{
// Original image size
int image_width = image.cols;
int image_height = image.rows;

// New image size
int new_image_width = image_width - FILTER_SIZE;
int new_image_height = image_height - FILTER_SIZE;

// Init the vector to store the new images
vector<Mat> new_images;
for (int i = 0; i < NUM_FILTERS; i++)
{
new_images.push_back(Mat::zeros(new_image_height, new_image_width, CV_8UC1));
}

// Loop for each pixel of new image
for (int i = 0; i < new_image_height; i++)
{
for (int j = 0; j < new_image_width; j++)
{
// Init vector to store the value of this pixel of each filter
vector<int> pixel_sum;
for (int pixel = 0; pixel < NUM_FILTERS; pixel++)
{
pixel_sum.push_back(0);
}

for (int filter_i = i; filter_i < i + FILTER_SIZE; filter_i++)
{
for (int filter_j = j; filter_j < j + FILTER_SIZE; filter_j++)
{
// The value of the pixel of original image
int image_value = image.at<uchar>(filter_i, filter_j);

// Loop each filter
for (size_t filter = 0; filter < filters.size(); filter++)
{
int filter_value = filters[filter][filter_i - i][filter_j - j];
int filter_sum = image_value * filter_value;

pixel_sum[filter] += filter_sum;
}
}
}

// Save the calculated new pixel to new images
for (size_t image = 0; image < new_images.size(); image++)
{
new_images[image].at<uchar>(i, j) = pixel_sum[image];
}
}
}

return new_images;
}

vector<Mat> new_images;
return new_images;
}