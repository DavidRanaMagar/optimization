#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

// Cross-platform filesystem handling
#if defined(_WIN32) || defined(_WIN64)
#include <direct.h>
    #include <windows.h>
    #define PATH_SEPARATOR "\\"
    #define MKDIR(dir) _mkdir(dir)
#else
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#define PATH_SEPARATOR "/"
#define MKDIR(dir) mkdir(dir, 0755)
#endif

using namespace cv;
using namespace std;
using namespace std::chrono;

// Configurations
const int NUM_FILTERS = 6;
const int FILTER_SIZE = 3;
const int POOLING_SIZE = 3;

// Function to normalize matrix to 0-255 range
Mat normalizeMatrix(const Mat &input)
{
    Mat normalized;
    double minVal, maxVal;
    minMaxLoc(input, &minVal, &maxVal);

    if (maxVal == minVal)
    {
        return Mat::zeros(input.size(), CV_8UC1);
    }

    input.convertTo(normalized, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    return normalized;
}

Mat applyConvAndPool(const Mat &image)
{
    if (image.empty())
    {
        return Mat();
    }
    vector<vector<int>> filter_vertical_line{{0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
    vector<vector<int>> filter_horizontal_line{{0, 0, 0},{1, 1, 1},{0, 0, 0}};
    vector<vector<int>> filter_diagonal_lbru_line{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}};
    vector<vector<int>> filter_diagonal_lurb_line{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    vector<vector<int>> filter_diagonal_x_line{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}};
    vector<vector<int>> filter_round_line{{0, 1, 0}, {1, 0, 1}, {0, 1, 0}};

    int image_width = image.cols;
    int image_height = image.rows;

    int new_image_width = image_width - FILTER_SIZE + 1;
    int new_image_height = image_height - FILTER_SIZE + 1;

    Mat convolved = Mat::zeros(new_image_height, new_image_width, CV_8UC1);

    for (int i = 0; i < new_image_height; i++)
    {
        for (int j = 0; j < new_image_width; j++)
        {
            int round_sum = 0;

            for (int filter_i = i; filter_i < i + FILTER_SIZE; filter_i++)
            {
                for (int filter_j = j; filter_j < j + FILTER_SIZE; filter_j++)
                {
                    int image_value = image.at<uchar>(filter_i, filter_j);
                    int filter_value = filter_round_line[filter_i - i][filter_j - j];
                    round_sum += image_value * filter_value;
                }
            }
            convolved.at<uchar>(i, j) = round_sum;
        }
    }

    // Step 2: Max Pooling
    int pooledWidth = convolved.cols / POOLING_SIZE;
    int pooledHeight = convolved.rows / POOLING_SIZE;
    Mat pooledResult = Mat::zeros(pooledHeight, pooledWidth, CV_8UC1);

    for (int i = 0; i < pooledHeight; i++) {
        for (int j = 0; j < pooledWidth; j++) {
            uchar maxVal = 0;
            int startY = i * POOLING_SIZE;
            int startX = j * POOLING_SIZE;

            for (int y = startY; y < startY + POOLING_SIZE && y < convolved.rows; y++) {
                for (int x = startX; x < startX + POOLING_SIZE && x < convolved.cols; x++) {
                    maxVal = max(maxVal, convolved.at<uchar>(y, x));
                }
            }
            pooledResult.at<uchar>(i, j) = maxVal;
        }
    }

    return pooledResult;
}

// Cross-platform directory creation
bool createDirectory(const string &path)
{
    return MKDIR(path.c_str()) == 0 || errno == EEXIST;
}

// Cross-platform file listing
vector<string> getImageFiles(const string &directory)
{
    vector<string> imageFiles;

#if defined(_WIN32) || defined(_WIN64)
    WIN32_FIND_DATA fileData;
    HANDLE hFind;
    string searchPattern = directory + PATH_SEPARATOR + "*";
    
    hFind = FindFirstFile(searchPattern.c_str(), &fileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                string filename = fileData.cFileName;
                string ext = filename.substr(filename.find_last_of(".") + 1);
                transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (ext == "jpg" || ext == "jpeg" || ext == "png") {
                    imageFiles.push_back(directory + PATH_SEPARATOR + filename);
                }
            }
        } while (FindNextFile(hFind, &fileData));
        FindClose(hFind);
    }
#else
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string filename = ent->d_name;

            // Skip directories
            struct stat s;
            string fullpath = directory + PATH_SEPARATOR + filename;
            if (stat(fullpath.c_str(), &s) == 0 && !S_ISDIR(s.st_mode)) {
                if (filename.length() > 4) {
                    string ext = filename.substr(filename.find_last_of(".") + 1);
                    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                    if (ext == "jpg" || ext == "jpeg" || ext == "png") {
                        imageFiles.push_back(fullpath);
                    }
                }
            }
        }
        closedir(dir);
    }
#endif

    return imageFiles;
}

// Cross-platform filename extraction
string getFilename(const string &path)
{
    size_t sepPos = path.find_last_of("/\\");
    if (sepPos != string::npos) {
        return path.substr(sepPos + 1);
    }
    return path;
}

int main()
{
    // Start total processing timer
    auto total_start = high_resolution_clock::now();

    // Cross-platform path construction
    string inputDir = ".." + string(PATH_SEPARATOR) + "data" + string(PATH_SEPARATOR) + "images";
    string outputDir = ".." + string(PATH_SEPARATOR) + "data" + string(PATH_SEPARATOR) + "cpu_output";

    // Create output directory if it doesn't exist
    if (!createDirectory(outputDir))
    {
        cerr << "Error: Could not create output directory!" << endl;
        return -1;
    }

    // Get all image files in the input directory
    vector<string> imageFiles = getImageFiles(inputDir);

    if (imageFiles.empty())
    {
        cerr << "Error: No image files found in " << inputDir << endl;
        return -1;
    }

    // Initialize counters and timers
    int processed_count = 0;
    int failed_count = 0;
    long long total_processing_time = 0;

    cout << "Starting processing of " << imageFiles.size() << " images..." << endl;

    // Process each image
    for (const auto &imagePath : imageFiles)
    {
        auto image_start = high_resolution_clock::now();

        // Load image
        Mat image = imread(imagePath, IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cerr << "Warning: Could not load image " << imagePath << " - skipping" << endl;
            failed_count++;
            continue;
        }

        // Process image
        Mat result = applyConvAndPool(image);
        if (result.empty())
        {
            cerr << "Warning: Processing failed for " << imagePath << " - skipping" << endl;
            failed_count++;
            continue;
        }

        // Create output filename - cross-platform way
        string filename = getFilename(imagePath);
        string outputPath = outputDir + PATH_SEPARATOR + "final_" + filename;

        // Save result
        if (!imwrite(outputPath, result))
        {
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