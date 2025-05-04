// Pooling_CPU.cpp
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <algorithm>            // <-- thêm dòng này
#include <opencv2/opencv.hpp>
#include <omp.h>

namespace fs = std::filesystem;
constexpr int POOL_SIZE = 3;
const std::string CAT_INPUT_PATH  = "./data/Convolution/cats_output/";
const std::string DOG_INPUT_PATH  = "./data/Convolution/dogs_output/";
const std::string CAT_OUTPUT_PATH = "./data/Final/cats_final/";
const std::string DOG_OUTPUT_PATH = "./data/Final/dogs_final/";

// Tạo folder nếu chưa tồn tại
bool createDirectory(const std::string &path) {
    try {
        fs::create_directories(path);
        return true;
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Cannot create directory " << path
                  << ": " << e.what() << "\n";
        return false;
    }
}

// Liệt kê file trong thư mục
std::vector<fs::path> getFiles(const std::string &directory) {
    std::vector<fs::path> files;
    for (auto &entry : fs::directory_iterator(directory))
        if (fs::is_regular_file(entry.status()))
            files.push_back(entry.path());
    return files;
}

// Hàm max-pool trên 1 ảnh grayscale
cv::Mat maxPool(const cv::Mat &image) {
    int pooledH = image.rows / POOL_SIZE;
    int pooledW = image.cols / POOL_SIZE;
    cv::Mat pooled(pooledH, pooledW, CV_8UC1);
    for (int i = 0; i < pooledH; ++i) {
        for (int j = 0; j < pooledW; ++j) {
            int maxV = 0;
            int sy = i * POOL_SIZE, sx = j * POOL_SIZE;
            for (int y = 0; y < POOL_SIZE; ++y) {
                for (int x = 0; x < POOL_SIZE; ++x) {
                    // cast pixel về int trước khi so sánh
                    int v = static_cast<int>(image.at<uchar>(sy + y, sx + x));
                    maxV = std::max(maxV, v);
                }
            }
            pooled.at<uchar>(i, j) = static_cast<uchar>(maxV);
        }
    }
    return pooled;
}

// Overload: đọc ảnh từ đường dẫn rồi gọi maxPool
cv::Mat maxPool(const std::string &path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) return {};
    return maxPool(img);
}

int main(int argc, char* argv[]) {
    int numImages = -1;
    if (argc == 3 && std::string(argv[1]) == "-n")
        numImages = std::stoi(argv[2]);
    else if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << " [-n NUM_IMAGES]\n";
        return 1;
    }

    if (!createDirectory(CAT_OUTPUT_PATH) || !createDirectory(DOG_OUTPUT_PATH))
        return 1;

    auto catFiles = getFiles(CAT_INPUT_PATH);
    auto dogFiles = getFiles(DOG_INPUT_PATH);
    if (numImages > 0) {
        if ((int)catFiles.size() > numImages) catFiles.resize(numImages);
        if ((int)dogFiles.size() > numImages) dogFiles.resize(numImages);
    }

    // ---- Total time (I/O + compute) ----
    auto t_total_start = std::chrono::high_resolution_clock::now();

    // ---- Pure compute time ----
    auto t_compute_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < (int)catFiles.size(); ++i) {
        cv::Mat out = maxPool(catFiles[i].string());
        if (!out.empty())
            cv::imwrite(CAT_OUTPUT_PATH + catFiles[i].filename().string(), out);
    }
    #pragma omp parallel for
    for (int i = 0; i < (int)dogFiles.size(); ++i) {
        cv::Mat out = maxPool(dogFiles[i].string());
        if (!out.empty())
            cv::imwrite(DOG_OUTPUT_PATH + dogFiles[i].filename().string(), out);
    }
    auto t_compute_end = std::chrono::high_resolution_clock::now();

    auto t_total_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> compute_time = t_compute_end - t_compute_start;
    std::chrono::duration<double> total_time   = t_total_end - t_total_start;

    std::cout << "Pure CPU pooling time: "
              << std::fixed << std::setprecision(4)
              << compute_time.count() << " seconds\n";
    std::cout << "Total CPU pooling time (I/O + compute): "
              << std::fixed << std::setprecision(4)
              << total_time.count() << " seconds\n";

    return 0;
}
