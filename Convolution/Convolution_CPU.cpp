// Convolution_CPU.cpp

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <omp.h>

namespace fs = std::filesystem;

// Kích thước filter và số lượng filter
constexpr int F = 3;
constexpr int NUM_FILTERS = 6;

// Tạo thư mục nếu chưa có
bool createDir(const std::string& path) {
    try {
        fs::create_directories(path);
        return true;
    } catch (...) {
        std::cerr << "Failed to create directory " << path << "\n";
        return false;
    }
}

// Đọc danh sách file ảnh (đuôi hợp lệ, skip hidden)
std::vector<std::string> listFiles(const std::string& path) {
    static const std::set<std::string> exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"
    };
    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(path)) {
        if (!fs::is_regular_file(e.status())) continue;
        std::string fn = e.path().filename().string();
        if (fn.empty() || fn[0]=='.') continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (exts.count(ext)) files.push_back(e.path().string());
    }
    return files;
}

// Tạo bộ filter trên host
std::vector<std::vector<std::vector<int>>> createFilters() {
    return {
        {{0,1,0},{0,1,0},{0,1,0}},
        {{0,0,0},{1,1,1},{0,0,0}},
        {{0,0,1},{0,1,0},{1,0,0}},
        {{1,0,0},{0,1,0},{0,0,1}},
        {{1,0,1},{0,1,0},{1,0,1}},
        {{0,1,0},{1,0,1},{0,1,0}}
    };
}

// Convolve một ảnh
std::vector<cv::Mat> convolve(const cv::Mat& img,
    const std::vector<std::vector<std::vector<int>>>& filters)
{
    int H = img.rows, W = img.cols, K = F;
    int outH = H - K + 1, outW = W - K + 1;
    std::vector<cv::Mat> outs(NUM_FILTERS,
        cv::Mat::zeros(outH, outW, CV_8UC1));
    for (int i = 0; i < outH; ++i) {
        for (int j = 0; j < outW; ++j) {
            for (int k = 0; k < NUM_FILTERS; ++k) {
                int sum = 0;
                for (int y = 0; y < K; ++y)
                    for (int x = 0; x < K; ++x)
                        sum += img.at<uchar>(i+y, j+x)
                             * filters[k][y][x];
                outs[k].at<uchar>(i,j) = static_cast<uchar>(sum);
            }
        }
    }
    return outs;
}

int main(int argc, char* argv[]) {
    // Thư mục input/output
    const std::string catIn  = "data/training_set/cats/";
    const std::string dogIn  = "data/training_set/dogs/";
    const std::string catOut = "data/Convolution_CPU/cats_output/";
    const std::string dogOut = "data/Convolution_CPU/dogs_output/";
    if (!createDir(catOut) || !createDir(dogOut)) return 1;

    // 1) READ: load danh sách file và đọc ảnh vào memory
    auto t_read_start = std::chrono::high_resolution_clock::now();

    auto cats = listFiles(catIn);
    auto dogs = listFiles(dogIn);
    std::vector<std::string> allImgs = cats;
    allImgs.insert(allImgs.end(), dogs.begin(), dogs.end());
    int nTotal = (int)allImgs.size();
    if (nTotal==0) {
        std::cerr<<"No images to process!\n"; return 1;
    }

    // Đọc mẫu để lấy kích thước
    cv::Mat sample = cv::imread(allImgs[0], cv::IMREAD_GRAYSCALE);
    int H = sample.rows, W = sample.cols;

    // Đọc tất cả ảnh vào vector
    std::vector<cv::Mat> images;
    images.reserve(nTotal);
    for (auto &p : allImgs) {
        cv::Mat m = cv::imread(p, cv::IMREAD_GRAYSCALE);
        if (m.empty()) continue;
        if (m.rows!=H || m.cols!=W)
            cv::resize(m, m, cv::Size(W,H));
        images.push_back(m);
    }
    nTotal = (int)images.size();

    auto t_read_end = std::chrono::high_resolution_clock::now();


    // 2) COMPUTE: chạy convolution
    auto filters = createFilters();

    auto t_compute_start = std::chrono::high_resolution_clock::now();

    // lưu output vào một vector song song
    std::vector<std::vector<cv::Mat>> outputs(nTotal);
    #pragma omp parallel for
    for (int i = 0; i < nTotal; ++i)
        outputs[i] = convolve(images[i], filters);

    auto t_compute_end = std::chrono::high_resolution_clock::now();


    // 3) WRITE: ghi ảnh ra đĩa
    auto t_write_start = std::chrono::high_resolution_clock::now();

    int catCount = (int)cats.size();
    for (int i = 0; i < nTotal; ++i) {
        bool isCat = (i < catCount);
        std::string outDir = isCat ? catOut : dogOut;
        std::string name = fs::path(allImgs[i]).filename().string();
        for (int k = 0; k < NUM_FILTERS; ++k) {
            cv::imwrite(
                outDir + "f" + std::to_string(k) + "_" + name,
                outputs[i][k]
            );
        }
    }

    auto t_write_end = std::chrono::high_resolution_clock::now();


    // 4) TOTAL
    auto t_total_start = t_read_start;
    auto t_total_end   = t_write_end;

    // In báo cáo
    using D = std::chrono::duration<double>;
    std::cout << std::fixed << std::setprecision(4)
              << "CPU read time    : " << D(t_read_end   - t_read_start).count() << " s\n"
              << "CPU compute time : " << D(t_compute_end - t_compute_start).count() << " s\n"
              << "CPU write time   : " << D(t_write_end  - t_write_start).count() << " s\n"
              << "CPU total time   : " << D(t_total_end  - t_total_start).count() << " s\n";

    return 0;
}
