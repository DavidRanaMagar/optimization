// Convolution_GPU.cu

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;

// Kích thước filter và số lượng filter
constexpr int F = 3;
constexpr int NUM_FILTERS = 6;

// Copy bộ filter vào constant memory
__constant__ int d_filters[NUM_FILTERS][F][F];

// Kernel convolution
__global__ void convKernel(const unsigned char* d_img, int H, int W,
                           unsigned char* d_out, int outH, int outW, int n) {
    int imgIdx = blockIdx.z;
    int row    = blockIdx.y*blockDim.y + threadIdx.y;
    int col    = blockIdx.x*blockDim.x + threadIdx.x;
    if (imgIdx<n && row<outH && col<outW) {
        const unsigned char* img = d_img + size_t(imgIdx)*H*W;
        unsigned char* outb      = d_out + size_t(imgIdx)*NUM_FILTERS*outH*outW;
        for (int k=0; k<NUM_FILTERS; ++k) {
            int sum = 0;
            for (int i=0;i<F;++i)
                for (int j=0;j<F;++j)
                    sum += d_filters[k][i][j]
                         * img[(row+i)*W + (col+j)];
            size_t idx = size_t(k)*outH*outW + size_t(row)*outW + col;
            outb[idx] = static_cast<unsigned char>(sum);
        }
    }
}

bool createDir(const std::string& path) {
    try { fs::create_directories(path); return true; }
    catch(...) { std::cerr<<"Cannot create "<<path<<"\n"; return false;}
}

std::vector<std::string> listFiles(const std::string& path) {
    static const std::set<std::string> exts = {
        ".jpg",".jpeg",".png",".bmp",".tiff",".tif"
    };
    std::vector<std::string> files;
    for (auto& e: fs::directory_iterator(path)) {
        if (!fs::is_regular_file(e.status())) continue;
        auto fn = e.path().filename().string();
        if (fn.empty()||fn[0]=='.') continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);
        if (exts.count(ext)) files.push_back(e.path().string());
    }
    return files;
}

int main(int argc, char* argv[]) {
    // Thư mục input/output
    const std::string catIn  = "data/training_set/cats/";
    const std::string dogIn  = "data/training_set/dogs/";
    const std::string catOut = "data/Convolution_GPU/cats_output/";
    const std::string dogOut = "data/Convolution_GPU/dogs_output/";
    if (!createDir(catOut)||!createDir(dogOut)) return 1;

    // 1) READ: load danh sách và nạp ảnh vào h_img[]
    auto t_read_start = std::chrono::high_resolution_clock::now();

    auto cats = listFiles(catIn);
    auto dogs = listFiles(dogIn);
    std::vector<std::string> allImgs = cats;
    allImgs.insert(allImgs.end(), dogs.begin(), dogs.end());
    int nTotal = (int)allImgs.size();
    if (nTotal==0) { std::cerr<<"No images!\n"; return 1; }

    cv::Mat sample = cv::imread(allImgs[0], cv::IMREAD_GRAYSCALE);
    int H = sample.rows, W = sample.cols;
    int outH = H - F + 1, outW = W - F + 1;

    // Chuẩn bị filter và copy vào GPU
    int h_filters[NUM_FILTERS][F][F] = {
        {{0,1,0},{0,1,0},{0,1,0}},
        {{0,0,0},{1,1,1},{0,0,0}},
        {{0,0,1},{0,1,0},{1,0,0}},
        {{1,0,0},{0,1,0},{0,0,1}},
        {{1,0,1},{0,1,0},{1,0,1}},
        {{0,1,0},{1,0,1},{0,1,0}}
    };
    cudaMemcpyToSymbol(d_filters, h_filters, sizeof(h_filters));

    size_t imgBytes = size_t(nTotal)*H*W;
    size_t outBytes = size_t(nTotal)*NUM_FILTERS*outH*outW;
    unsigned char *h_img = new unsigned char[imgBytes];
    unsigned char *h_out = new unsigned char[outBytes];

    // load vào buffer
    for (int i=0;i<nTotal;++i) {
        cv::Mat m = cv::imread(allImgs[i], cv::IMREAD_GRAYSCALE);
        if (m.empty()) continue;
        if (m.rows!=H||m.cols!=W) cv::resize(m,m,cv::Size(W,H));
        std::memcpy(h_img + size_t(i)*H*W, m.data, H*W);
    }

    auto t_read_end = std::chrono::high_resolution_clock::now();


    // 2) H2D copy
    auto t_h2d_start = std::chrono::high_resolution_clock::now();

    unsigned char *d_img = nullptr, *d_out = nullptr;
    cudaMalloc(&d_img, imgBytes);
    cudaMalloc(&d_out, outBytes);
    cudaMemcpy(d_img, h_img, imgBytes, cudaMemcpyHostToDevice);

    auto t_h2d_end = std::chrono::high_resolution_clock::now();


    // 3) KERNEL
    auto t_kernel_start = std::chrono::high_resolution_clock::now();

    const int TILE = 16;
    dim3 block(TILE,TILE);
    dim3 grid((outW+TILE-1)/TILE, (outH+TILE-1)/TILE, nTotal);
    convKernel<<<grid,block>>>(d_img, H, W, d_out, outH, outW, nTotal);
    cudaDeviceSynchronize();

    auto t_kernel_end = std::chrono::high_resolution_clock::now();


    // 4) D2H copy
    auto t_d2h_start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost);

    auto t_d2h_end = std::chrono::high_resolution_clock::now();


    // 5) WRITE: ghi ảnh ra đĩa
    auto t_write_start = std::chrono::high_resolution_clock::now();

    int catCount = (int)cats.size();
    for (int i=0;i<nTotal;++i) {
        bool isCat = (i<catCount);
        std::string outDir = isCat?catOut:dogOut;
        std::string name   = fs::path(allImgs[i]).filename().string();
        for (int k=0;k<NUM_FILTERS;++k) {
            cv::Mat out(outH,outW,CV_8UC1,
                        h_out + (size_t(i)*NUM_FILTERS + k)*outH*outW);
            cv::imwrite(outDir + "f" + std::to_string(k)
                        + "_" + name, out);
        }
    }

    auto t_write_end = std::chrono::high_resolution_clock::now();


    // TOTAL
    auto t_total_start = t_read_start;
    auto t_total_end   = t_write_end;

    using D = std::chrono::duration<double>;
    std::cout << std::fixed << std::setprecision(4)
              << "GPU read time   : " << D(t_read_end    - t_read_start).count()  << " s\n"
              << "GPU H2D time    : " << D(t_h2d_end     - t_h2d_start).count()  << " s\n"
              << "GPU kernel time : " << D(t_kernel_end  - t_kernel_start).count() << " s\n"
              << "GPU D2H time    : " << D(t_d2h_end     - t_d2h_start).count()  << " s\n"
              << "GPU write time  : " << D(t_write_end   - t_write_start).count()  << " s\n"
              << "GPU total time  : " << D(t_total_end   - t_total_start).count()  << " s\n";

    // Dọn dẹp
    delete[] h_img;
    delete[] h_out;
    cudaFree(d_img);
    cudaFree(d_out);

    return 0;
}
