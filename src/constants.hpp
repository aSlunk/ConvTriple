#ifndef CONSTANTS_HPP_
#define CONSTANTS_HPP_

#include <chrono>
#include <thread>

#ifndef USE_CONV_CUDA
#define USE_CONV_CUDA 0
#endif

#ifdef COLOR
#define NC "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define PURPLE "\033[35m"
#else
#define NC ""
#define RED ""
#define GREEN ""
#define PURPLE ""
#endif

#define EXEC_FAILED -1
// #define PROTO 1 // 1 or 2

using Unit    = std::chrono::microseconds;
using measure = std::chrono::high_resolution_clock;

const int N_THREADS = std::max(1u, std::thread::hardware_concurrency());

constexpr size_t filter_prec = 0ULL;

constexpr uint64_t BIT_LEN   = 32;
constexpr uint64_t POLY_MOD  = 1ULL << 12;
constexpr uint64_t PLAIN_MOD = 1ULL << BIT_LEN;

constexpr uint64_t MOD         = PLAIN_MOD;
constexpr uint64_t moduloMask  = MOD - 1;
constexpr uint64_t moduloMidPt = MOD / 2;

namespace Utils {

template <class T>
std::tuple<int, int> pad_zero(const T* src, std::vector<uint32_t>& dest, const int& channels,
                              const int& height, const int& width, const size_t& padding,
                              const int& batchsize);

}

template <class T>
std::tuple<int, int> Utils::pad_zero(const T* src, std::vector<uint32_t>& dest, const int& channels,
                                     const int& height, const int& width, const size_t& padding,
                                     const int& batchsize) {
    size_t new_h   = height + padding * 2;
    size_t new_w   = width + padding * 2;
    size_t new_dim = new_h * new_w;

    dest.clear();
    dest.resize(new_dim * channels * batchsize, 0);

    size_t old_dim = width * height;
    for (int b = 0; b < batchsize; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    dest[b * new_dim * channels + c * new_dim + padding * new_w + h * new_w
                         + padding + w]
                        = src[b * old_dim * channels + c * old_dim + h * width + w];
                }
            }
        }
    }
    return std::make_tuple(new_h, new_w);
}

#endif