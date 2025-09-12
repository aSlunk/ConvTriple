#ifndef DEFS_HPP_
#define DEFS_HPP_

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <ostream>
#include <thread>

#include <gemini/cheetah/hom_bn_ss.h>
#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/hom_fc_ss.h>
#include <gemini/cheetah/tensor.h>

#include <seal/seal.h>

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

constexpr seal::sec_level_type SEC_LEVEL = seal::sec_level_type::tc128;

constexpr uint64_t BIT_LEN   = 32;
constexpr uint64_t POLY_MOD  = 1ULL << 12;
constexpr uint64_t PLAIN_MOD = 1ULL << BIT_LEN;

constexpr uint64_t MOD         = PLAIN_MOD;
constexpr uint64_t moduloMask  = MOD - 1;
constexpr uint64_t moduloMidPt = MOD / 2;

namespace Utils {

enum class PROTO {
    AB,
    AB2,
};

enum class Level {
    DEBUG,
    INFO,
    PASSED,
    FAILED,
    ERROR,
};

template <class... Args>
void log(const Level& l, const Args&... args) {
    auto* stream = &std::cerr;
    switch (l) {
    case Level::DEBUG:
        *stream << PURPLE;
        break;
    case Level::PASSED:
        *stream << GREEN;
        break;
    case Level::FAILED:
        *stream << RED;
        break;
    case Level::ERROR:
        stream = &std::cout;
        *stream << RED;
        break;
    default:
        stream = &std::cerr;
    }

    (*stream << ... << args) << (l != Level::INFO ? NC : "") << std::endl;

    if (l == Level::ERROR)
        exit(EXEC_FAILED);
}

template <class T>
inline double to_sec(const T& num) {
    if constexpr (std::is_same<Unit, std::chrono::microseconds>::value) {
        return num / 1'000'000.0;
    }

    log(Level::ERROR, "Unknown <Unit>");
    return 0.0;
}

template <class T>
inline double to_msec(const T& num) {
    if constexpr (std::is_same<Unit, std::chrono::microseconds>::value) {
        return num / 1'000.0;
    }

    log(Level::ERROR, "Unknown <Unit>");
    return 0.0;
}

template <class T>
inline double to_MB(const T& bytes, std::string& unit) {
    // return bytes / 1'000'000.0;
    unit = "MiB";
    return static_cast<double>(bytes) / (1 << 20);
}

template <class Time>
inline size_t time_diff(const Time& start) {
    return std::chrono::duration_cast<Unit>(measure::now() - start).count();
}

struct Result {
    double encryption = 0;
    double cipher_op  = 0;
    double plain_op   = 0;
    double decryption = 0;
    double send_recv  = 0;
    double serial     = 0;
    double bytes      = 0;
    Code ret          = Code::OK;
};

seal::SEALContext init_he_context();
void print_info(const gemini::HomConv2DSS::Meta& meta, const size_t& padding);

gemini::HomFCSS::Meta init_meta_fc(const long& image_h, const long& filter_h);
gemini::HomBNSS::Meta init_meta_bn(const long& image_h, const long& filter_h);

gemini::HomConv2DSS::Meta init_meta_conv(const long& ic, const long& ih, const long& iw,
                                         const long& fc, const long& fh, const long& fw,
                                         const size_t& n_filter, const size_t& stride,
                                         const size_t& padding);

std::vector<gemini::HomFCSS::Meta> init_layers_fc();

/**
 * Assigns a port to each thread
 * @param addr NULL for server otherwise IP-addr
 * @param port [Port..(port + threads)] to use
 * @param threads number of ports to listen
 * @return The created channels
 */
template <class Channel>
Channel** init_ios(const char* addr, const int& port, const size_t& threads, const int& offset = 1);

template <class T>
gemini::Tensor<uint64_t> convert_fix_point(const gemini::Tensor<T>& in);

template <class T>
void print_tensor(const gemini::Tensor<T>& t, const long& channel = 0);

double convert(uint64_t v, int nbits);
gemini::Tensor<double> convert_double(const gemini::Tensor<uint64_t>& in);
template <class Meta>
gemini::Tensor<uint64_t> init_image(const Meta& meta, const double& num);
std::vector<gemini::Tensor<uint64_t>> init_filter(const gemini::HomConv2DSS::Meta& meta,
                                                  const double& num);

template <class T>
void op_inplace(gemini::Tensor<T>& A, const gemini::Tensor<T>& B, std::function<T(T, T)> op);

template <class T>
static inline uint64_t getRingElt(T x) {
    return ((uint64_t)x) & moduloMask;
}

template <class T>
inline uint64_t add(T a, T b) {
    uint64_t sum;
    seal::util::add_uint(&a, 1, b, &sum);
    return getRingElt(sum);
}

static inline int64_t getSignedVal(uint64_t x) {
    assert(x < MOD);
    if (x > moduloMidPt)
        return static_cast<int64_t>(x - MOD);
    else
        return static_cast<int64_t>(x);
}

void add_result(Result& res, const Result& res2);

Result average(const std::vector<Result>& res, bool average_bytes);

double print_results(const Result& res, const size_t& layer, const size_t& batchSize,
                     const size_t& threads, std::ostream& out = std::cout);

void make_csv(const std::vector<Result>& results, const size_t& batchSize, const size_t& threads,
              const std::string& path = "");

template <class T>
std::vector<gemini::Tensor<uint64_t>> to_tensor64(T* buf, const gemini::TensorShape& shape,
                                                  const size_t& batch = 1) {
    std::vector<gemini::Tensor<uint64_t>> res(batch, gemini::Tensor<uint64_t>(shape));

    for (size_t cur = 0; cur < batch; ++cur) {
        uint64_t* data = res[cur].data();
        for (ssize_t i = 0; i < shape.num_elements(); ++i) {
            data[i] = static_cast<uint64_t>(buf[i + cur * shape.num_elements()]);
        }
    }

    return res;
}

template <class T>
bool save_to_file(const char* path, const T* a, const T* b, const T* c, const size_t& n);

template <class T>
bool read_from_file(const char* path, T* a, T* b, T* c, const size_t& n, bool trunc = true);

template <class T>
std::tuple<int, int> pad_zero(std::vector<T>& vec, const int& channels, const int& height,
                              const int& width, const size_t& padding);

} // namespace Utils

template <class Meta>
gemini::Tensor<uint64_t> Utils::init_image(const Meta& meta, const double& num) {
    gemini::Tensor<uint64_t> image(meta.ishape);

    for (int c = 0; c < image.channels(); ++c) {
        for (int i = 0; i < image.height(); ++i) {
            for (int j = 0; j < image.width(); ++j) {
                image(c, i, j) = num + j;
            }
        }
    }

    return image;
}

template <class T>
void Utils::op_inplace(gemini::Tensor<T>& A, const gemini::Tensor<T>& B,
                       std::function<T(T, T)> op) {
    assert(A.shape() == B.shape());

    switch (A.dims()) {
    case 3:
        for (int i = 0; i < A.channels(); ++i) {
            for (int j = 0; j < A.height(); ++j) {
                for (int k = 0; k < A.width(); ++k) {
                    A(i, j, k) = op(A(i, j, k), B(i, j, k));
                }
            }
        }
        break;
    case 2:
        for (int i = 0; i < A.rows(); ++i) {
            for (int j = 0; j < A.cols(); ++j) {
                A(i, j) = op(A(i, j), B(i, j));
            }
        }
        break;
    case 1:
        for (int i = 0; i < A.length(); ++i) {
            A(i) = op(A(i), B(i));
        }
        break;
    }
}

template <class Channel>
Channel** Utils::init_ios(const char* addr, const int& port, const size_t& threads,
                          const int& offset) {
    Channel** res = new Channel*[threads];
    for (size_t wid = 0; wid < threads; ++wid) {
        res[wid] = new Channel(addr, port + wid * offset, true);
    }
    return res;
}

template <class T>
gemini::Tensor<uint64_t> Utils::convert_fix_point(const gemini::Tensor<T>& in) {
    gemini::Tensor<uint64_t> out(in.shape());
    out.tensor() = in.tensor().unaryExpr([](T num) {
        double u    = static_cast<double>(num);
        auto sign   = std::signbit(u);
        uint64_t su = std::floor(std::abs(u * (1 << filter_prec)));
        return getRingElt(sign ? -su : su);
    });

    return out;
}

template <class T>
void Utils::print_tensor(const gemini::Tensor<T>& t, const long& channel) {
    switch (t.dims()) {
    case 3:
        for (long i = 0; i < t.height(); ++i) {
            for (long j = 0; j < t.width(); ++j) {
                std::cerr << static_cast<T>(t(channel, i, j)) << " ";
            }
            std::cerr << "\n";
        }
        break;
    case 2:
        for (long i = 0; i < t.rows(); ++i) {
            for (long j = 0; j < t.cols(); ++j) {
                std::cerr << static_cast<T>(t(i, j)) << " ";
            }
            std::cerr << "\n";
        }
        break;
    case 1:
        for (long i = 0; i < t.length(); ++i) {
            std::cerr << static_cast<T>(t(i)) << "\n";
        }
        break;
    }
}

template <class T>
bool Utils::save_to_file(const char* path, const T* a, const T* b, const T* c, const size_t& n) {
    std::fstream file(path, std::ios_base::out | std::ios_base::app | std::ios_base::binary);

    if (!file.is_open())
        return false;

    file.write((char*)a, n * sizeof(T));
    file.write((char*)b, n * sizeof(T));
    file.write((char*)c, n * sizeof(T));

    file.close();
    return !file.fail();
}

template <class T>
bool Utils::read_from_file(const char* path, T* a, T* b, T* c, const size_t& n, bool trunc) {
    std::fstream file;
    file.open(path, std::ios_base::in | std::ios_base::ate | std::ios_base::binary);
    if (!file.is_open()) {
        log(Level::FAILED, "Couldn't open: ", path);
        return false;
    }

    size_t size = file.tellg();
    if (file.fail()) {
        log(Level::ERROR, "Couln't read file size");
    }

    std::cout << "SIZE: " << size << "\n";
    std::cout << "n: " << n << "\n";
    if (n * sizeof(T) * 3 > size) {
        file.close();
        log(Level::ERROR, "file too small");
        return false;
    }

    file.seekg(0, std::ios::beg);

    file.read((char*)a, n * sizeof(T));
    file.read((char*)b, n * sizeof(T));
    file.read((char*)c, n * sizeof(T));

    file.close();

    if (trunc) {
        size_t to_trunc = n * 3 * sizeof(T);

        file.open(path, std::ios::in | std::ios::out | std::ios::binary);
        if (file.is_open()) {
            std::vector<char> buffer(size - to_trunc);
            file.seekg(to_trunc, std::ios::beg);
            file.read(buffer.data(), buffer.size());
            file.seekg(0, std::ios::beg);
            file.write(buffer.data(), buffer.size());
            file.close();
        }

        if (size == to_trunc) {
            if (remove(path) != 0) {
                std::perror("Truncation failed");
            }
        } else if (size > n * 3 * sizeof(T)) {
            int res = truncate(path, size - to_trunc);
            if (res != 0)
                std::perror("Truncation failed");
        }
    }

    return !file.fail();
}

template <class T>
std::tuple<int, int> Utils::pad_zero(std::vector<T>& vec, const int& channels, const int& height,
                                     const int& width, const size_t& padding) {
    size_t new_h   = height + padding * 2;
    size_t new_w   = width + padding * 2;
    size_t new_dim = new_h * new_w;

    std::vector<T> old = vec;

    vec.clear();
    vec.resize(new_dim * channels, 0);

    size_t old_dim = width * height;
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                vec[c * new_dim + padding * new_w + h * new_w + padding + w]
                    = old[c * old_dim + h * width + w];
            }
        }
    }
    return std::make_tuple(new_h, new_w);
}

#endif // DEFS_HPP_
