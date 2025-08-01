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
#define PROTO 1 // 1 or 2

using Unit    = std::chrono::microseconds;
using measure = std::chrono::high_resolution_clock;

const int N_THREADS = std::max(1u, std::thread::hardware_concurrency());

constexpr size_t filter_prec = 0ULL;

constexpr seal::sec_level_type SEC_LEVEL = seal::sec_level_type::tc128;

constexpr uint64_t BIT_LEN   = 37;
constexpr uint64_t POLY_MOD  = 1ULL << 12;
constexpr uint64_t PLAIN_MOD = 1ULL << BIT_LEN;

constexpr uint64_t MOD         = PLAIN_MOD;
constexpr uint64_t moduloMask  = MOD - 1;
constexpr uint64_t moduloMidPt = MOD / 2;

namespace Utils {

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
    return bytes / static_cast<double>(1 << 20);
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

template <class Channel>
std::vector<Channel> init_ios(const char* addr, const int& port, const size_t& threads);

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

static inline uint64_t getRingElt(int64_t x) { return ((uint64_t)x) & moduloMask; }

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

std::vector<gemini::Tensor<uint64_t>> Utils::init_filter(const gemini::HomConv2DSS::Meta& meta,
                                                         const double& num) {
    std::vector<gemini::Tensor<uint64_t>> filters(meta.n_filters);

    for (auto& filter : filters) {
        gemini::Tensor<double> tmp(meta.fshape);
        // tmp.Randomize(0.3);

        for (int c = 0; c < tmp.channels(); ++c) {
            for (int i = 0; i < tmp.height(); ++i) {
                for (int j = 0; j < tmp.width(); ++j) {
                    tmp(c, i, j) = num;
                }
            }
        }
        filter = Utils::convert_fix_point(tmp);
    }

    return filters;
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

std::vector<gemini::HomFCSS::Meta> Utils::init_layers_fc() {
    std::vector<gemini::HomFCSS::Meta> layers;
    layers.push_back(Utils::init_meta_fc(1000, 2048));
    return layers;
}

void Utils::add_result(Result& res, const Result& res2) {
    res.encryption += res2.encryption;
    res.cipher_op += res2.cipher_op;
    res.plain_op += res2.plain_op;
    res.decryption += res2.decryption;
    res.send_recv += res2.send_recv;
    res.serial += res2.serial;
    res.bytes += res2.bytes;
}

template <class Channel>
std::vector<Channel> Utils::init_ios(const char* addr, const int& port, const size_t& threads) {
    std::vector<Channel> ioss;
    ioss.reserve(threads);
    for (size_t wid = 0; wid < threads; ++wid) {
        ioss.emplace_back(addr, port + wid, true);
    }
    return ioss;
}

void Utils::print_info(const gemini::HomConv2DSS::Meta& meta, const size_t& padding) {
    log(Level::DEBUG, "n_threads: ", N_THREADS);
    log(Level::DEBUG, "Padding: ", padding);
    log(Level::DEBUG, "Stride: ", meta.stride);
    log(Level::DEBUG, "i_channel: ", meta.ishape.channels());
    log(Level::DEBUG, "i_width: ", meta.ishape.width());
    log(Level::DEBUG, "i_height: ", meta.ishape.height());
    log(Level::DEBUG, "f_channel: ", meta.fshape.channels());
    log(Level::DEBUG, "f_width: ", meta.fshape.width());
    log(Level::DEBUG, "f_height: ", meta.fshape.height());
    log(Level::DEBUG, "n_filters: ", meta.n_filters);
}

gemini::HomBNSS::Meta Utils::init_meta_bn(const long& rows, const long& cols) {
    gemini::HomBNSS::Meta meta;
    long tmp             = sqrt(cols);
    meta.ishape          = {rows, tmp, tmp};
    meta.vec_shape       = {rows};
    meta.target_base_mod = PLAIN_MOD;
    meta.is_shared_input = true;
    return meta;
}

gemini::HomFCSS::Meta Utils::init_meta_fc(const long& common, const long& filter_h) {
    gemini::HomFCSS::Meta meta;

    meta.input_shape     = {common};
    meta.weight_shape    = {filter_h, common};
    meta.is_shared_input = true;

    return meta;
}

gemini::HomConv2DSS::Meta Utils::init_meta_conv(const long& ic, const long& ih, const long& iw,
                                                const long& fc, const long& fh, const long& fw,
                                                const size_t& n_filter, const size_t& stride,
                                                const size_t& padding) {
    gemini::HomConv2DSS::Meta meta;

    meta.ishape          = {ic, ih, iw};
    meta.fshape          = {fc, fh, fw};
    meta.is_shared_input = true;
    meta.n_filters       = n_filter;
    meta.padding         = padding == 0 ? gemini::Padding::VALID : gemini::Padding::SAME;
    meta.stride          = stride;

    return meta;
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

seal::SEALContext Utils::init_he_context() {
    seal::EncryptionParameters params(seal::scheme_type::bfv);
    params.set_poly_modulus_degree(POLY_MOD);
    params.set_n_special_primes(0);
    params.set_coeff_modulus(seal::CoeffModulus::Create(POLY_MOD, {60, 49}));
    params.set_plain_modulus(PLAIN_MOD);

    seal::SEALContext context(params, true, SEC_LEVEL);
    return context;
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

double Utils::convert(uint64_t v, int nbits) {
    int64_t sv = getSignedVal(getRingElt(v));
    return sv / (1. * std::pow(2, nbits));
}

gemini::Tensor<double> Utils::convert_double(const gemini::Tensor<uint64_t>& in) {
    gemini::Tensor<double> out(in.shape());
    out.tensor()
        = in.tensor().unaryExpr([&](uint64_t v) { return Utils::convert(v, filter_prec); });

    return out;
}

double Utils::print_results(const Result& res, const size_t& layer, const size_t& batchSize,
                            const size_t& threads, std::ostream& out) {
    std::string unit;
    auto data = to_MB(res.bytes, unit);
    if (!layer)
        out << "Encryption [ms],Cipher Calculations [s],Serialization [s],Decryption [ms],Plain "
               "Calculations [ms], "
               "Sending and Receiving [ms],Total [s],Bytes Send ["
            << unit << "],batchSize,threads\n";

    double total = to_msec(res.encryption) + to_msec(res.cipher_op) + to_msec(res.send_recv)
                   + to_msec(res.decryption) + to_msec(res.plain_op) + to_msec(res.serial);
    total /= 1'000.0;

    out << to_msec(res.encryption) << ", " << to_sec(res.cipher_op) << ", " << to_sec(res.serial)
        << ", " << to_msec(res.decryption) << ", " << to_msec(res.plain_op) << ", "
        << to_msec(res.send_recv) << ", " << total << ", " << data << ", " << batchSize << ", "
        << threads << "\n";

    return total;
}

Utils::Result Utils::average(const std::vector<Result>& res, bool average_bytes) {
    Result avg = {.encryption = 0,
                  .cipher_op  = 0,
                  .plain_op   = 0,
                  .decryption = 0,
                  .send_recv  = 0,
                  .serial     = 0,
                  .bytes      = 0,
                  .ret        = Code::OK};

    if (!res.size())
        return avg;

    size_t len = 0;
    for (size_t i = 0; i < res.size(); ++i) {
        auto& cur = res[i];
        if (!cur.bytes)
            continue;
        avg.send_recv += cur.send_recv;
        avg.encryption += cur.encryption;
        avg.decryption += cur.decryption;
        avg.cipher_op += cur.cipher_op;
        avg.plain_op += cur.plain_op;
        avg.serial += cur.serial;
        avg.bytes += cur.bytes;
        len++;
    }

    avg.send_recv /= len;
    avg.encryption /= len;
    avg.decryption /= len;
    avg.cipher_op /= len;
    avg.plain_op /= len;
    avg.serial /= len;

    if (average_bytes)
        avg.bytes /= len;

    return avg;
}

void Utils::make_csv(const std::vector<Result>& results, const size_t& batchSize,
                     const size_t& threads, const std::string& path) {
    std::ofstream os;

    if (!path.empty()) {
        log(Level::INFO, "writing results to: ", path);
        os.open(path, std::ios::out | std::ios::trunc);
    }

    double total = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        total += print_results(results[i], i, batchSize, threads, path.empty() ? std::cout : os);
    }

    os << "Total time[s]: " << total << "\n";
    double data = 0;
    std::string unit;
    for (auto& res : results) data += Utils::to_MB(res.bytes, unit);
    os << "Total data[" << unit << "]: " << data << "\n";

    if (os.is_open())
        os.close();
}

#endif // DEFS_HPP
