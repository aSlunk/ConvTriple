#ifndef DEFS_HPP
#define DEFS_HPP

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>

#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/tensor.h>
#include <seal/seal.h>

#define EXEC_FAILED -1

using Unit    = std::chrono::microseconds;
using measure = std::chrono::high_resolution_clock;

const int N_THREADS = std::max(1u, std::thread::hardware_concurrency());
constexpr int PORT  = 6969;

constexpr size_t filter_prec = 0ULL;

constexpr int64_t POLY_MOD  = 4096;
constexpr int64_t PLAIN_MOD = 1ULL << 50;

constexpr uint64_t MOD         = PLAIN_MOD;
constexpr uint64_t moduloMask  = MOD - 1;
constexpr uint64_t moduloMidPt = MOD / 2;

namespace Utils {

seal::SEALContext init_he_context();
void print_info();

gemini::HomConv2DSS::Meta init_meta(const long& ic, const long& ih, const long& iw,
    const long& fc, const long& fh, const long& fw, const size_t& n_filter, const size_t& stride, const size_t& padding) {
    gemini::HomConv2DSS::Meta meta;

    meta.ishape = {ic, ih, iw};
    meta.fshape = {fc, fh, fw};
    meta.is_shared_input = false;
    meta.n_filters = n_filter;
    meta.padding = padding == 0 ? gemini::Padding::VALID : gemini::Padding::SAME;
    meta.stride = stride;

    return meta;
}

template <class T>
gemini::Tensor<uint64_t> convert_fix_point(const gemini::Tensor<T>& in);

template <class T>
void print_tensor(const gemini::Tensor<T>& t);

double convert(uint64_t v, int nbits);
gemini::Tensor<double> convert_double(const gemini::Tensor<uint64_t>& in);
gemini::Tensor<uint64_t> init_image(const gemini::HomConv2DSS::Meta& meta, const double& num);
std::vector<gemini::Tensor<uint64_t>> init_filter(const gemini::HomConv2DSS::Meta& meta,
                                                  const double& num);

template <class T>
void op_inplace(gemini::Tensor<T>& A, const gemini::Tensor<T>& B, std::function<T(T, T)> op);

seal::SEALContext init_he_context() {
    seal::EncryptionParameters params(seal::scheme_type::bfv);
    params.set_poly_modulus_degree(POLY_MOD);
    params.set_coeff_modulus(seal::CoeffModulus::BFVDefault(POLY_MOD));
    params.set_plain_modulus(PLAIN_MOD);

    seal::SEALContext context(params);

    return context;
}

void print_info(const gemini::HomConv2DSS::Meta& meta, const size_t& padding) {
    std::cerr << "n_threads: " << N_THREADS << "\n";
    std::cerr << "Padding: " << padding << "\n";
    std::cerr << "Stride: " << meta.stride << "\n";
    std::cerr << "i_channel: " << meta.ishape.channels() << "\n";
    std::cerr << "i_width: " << meta.ishape.width() << "\n";
    std::cerr << "i_height: " << meta.ishape.height() << "\n";
    std::cerr << "f_channel: " << meta.fshape.channels() << "\n";
    std::cerr << "f_width: " << meta.fshape.width() << "\n";
    std::cerr << "f_height: " << meta.fshape.height() << "\n";
    std::cerr << "n_filters: " << meta.n_filters << "\n";
}

static inline uint64_t getRingElt(int64_t x) { return ((uint64_t)x) & moduloMask; }

static inline int64_t getSignedVal(uint64_t x) {
    assert(x < MOD);
    if (x > moduloMidPt)
        return static_cast<int64_t>(x - MOD);
    else
        return static_cast<int64_t>(x);
}

template <class T>
gemini::Tensor<uint64_t> convert_fix_point(const gemini::Tensor<T>& in) {
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
void print_tensor(const gemini::Tensor<T>& t) {
    for (long i = 0; i < t.height(); ++i) {
        for (long j = 0; j < t.width(); ++j) {
            std::cerr << static_cast<T>(t(0, i, j)) << " ";
        }
        std::cerr << "\n";
    }
}

double convert(uint64_t v, int nbits) {
    int64_t sv = getSignedVal(getRingElt(v));
    // return sv / (1. * std::pow(2, nbits));
    return sv >> filter_prec;
}

gemini::Tensor<double> convert_double(const gemini::Tensor<uint64_t>& in) {
    gemini::Tensor<double> out(in.shape());
    out.tensor()
        = in.tensor().unaryExpr([&](uint64_t v) { return Utils::convert(v, filter_prec); });

    return out;
}

gemini::Tensor<uint64_t> init_image(const gemini::HomConv2DSS::Meta& meta, const double& num) {
    gemini::Tensor<uint64_t> image(meta.ishape);

    for (int c = 0; c < image.channels(); ++c) {
        for (int i = 0; i < image.height(); ++i) {
            for (int j = 0; j < image.width(); ++j) {
                image(c, i, j) = num;
            }
        }
    }

    return image;
}

std::vector<gemini::Tensor<uint64_t>> init_filter(const gemini::HomConv2DSS::Meta& meta,
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
void op_inplace(gemini::Tensor<T>& A, const gemini::Tensor<T>& B, std::function<T(T, T)> op) {
    assert(A.channels() == B.channels());
    assert(A.height() == B.height());
    assert(A.width() == B.width());

    for (int i = 0; i < A.channels(); ++i) {
        for (int j = 0; j < A.width(); ++j) {
            for (int k = 0; k < A.height(); ++k) {
                A(i, j, k) = op(A(i, j, k), B(i, j, k));
            }
        }
    }
}

std::vector<gemini::HomConv2DSS::Meta> init_layers() {
    std::vector<gemini::HomConv2DSS::Meta> layers;
    layers.push_back(Utils::init_meta(3, 224, 224, 3, 7, 7, 64, 2, 3));
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 1, 1, 64, 1, 0)); // L1
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 3, 3, 64, 1, 1)); // L2
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 1, 1, 256, 1, 0)); // L3
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 1, 1, 256, 1, 0)); // L4
    layers.push_back(Utils::init_meta(256, 56, 56, 256, 1, 1, 64, 1, 0)); // L5
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 3, 3, 64, 1, 1)); // L6
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 1, 1, 256, 1, 0)); // L7
    layers.push_back(Utils::init_meta(256, 56, 56, 256, 1, 1, 64, 1, 0)); // L8
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 3, 3, 64, 1, 1)); // L9
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 1, 1, 256, 1, 1)); // L10
    layers.push_back(Utils::init_meta(256, 56, 56, 256, 1, 1, 128, 1, 0)); // L11
    layers.push_back(Utils::init_meta(128, 56, 56, 128, 3, 3, 128, 2, 1)); // L12
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 1, 1, 512, 1, 0)); // L13
    layers.push_back(Utils::init_meta(256, 56, 56, 256, 1, 1, 512, 2, 0)); // L14
    layers.push_back(Utils::init_meta(512, 28, 28, 512, 1, 1, 128, 1, 0)); // L15
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 3, 3, 128, 1, 1)); // L16
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 1, 1, 512, 1, 0)); // L17
    layers.push_back(Utils::init_meta(512, 28, 28, 512, 1, 1, 128, 1, 0)); // L18
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 3, 3, 128, 1, 1)); // L19
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 1, 1, 512, 1, 0)); // L20
    layers.push_back(Utils::init_meta(512, 28, 28, 512, 1, 1, 128, 1, 0)); // L21
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 3, 3, 128, 1, 1)); // L22
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 1, 1, 512, 1, 0)); // L23
    layers.push_back(Utils::init_meta(512, 28, 28, 512, 1, 1, 256, 1, 0)); // L24
    layers.push_back(Utils::init_meta(256, 28, 28, 256, 3, 3, 256, 2, 1)); // L25
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0)); // L26
    layers.push_back(Utils::init_meta(512, 28, 28, 512, 1, 1, 1024, 2, 0)); // L27
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 256, 1, 0)); // L28
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 3, 3, 256, 1, 1)); // L29
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0)); // L30
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 256, 1, 0)); // L31
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 3, 3, 256, 1, 1)); // L32
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0)); // L33
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 256, 1, 0)); // L34
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 3, 3, 256, 1, 1)); // L35
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0)); // L36
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 256, 1, 0)); // L37
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 3, 3, 256, 1, 1)); // L38
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0)); // L39
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 256, 1, 0)); // L40
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 3, 3, 256, 1, 1)); // L41
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0)); // L42
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 512, 1, 0)); // L43
    layers.push_back(Utils::init_meta(512, 14, 14, 512, 3, 3, 512, 2, 1)); // L44
    layers.push_back(Utils::init_meta(512, 7, 7, 512, 1, 1, 2048, 1, 0)); // L45
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 2048, 2, 0)); // L46
    layers.push_back(Utils::init_meta(2048, 7, 7, 2048, 1, 1, 512, 1, 0)); // L47
    layers.push_back(Utils::init_meta(512, 7, 7, 512, 3, 3, 512, 1, 1)); // L48
    layers.push_back(Utils::init_meta(512, 7, 7, 512, 1, 1, 2048, 1, 0)); // L49
    layers.push_back(Utils::init_meta(2048, 7, 7, 2048, 1, 1, 512, 1, 0)); // L50
    layers.push_back(Utils::init_meta(512, 7, 7, 512, 3, 3, 512, 1, 1)); // L51
    layers.push_back(Utils::init_meta(512, 7, 7, 512, 1, 1, 2048, 1, 0)); // L52
    return layers;
}

} // namespace Utils

#endif // DEFS_HPP