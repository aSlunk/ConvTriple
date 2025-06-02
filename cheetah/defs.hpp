#ifndef DEFS_HPP
#define DEFS_HPP

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

const int N_THREADS
    = std::thread::hardware_concurrency() == 0 ? 1 : std::thread::hardware_concurrency();
constexpr int PORT = 6969;

constexpr size_t filter_prec = 0ULL;

constexpr int64_t POLY_MOD  = 4096;
constexpr int64_t PLAIN_MOD = 1ULL << 50;

constexpr size_t iw = 224;
constexpr size_t ih = 224;
constexpr size_t ic = 3;

constexpr size_t fw       = 7;
constexpr size_t fh       = 7;
constexpr size_t fc       = 3;
constexpr size_t n_filter = 64;

constexpr size_t PADDING = 3;
constexpr size_t STRIDE  = 2;

constexpr gemini::Padding PAD = PADDING == 0 ? gemini::Padding::VALID : gemini::Padding::SAME;

constexpr uint64_t MOD         = PLAIN_MOD;
constexpr uint64_t moduloMask  = MOD - 1;
constexpr uint64_t moduloMidPt = MOD / 2;

const gemini::HomConv2DSS::Meta META = {
    .ishape          = {ic, ih, iw},
    .fshape          = {fc, fh, fw},
    .n_filters       = n_filter,
    .padding         = PADDING == 0 ? gemini::Padding::VALID : gemini::Padding::SAME,
    .stride          = STRIDE,
    .is_shared_input = false,
};

namespace Utils {

seal::SEALContext init_he_context();
void print_info();

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

void print_info() {
    std::cerr << "n_threads: " << N_THREADS << "\n";
    std::cerr << "Padding: " << PADDING << "\n";
    std::cerr << "Stride: " << META.stride << "\n";
    std::cerr << "i_channel: " << META.ishape.channels() << "\n";
    std::cerr << "i_width: " << META.ishape.width() << "\n";
    std::cerr << "i_height: " << META.ishape.height() << "\n";
    std::cerr << "f_channel: " << META.fshape.channels() << "\n";
    std::cerr << "f_width: " << META.fshape.width() << "\n";
    std::cerr << "f_height: " << META.fshape.height() << "\n";
    std::cerr << "n_filters: " << META.n_filters << "\n";
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
            std::cout << static_cast<T>(t(0, i, j)) << " ";
        }
        std::cout << "\n";
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

} // namespace Utils

#endif // DEFS_HPP