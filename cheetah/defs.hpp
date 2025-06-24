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
#define PROTO 2

using Unit    = std::chrono::microseconds;
using measure = std::chrono::high_resolution_clock;

const int N_THREADS = std::max(1u, std::thread::hardware_concurrency());

constexpr size_t filter_prec = 0ULL;

constexpr int64_t POLY_MOD  = 4096;
constexpr int64_t PLAIN_MOD = 1ULL << 50;

constexpr uint64_t MOD         = PLAIN_MOD;
constexpr uint64_t moduloMask  = MOD - 1;
constexpr uint64_t moduloMidPt = MOD / 2;

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

namespace Utils {

seal::SEALContext init_he_context();
void print_info();

gemini::HomConv2DSS::Meta init_meta(const long& ic, const long& ih, const long& iw, const long& fc,
                                    const long& fh, const long& fw, const size_t& n_filter,
                                    const size_t& stride, const size_t& padding);

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

gemini::HomConv2DSS::Meta init_meta(const long& ic, const long& ih, const long& iw, const long& fc,
                                    const long& fh, const long& fw, const size_t& n_filter,
                                    const size_t& stride, const size_t& padding) {
    gemini::HomConv2DSS::Meta meta;

    meta.ishape          = {ic, ih, iw};
    meta.fshape          = {fc, fh, fw};
    meta.is_shared_input = true;
    meta.n_filters       = n_filter;
    meta.padding         = padding == 0 ? gemini::Padding::VALID : gemini::Padding::SAME;
    meta.stride          = stride;

    return meta;
}

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
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 1, 1, 64, 1, 0));       // L1
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 3, 3, 64, 1, 1));       // L2
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 1, 1, 256, 1, 0));      // L3
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 1, 1, 256, 1, 0));      // L4
    layers.push_back(Utils::init_meta(256, 56, 56, 256, 1, 1, 64, 1, 0));     // L5
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 3, 3, 64, 1, 1));       // L6
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 1, 1, 256, 1, 0));      // L7
    layers.push_back(Utils::init_meta(256, 56, 56, 256, 1, 1, 64, 1, 0));     // L8
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 3, 3, 64, 1, 1));       // L9
    layers.push_back(Utils::init_meta(64, 56, 56, 64, 1, 1, 256, 1, 1));      // L10
    layers.push_back(Utils::init_meta(256, 56, 56, 256, 1, 1, 128, 1, 0));    // L11
    layers.push_back(Utils::init_meta(128, 56, 56, 128, 3, 3, 128, 2, 1));    // L12
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 1, 1, 512, 1, 0));    // L13
    layers.push_back(Utils::init_meta(256, 56, 56, 256, 1, 1, 512, 2, 0));    // L14
    layers.push_back(Utils::init_meta(512, 28, 28, 512, 1, 1, 128, 1, 0));    // L15
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 3, 3, 128, 1, 1));    // L16
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 1, 1, 512, 1, 0));    // L17
    layers.push_back(Utils::init_meta(512, 28, 28, 512, 1, 1, 128, 1, 0));    // L18
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 3, 3, 128, 1, 1));    // L19
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 1, 1, 512, 1, 0));    // L20
    layers.push_back(Utils::init_meta(512, 28, 28, 512, 1, 1, 128, 1, 0));    // L21
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 3, 3, 128, 1, 1));    // L22
    layers.push_back(Utils::init_meta(128, 28, 28, 128, 1, 1, 512, 1, 0));    // L23
    layers.push_back(Utils::init_meta(512, 28, 28, 512, 1, 1, 256, 1, 0));    // L24
    layers.push_back(Utils::init_meta(256, 28, 28, 256, 3, 3, 256, 2, 1));    // L25
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L26
    layers.push_back(Utils::init_meta(512, 28, 28, 512, 1, 1, 1024, 2, 0));   // L27
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 256, 1, 0));  // L28
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 3, 3, 256, 1, 1));    // L29
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L30
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 256, 1, 0));  // L31
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 3, 3, 256, 1, 1));    // L32
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L33
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 256, 1, 0));  // L34
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 3, 3, 256, 1, 1));    // L35
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L36
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 256, 1, 0));  // L37
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 3, 3, 256, 1, 1));    // L38
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L39
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 256, 1, 0));  // L40
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 3, 3, 256, 1, 1));    // L41
    layers.push_back(Utils::init_meta(256, 14, 14, 256, 1, 1, 1024, 1, 0));   // L42
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 512, 1, 0));  // L43
    layers.push_back(Utils::init_meta(512, 14, 14, 512, 3, 3, 512, 2, 1));    // L44
    layers.push_back(Utils::init_meta(512, 7, 7, 512, 1, 1, 2048, 1, 0));     // L45
    layers.push_back(Utils::init_meta(1024, 14, 14, 1024, 1, 1, 2048, 2, 0)); // L46
    layers.push_back(Utils::init_meta(2048, 7, 7, 2048, 1, 1, 512, 1, 0));    // L47
    layers.push_back(Utils::init_meta(512, 7, 7, 512, 3, 3, 512, 1, 1));      // L48
    layers.push_back(Utils::init_meta(512, 7, 7, 512, 1, 1, 2048, 1, 0));     // L49
    layers.push_back(Utils::init_meta(2048, 7, 7, 2048, 1, 1, 512, 1, 0));    // L50
    layers.push_back(Utils::init_meta(512, 7, 7, 512, 3, 3, 512, 1, 1));      // L51
    layers.push_back(Utils::init_meta(512, 7, 7, 512, 1, 1, 2048, 1, 0));     // L52
    return layers;
}

void add_result(Result& res, const Result& res2) {
    res.encryption += res2.encryption;
    res.cipher_op += res2.cipher_op;
    res.plain_op += res2.plain_op;
    res.decryption += res2.decryption;
    res.send_recv += res2.send_recv;
    res.serial += res2.serial;
    res.bytes += res2.bytes;
}

template <class Vec>
std::stringstream serialize(Vec& ct) {
    std::stringstream st;
    for (auto& ele : ct) {
        ele.save(st);
    }
    return st;
}

template <class Ctx>
void deserialize(const Ctx& context, std::stringstream& is, std::vector<seal::Ciphertext>& res) {
    for (auto& ele : res) {
        ele.load(context, is);
    }
    is.clear();
}

} // namespace Utils

double print_results(const Result& res, const int& layer, const size_t& batchSize,
                     const size_t& threads, std::ostream& out = std::cout) {
    if (!layer)
        out << "Encryption [ms],Cipher Calculations [s],Serialization [s],Decryption [ms],Plain "
               "Calculations [ms], "
               "Sending and Receiving [ms],Total [s],Bytes Send [MB],batchSize,threads\n";

    double total = res.encryption / 1'000.0 + res.cipher_op / 1'000.0 + res.send_recv / 1'000.0
                   + res.decryption / 1'000.0 + res.plain_op / 1'000.0 + res.serial / 1'000.0;
    total /= 1'000.0;

    out << res.encryption / 1'000.0 << ", " << res.cipher_op / 1'000'000.0 << ", "
        << res.serial / 1'000'000.0 << ", " << res.decryption / 1'000. << ", "
        << res.plain_op / 1'000.0 << ", " << res.send_recv / 1'000.0 << ", " << total << ", "
        << res.bytes / 1'000'000.0 << ", " << batchSize << ", " << threads << "\n";

    return total;
}

Result average(const std::vector<Result>& res) {
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
    avg.bytes /= len;

    return avg;
}


template <class EncVec>
Code send_recv(const seal::SEALContext& ctx, std::vector<IO::NetIO>& ios, EncVec& send, std::vector<seal::Ciphertext>& recv) {
    std::vector<std::vector<seal::Ciphertext>> result(ios.size(), std::vector<seal::Ciphertext>(0));

    auto program = [&](long wid, size_t start, size_t end) -> Code {
        auto& io = ios[wid];
        std::stringstream is;
        for (size_t cur = start; cur < end; ++cur) {
            send[cur].save(is);
        }

        IO::send_encrypted_vector(io, is, end - start);
        uint32_t ncts = IO::recv_encrypted_vector(io, is);
        result[wid].resize(ncts);
        Utils::deserialize(ctx, is, result[wid]);

        return Code::OK;
    };

    gemini::ThreadPool tpool(ios.size());
    gemini::LaunchWorks(tpool, send.size(), program);

    for (auto& vec : result) {
        if (!vec.size())
            continue;

        recv.reserve(recv.size() + vec.size());
        for (auto& ctx : vec)
            recv.push_back(ctx);
    }
    return Code::OK;
}

template <class Vec>
Code recv_send(const seal::SEALContext& ctx, std::vector<IO::NetIO>& ios, const Vec& send, std::vector<seal::Ciphertext>& recv) {
    std::vector<std::vector<seal::Ciphertext>> result(ios.size(), std::vector<seal::Ciphertext>(0));

    auto program = [&](long wid, size_t start, size_t end) -> Code {
        auto& io = ios[wid];
        std::stringstream is;
        for (size_t cur = start; cur < end; ++cur) {
            send[cur].save(is);
        }

        uint32_t ncts = IO::recv_encrypted_vector(io, is);
        result[wid].resize(ncts);
        IO::send_encrypted_vector(io, is, end - start);

        Utils::deserialize(ctx, is, result[wid]);

        return Code::OK;
    };

    gemini::ThreadPool tpool(ios.size());
    gemini::LaunchWorks(tpool, send.size(), program);

    for (auto& vec : result) {
        if (!vec.size())
            continue;

        recv.reserve(recv.size() + vec.size());
        for (auto& ctx : vec)
            recv.push_back(ctx);
    }
    return Code::OK;
}

#endif // DEFS_HPP