#ifndef BN_PROTO_HPP_
#define BN_PROTO_HPP_

#include <gemini/cheetah/hom_bn_ss.h>
#include <gemini/cheetah/shape_inference.h>
#include <gemini/cheetah/tensor_encoder.h>
#include <gemini/cheetah/tensor_shape.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "core/defs.hpp"

using gemini::Tensor;
using Utils::Result;

namespace Server {

template <class Channel>
Result Protocol2_alt(const gemini::HomBNSS::Meta& meta, Channel** server, const gemini::HomBNSS& bn,
                     const Tensor<uint64_t>& A1, Tensor<uint64_t>& C1, const size_t& threads = 1);

template <class Channel>
Result Protocol1_alt(const gemini::HomBNSS::Meta& meta, Channel** server, const gemini::HomBNSS& bn,
                     const Tensor<uint64_t>& A1, const Tensor<uint64_t>& B1, Tensor<uint64_t>& C1,
                     const size_t& threads = 1);

template <class Channel>
Result perform_proto(Channel** ios, const gemini::HomBNSS& bn, const gemini::HomBNSS::Meta& meta,
                     const Tensor<uint64_t>& A, const Tensor<uint64_t>& B, Tensor<uint64_t>& C,
                     const size_t& threads, Utils::PROTO proto = Utils::PROTO::AB);

template <class Channel>
Result perform_elem(Channel** ios, const gemini::HomBNSS& bn, const gemini::HomBNSS::Meta& meta,
                    const Tensor<uint64_t>& A, const Tensor<uint64_t>& B, Tensor<uint64_t>& C,
                    const size_t& threads, Utils::PROTO proto = Utils::PROTO::AB);

#ifdef VERIFY
template <class T>
void Verify_BN(IO::NetIO* io, const gemini::HomBNSS::Meta& meta, const gemini::HomBNSS& bn,
               const Tensor<T>& A1, const Tensor<T>& B1, const Tensor<T>& C1, Utils::PROTO proto);
#endif
} // namespace Server

namespace Client {

template <class Channel>
Result Protocol1_alt(Channel** client, const gemini::HomBNSS& bn, const gemini::HomBNSS::Meta& meta,
                     const Tensor<uint64_t>& A2, const Tensor<uint64_t>& B2, Tensor<uint64_t>& C2,
                     const size_t& threads = 1);

template <class Channel>
Result Protocol2_alt(Channel** client, const gemini::HomBNSS& bn, const gemini::HomBNSS::Meta& meta,
                     const Tensor<uint64_t>& A2, const Tensor<uint64_t>& B2, Tensor<uint64_t>& C2,
                     const size_t& threads = 1);

template <class Channel>
Result perform_proto(Channel** ios, const gemini::HomBNSS& bn, const gemini::HomBNSS::Meta& meta,
                     const Tensor<uint64_t>& A, const Tensor<uint64_t>& B, Tensor<uint64_t>& C,
                     const size_t& threads, Utils::PROTO proto = Utils::PROTO::AB);

template <class Channel>
Result perform_elem(Channel** ios, const gemini::HomBNSS& bn, const gemini::HomBNSS::Meta& meta,
                    const Tensor<uint64_t>& A, const Tensor<uint64_t>& B, Tensor<uint64_t>& C,
                    const size_t& threads, Utils::PROTO proto = Utils::PROTO::AB);

#ifdef VERIFY
template <class T>
void Verify_BN(IO::NetIO* io, const Tensor<T>& A1, const Tensor<T>& B1, const Tensor<T>& C1);
#endif
} // namespace Client

template <class Channel>
Result Client::Protocol2_alt(Channel** client, const gemini::HomBNSS& bn,
                             const gemini::HomBNSS::Meta& meta, const Tensor<uint64_t>& A2,
                             const Tensor<uint64_t>& B2, Tensor<uint64_t>& C2,
                             const size_t& threads) {
    Result measures;

    auto start = measure::now();

    std::vector<seal::Plaintext> encoded_A2;
    measures.ret = bn.encodeVector(A2, meta, encoded_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<seal::Plaintext> encoded_B2;
    measures.ret = bn.encodeScales(B2, meta, encoded_B2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // recv A' + deserialize
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    bn.recvEncryptVector(client[0], enc_A1, meta);

    measures.send_recv += Utils::time_diff(start);

    if (enc_A1.size() != encoded_B2.size()) {
        measures.ret = Code::ERR_DIM_MISMATCH;
        return measures;
    }
    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M2;
    measures.ret = bn.bn(enc_A1, encoded_A2, encoded_B2, meta, M2, C2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // serialize + send M2'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    measures.ret = bn.sendEncryptVector(client[0], M2, meta);
    if (measures.ret != Code::OK)
        return measures;

    measures.send_recv += Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += client[i]->counter;
    return measures;
}

template <class Channel>
Result Client::Protocol1_alt(Channel** client, const gemini::HomBNSS& bn,
                             const gemini::HomBNSS::Meta& meta, const Tensor<uint64_t>& A2,
                             const Tensor<uint64_t>& B2, Tensor<uint64_t>& C2,
                             const size_t& threads) {
    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // Receive A1' and enc/send A2'
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Plaintext> encoded_A2;
    std::vector<seal::Serializable<seal::Ciphertext>> enc_A2;
    measures.ret = bn.encryptVector(A2, meta, enc_A2, encoded_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<seal::Plaintext> encoded_B2;
    measures.ret = bn.encodeScales(B2, meta, encoded_B2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);
    start               = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    // measures.ret = IO::recv_send(context, client, enc_A2, enc_A1, threads);
    bn.recvEncryptVector(client[0], enc_A1, meta);
    bn.sendEncryptVector(client[0], enc_A2, meta);

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M2;
    Tensor<uint64_t> R2(meta.vec_shape);
    measures.ret = bn.bn(enc_A1, encoded_A2, encoded_B2, meta, enc_M2, R2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // send M2' + recv M1'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M1;
    bn.recvEncryptVector(client[0], enc_M1, meta);
    bn.sendEncryptVector(client[0], enc_M2, meta);

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // dec(M1') + R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    bn.decryptToVector(enc_M1, meta, C2, threads);

    measures.decryption = Utils::time_diff(start);

    start = measure::now();

    Utils::op_inplace<uint64_t>(
        C2, R2, [](uint64_t a, uint64_t b) -> uint64_t { return Utils::add(a, b); });

    measures.plain_op = Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += client[i]->counter;
    measures.ret = Code::OK;

    return measures;
}

template <class Channel>
Result Server::Protocol2_alt(const gemini::HomBNSS::Meta& meta, Channel** server,
                             const gemini::HomBNSS& bn, const Tensor<uint64_t>& A1,
                             Tensor<uint64_t>& C1, const size_t& threads) {

    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // send Enc(A1)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    measures.ret = bn.encryptVector(A1, meta, enc_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    start = measure::now();

    bn.sendEncryptVector(server[0], enc_A1, meta);

    measures.send_recv = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // recv C1 = dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_C1;
    measures.ret = bn.recvEncryptVector(server[0], enc_C1, meta);
    if (measures.ret != Code::OK)
        return measures;

    measures.send_recv += Utils::time_diff(start);
    start = measure::now();

    measures.ret        = bn.decryptToVector(enc_C1, meta, C1, threads);
    measures.decryption = Utils::time_diff(start);

    // Utils::log(Utils::Level::DEBUG, C1.channels(), " x ", C1.height(), " x ", C1.width());
    // Utils::log(Utils::Level::DEBUG, C1.NumElements());

    for (size_t i = 0; i < threads; ++i) measures.bytes += server[i]->counter;
    return measures;
}

template <class Channel>
Result Server::Protocol1_alt(const gemini::HomBNSS::Meta& meta, Channel** server,
                             const gemini::HomBNSS& bn, const Tensor<uint64_t>& A1,
                             const Tensor<uint64_t>& B1, Tensor<uint64_t>& C1,
                             const size_t& threads) {
    Result measures;
    measures.send_recv = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Enc(A1), enc(B1), send(A1), recv(A2)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();
    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    std::vector<seal::Plaintext> encoded_A1;
    measures.ret = bn.encryptVector(A1, meta, enc_A1, encoded_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<seal::Plaintext> encoded_B1;
    measures.ret = bn.encodeScales(B1, meta, encoded_B1, threads);

    measures.encryption = Utils::time_diff(start);

    start = measure::now();

    std::vector<seal::Ciphertext> enc_A2;
    bn.sendEncryptVector(server[0], enc_A1, meta);
    bn.recvEncryptVector(server[0], enc_A2, meta);

    measures.send_recv = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M1 = (A1 + A2') ⊙ B1 - R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M1;
    Tensor<uint64_t> R1(meta.vec_shape);
    measures.ret = bn.bn(enc_A2, encoded_A1, encoded_B1, meta, M1, R1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // Send(M1), Recv(M2), Dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M2;
    bn.sendEncryptVector(server[0], M1, meta);
    bn.recvEncryptVector(server[0], enc_M2, meta);

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // Dec(M2) + R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    measures.ret = bn.decryptToVector(enc_M2, meta, C1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.decryption = Utils::time_diff(start);
    start               = measure::now();

    Utils::op_inplace<uint64_t>(C1, R1, [](uint64_t a, uint64_t b) { return Utils::add(a, b); });

    measures.plain_op = Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += server[i]->counter;
    measures.ret = Code::OK;
    return measures;
}

namespace {

void pack(const Tensor<uint64_t>& mat, const Tensor<uint64_t>& scales,
          Tensor<uint64_t>& flattend_mat, Tensor<uint64_t>& flattened_scales) {
    assert(mat.channels() == scales.NumElements() && "Dimension missmatch");
    flattend_mat.Reshape({mat.NumElements()});
    flattened_scales.Reshape({mat.NumElements()});
    // flattend_mat = gemini::Tensor<uint64_t>::Wrap(mat.data(), {mat.NumElements()});

    const auto& C = mat.channels();
    const auto& H = mat.height();
    const auto& W = mat.width();

    for (long c = 0; c < C; ++c) {
        for (long h = 0; h < H; ++h) {
            for (long w = 0; w < W; ++w) {
                // size_t index            = h * W * C + w * C + c;
                size_t index            = c * H * W + h * W + w;
                flattend_mat(index)     = mat(c, h, w);
                flattened_scales(index) = scales(c);
            }
        }
    }
}

} // namespace

template <class Channel>
Result Server::perform_proto(Channel** ios, const gemini::HomBNSS& bn,
                             const gemini::HomBNSS::Meta& meta, const Tensor<uint64_t>& A,
                             const Tensor<uint64_t>& B, Tensor<uint64_t>& C, const size_t& threads,
                             Utils::PROTO proto) {
    Tensor<uint64_t> vec, scales;
    pack(A, B, vec, scales);

    C.Reshape(meta.ishape);

    auto func = [&](size_t wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;

        Result res;
        gemini::HomBNSS::Meta tmp;
        tmp.is_shared_input = meta.is_shared_input;
        tmp.target_base_mod = meta.target_base_mod;
        tmp.vec_shape       = {end - start};

        Tensor<uint64_t> tmp_A = Tensor<uint64_t>::Wrap(vec.data() + start, tmp.vec_shape);
        Tensor<uint64_t> tmp_B = Tensor<uint64_t>::Wrap(scales.data() + start, tmp.vec_shape);
        Tensor<uint64_t> tmp_C = Tensor<uint64_t>::Wrap(C.data() + start, tmp.vec_shape);

        switch (proto) {
        case Utils::PROTO::AB:
            res = Server::Protocol1_alt(tmp, ios + wid, bn, tmp_A, tmp_B, tmp_C, 1);
            break;
        case Utils::PROTO::AB2:
            res = Server::Protocol2_alt(tmp, ios + wid, bn, tmp_A, tmp_C, 1);
            break;
        }
        return res.ret;
    };

    Result res;
    gemini::ThreadPool tpool(threads);
    res.ret = gemini::LaunchWorks(tpool, vec.NumElements(), func);

    return res;
}

template <class Channel>
Result Server::perform_elem(Channel** ios, const gemini::HomBNSS& bn,
                            const gemini::HomBNSS::Meta& meta, const Tensor<uint64_t>& A,
                            const Tensor<uint64_t>& B, Tensor<uint64_t>& C, const size_t& threads,
                            Utils::PROTO proto) {
    Result result;

    switch (proto) {
    case Utils::PROTO::AB:
        result = Protocol1_alt(meta, ios, bn, A, B, C, threads);
        break;
    case Utils::PROTO::AB2:
        result = Protocol2_alt(meta, ios, bn, A, C, threads);
        break;
    }

    if (result.ret != Code::OK)
        return result;

#ifdef VERIFY
    Verify_BN(ios[0], meta, bn, A, B, C, proto);
#endif

    return result;
}

template <class Channel>
Result Client::perform_proto(Channel** ios, const gemini::HomBNSS& bn,
                             const gemini::HomBNSS::Meta& meta, const Tensor<uint64_t>& A,
                             const Tensor<uint64_t>& B, Tensor<uint64_t>& C, const size_t& threads,
                             Utils::PROTO proto) {
#ifndef NDEBUG
    Utils::log(Utils::Level::DEBUG, "Using alternative BN");
#endif
    Tensor<uint64_t> vec, scales;
    pack(A, B, vec, scales);

    C.Reshape(meta.ishape);

    auto func = [&](size_t wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;

        Result res;
        gemini::HomBNSS::Meta tmp;
        tmp.is_shared_input = meta.is_shared_input;
        tmp.target_base_mod = meta.target_base_mod;
        tmp.vec_shape       = {end - start};

        Tensor<uint64_t> tmp_A = Tensor<uint64_t>::Wrap(vec.data() + start, tmp.vec_shape);
        Tensor<uint64_t> tmp_B = Tensor<uint64_t>::Wrap(scales.data() + start, tmp.vec_shape);
        Tensor<uint64_t> tmp_C = Tensor<uint64_t>::Wrap(C.data() + start, tmp.vec_shape);

        switch (proto) {
        case Utils::PROTO::AB:
            res = Client::Protocol1_alt(ios + wid, bn, tmp, tmp_A, tmp_B, tmp_C, 1);
            break;
        case Utils::PROTO::AB2:
            res = Client::Protocol2_alt(ios + wid, bn, tmp, tmp_A, tmp_B, tmp_C, 1);
            break;
        }

        return res.ret;
    };

    Result res;

    gemini::ThreadPool tpool(threads);
    res.ret = gemini::LaunchWorks(tpool, vec.NumElements(), func);

    return res;
}

template <class Channel>
Result Client::perform_elem(Channel** ios, const gemini::HomBNSS& bn,
                            const gemini::HomBNSS::Meta& meta, const Tensor<uint64_t>& A,
                            const Tensor<uint64_t>& B, Tensor<uint64_t>& C, const size_t& threads,
                            Utils::PROTO proto) {
    Result result;

    switch (proto) {
    case Utils::PROTO::AB:
        result = Client::Protocol1_alt(ios, bn, meta, A, B, C, threads);
        break;
    case Utils::PROTO::AB2:
        result = Client::Protocol2_alt(ios, bn, meta, A, B, C, threads);
        break;
    }

    if (result.ret != Code::OK)
        return result;

#ifdef VERIFY
    Verify_BN(ios[0], A, B, C);
#endif

    return result;
}

#ifdef VERIFY
template <class T>
void Server::Verify_BN(IO::NetIO* io, const gemini::HomBNSS::Meta& meta, const gemini::HomBNSS& bn,
                       const Tensor<T>& A1, const Tensor<T>& B1, const Tensor<T>& C1,
                       Utils::PROTO proto) {
    Utils::log(Utils::Level::DEBUG, "VERIFYING Elem. Mult with ",
               proto == Utils::PROTO::AB ? "AB" : "AB2");
    Utils::log(Utils::Level::DEBUG, "num_elements: ", meta.vec_shape.num_elements());

    Tensor<T> A2(A1.shape());
    Tensor<T> B2(meta.vec_shape);
    Tensor<T> C2(C1.shape());

    io->recv_data(A2.data(), A2.NumElements() * sizeof(T));
    io->recv_data(B2.data(), B2.NumElements() * sizeof(T));
    io->recv_data(C2.data(), C2.NumElements() * sizeof(T));

    Utils::op_inplace<T>(C2, C1, [&bn](T a, T b) -> T { return Utils::add(a, b); }); // C
    Utils::op_inplace<T>(A2, A1, [&bn](T a, T b) -> T { return Utils::add(a, b); }); // A1 + A2

    if (proto == Utils::PROTO::AB)
        Utils::op_inplace<T>(B2, B1, [&bn](T a, T b) -> T { return Utils::add(a, b); });

    Tensor<T> test;                            // (A1 + A2) (B1 + B2)
    bn.idealFunctionality(A2, B2, meta, test); // (A1 + A2) (B1 + B2)

    // Utils::print_tensor(A2);
    // Utils::print_tensor(B2);
    // Utils::print_tensor(C1);

    bool same = C2.shape() == test.shape();
    for (long w = 0; w < C2.NumElements(); ++w)
        if (!same || test(w) != C2(w)) {
            Utils::log(Utils::Level::FAILED, w, ": ", test(w), ", ", C2(w));
            same = false;
            goto end;
        }
end:
    if (same)
        Utils::log(Utils::Level::PASSED, "Elem. Mult: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "Elem. Mult: FAILED");
}
template <class T>
void Client::Verify_BN(IO::NetIO* io, const Tensor<T>& A2, const Tensor<T>& B2,
                       const Tensor<T>& C2) {
    log(Utils::Level::INFO, "SENDING");
    io->send_data(A2.data(), A2.NumElements() * sizeof(T), false);
    io->send_data(B2.data(), B2.NumElements() * sizeof(T), false);
    io->send_data(C2.data(), C2.NumElements() * sizeof(T), false);
    io->flush();
}
#endif

#endif