#ifndef BN_DIRECT_PROTO_HPP
#define BN_DIRECT_PROTO_HPP

#include <gemini/cheetah/hom_bn_ss.h>
#include <gemini/cheetah/shape_inference.h>
#include <gemini/cheetah/tensor_encoder.h>
#include <gemini/cheetah/tensor_shape.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"
#include "protocols/bn_proto.hpp"

using gemini::Tensor;
using Utils::Result;

namespace Server {

template <class Channel>
Result Protocol2(const gemini::HomBNSS::Meta& meta, Channel** server,
                 const seal::SEALContext& context, const gemini::HomBNSS& conv,
                 const Tensor<uint64_t>& A1, Tensor<uint64_t>& C1, const size_t& threads = 1);

template <class Channel>
Result Protocol1(const gemini::HomBNSS::Meta& meta, Channel** server,
                 const seal::SEALContext& context, const gemini::HomBNSS& conv,
                 const Tensor<uint64_t>& A1, const Tensor<uint64_t>& B1, Tensor<uint64_t>& C1,
                 const size_t& threads = 1);

template <class Channel>
Result perform_proto(gemini::HomBNSS::Meta& meta, Channel** server,
                     const seal::SEALContext& context, const gemini::HomBNSS& hom_conv,
                     const size_t& threads = 1);

#ifdef VERIFY
template <class T>
void Verify_Conv(IO::NetIO& io, const gemini::HomBNSS::Meta& meta, const gemini::HomBNSS& conv,
                 const Tensor<T>& A1, const Tensor<T>& B1, const Tensor<T>& C1);
#endif

} // namespace Server

namespace Client {

template <class Channel>
Result Protocol1(Channel** client, const seal::SEALContext& context,
                 const gemini::HomBNSS& hom_conv, const gemini::HomBNSS::Meta& meta,
                 const Tensor<uint64_t>& A2, const Tensor<uint64_t>& B2, Tensor<uint64_t>& C2,
                 const size_t& threads = 1);

template <class Channel>
Result Protocol2(Channel** client, const seal::SEALContext& context,
                 const gemini::HomBNSS& hom_conv, const gemini::HomBNSS::Meta& meta,
                 const Tensor<uint64_t>& A2, const Tensor<uint64_t>& B2, Tensor<uint64_t>& C2,
                 const size_t& threads = 1);

template <class Channel>
Result perform_proto(gemini::HomBNSS::Meta& meta, Channel** client,
                     const seal::SEALContext& context, const gemini::HomBNSS& hom_conv,
                     const size_t& threads);

#ifdef VERIFY
// template <class T>
// void Verify_Conv(IO::NetIO& io, const Tensor<T>& A1, const Tensor<T>& B1,
//                  const Tensor<T>& C1);
#endif

} // namespace Client

template <class Channel>
Result Client::Protocol2(Channel** client, const seal::SEALContext& context,
                         const gemini::HomBNSS& hom_conv, const gemini::HomBNSS::Meta& meta,
                         const Tensor<uint64_t>& A2, const Tensor<uint64_t>& B2,
                         Tensor<uint64_t>& C2, const size_t& threads) {
    Result measures;

    auto start = measure::now();

    std::vector<seal::Plaintext> enc_A2;
    measures.ret = hom_conv.encodeTensor(A2, meta, enc_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // recv A' + deserialize
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    IO::recv_encrypted_vector(client, context, enc_A1, threads);

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M2;
    measures.ret = hom_conv.bn_direct(enc_A1, enc_A2, B2, meta, M2, C2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // serialize + send M2'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    IO::send_encrypted_vector(client, M2, threads);

    measures.send_recv += Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += client[i]->counter;
    return measures;
}

template <class Channel>
Result Client::Protocol1(Channel** client, const seal::SEALContext& context,
                         const gemini::HomBNSS& hom_conv, const gemini::HomBNSS::Meta& meta,
                         const Tensor<uint64_t>& A2, const Tensor<uint64_t>& B2,
                         Tensor<uint64_t>& C2, const size_t& threads) {
    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // Receive A1' and enc/send A2'
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Plaintext> encoded_A2;
    std::vector<seal::Serializable<seal::Ciphertext>> enc_A2;
    measures.ret = hom_conv.encryptTensor(A2, meta, enc_A2, encoded_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);
    start               = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    measures.ret = IO::recv_send(context, client, enc_A2, enc_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M2;
    Tensor<uint64_t> R2;
    measures.ret = hom_conv.bn_direct(enc_A1, encoded_A2, B2, meta, enc_M2, R2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // send M2' + recv M1'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M1;
    measures.ret = IO::recv_send(context, client, enc_M2, enc_M1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // dec(M1') + R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    hom_conv.decryptToTensor(enc_M1, meta, C2, threads);

    measures.decryption = Utils::time_diff(start);

    start = measure::now();

    Utils::op_inplace<uint64_t>(
        C2, R2, [&hom_conv](uint64_t a, uint64_t b) -> uint64_t { return add(hom_conv, a, b); });

    measures.plain_op = Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += client[i]->counter;
    measures.ret = Code::OK;

    return measures;
}

template <class Channel>
Result Server::Protocol2(const gemini::HomBNSS::Meta& meta, Channel** server,
                         const seal::SEALContext& context, const gemini::HomBNSS& conv,
                         const Tensor<uint64_t>& A1, Tensor<uint64_t>& C1, const size_t& threads) {

    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // send Enc(A1)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    measures.ret = conv.encryptTensor(A1, meta, enc_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    start = measure::now();

    IO::send_encrypted_vector(server, enc_A1, threads);

    measures.send_recv = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // recv C1 = dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_C1;
    IO::recv_encrypted_vector(server, context, enc_C1, threads);

    measures.send_recv += Utils::time_diff(start);
    start = measure::now();

    measures.ret        = conv.decryptToTensor(enc_C1, meta, C1, threads);
    measures.decryption = Utils::time_diff(start);

    Utils::log(Utils::Level::DEBUG, C1.channels(), " x ", C1.height(), " x ", C1.width());

    for (size_t i = 0; i < threads; ++i) measures.bytes += server[i]->counter;
    return measures;
}

template <class Channel>
Result Server::Protocol1(const gemini::HomBNSS::Meta& meta, Channel** server,
                         const seal::SEALContext& context, const gemini::HomBNSS& conv,
                         const Tensor<uint64_t>& A1, const Tensor<uint64_t>& B1,
                         Tensor<uint64_t>& C1, const size_t& threads) {
    Result measures;
    measures.send_recv = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Enc(A1), enc(B1), send(A1), recv(A2)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();
    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    std::vector<seal::Plaintext> encoded_A1;
    measures.ret = conv.encryptTensor(A1, meta, enc_A1, encoded_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    start = measure::now();

    std::vector<seal::Ciphertext> enc_A2;
    IO::send_recv(context, server, enc_A1, enc_A2, threads);

    measures.send_recv = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M1 = (A1 + A2') ⊙ B1 - R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M1;
    Tensor<uint64_t> R1;
    measures.ret = conv.bn_direct(enc_A2, encoded_A1, B1, meta, M1, R1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // Send(M1), Recv(M2), Dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M2;
    IO::send_recv(context, server, M1, enc_M2, threads);

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // Dec(M2) + R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    measures.ret = conv.decryptToTensor(enc_M2, meta, C1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.decryption = Utils::time_diff(start);
    start               = measure::now();

    Utils::op_inplace<uint64_t>(C1, R1,
                                [&conv](uint64_t a, uint64_t b) { return add(conv, a, b); });

    measures.plain_op = Utils::time_diff(start);

    Utils::log(Utils::Level::DEBUG, C1.channels(), " x ", C1.height(), " x ", C1.width());

    for (size_t i = 0; i < threads; ++i) measures.bytes += server[i]->counter;
    measures.ret = Code::OK;
    return measures;
}

template <class Channel>
Result Server::perform_proto(gemini::HomBNSS::Meta& meta, Channel** server,
                             const seal::SEALContext& context, const gemini::HomBNSS& hom_conv,
                             const size_t& threads) {
    auto A1 = Utils::init_image(meta, 5);
    auto B1 = gemini::Tensor(meta.vec_shape);

    for (long i = 0; i < B1.NumElements(); ++i) {
        B1(i) = i + 1;
    }

    Tensor<uint64_t> C1;

    // auto s2 = meta.ishape.height();
    // auto s3 = meta.ishape.width();
    // auto s4 = meta.ishape.channels();
    // size_t n_ct_coeff_packing = ((s2 * s3 + POLY_MOD - 1) / POLY_MOD) * s4;
    // size_t n_ct_bfv_packing = ((s2 * s3 * s4 + POLY_MOD - 1) / POLY_MOD) * 3;
    // if (n_ct_coeff_packing >= n_ct_bfv_packing)
    //     return Server::perform_proto(server, context, hom_conv, meta, A1, B1, threads);

    server[0]->sync();

#if PROTO == 1
    auto measures = Server::Protocol1(meta, server, context, hom_conv, A1, B1, C1, threads);
#else
    auto measures = Server::Protocol2(meta, server, context, hom_conv, A1, C1, threads);
#endif
    for (size_t i = 0; i < threads; ++i) server[i]->counter = 0;

#ifdef VERIFY
    Verify_Conv(*(server[0]), meta, hom_conv, A1, B1, C1);
#endif
    return measures;
}

template <class Channel>
Result Client::perform_proto(gemini::HomBNSS::Meta& meta, Channel** client,
                             const seal::SEALContext& context, const gemini::HomBNSS& hom_conv,
                             const size_t& threads) {
    Tensor<uint64_t> A2 = Utils::init_image(meta, 5);
    Tensor<uint64_t> B2 = gemini::Tensor(meta.vec_shape);

    for (long i = 0; i < B2.NumElements(); ++i) {
        B2(i) = i + 1;
    }

    // auto s2 = meta.ishape.height();
    // auto s3 = meta.ishape.width();
    // auto s4 = meta.ishape.channels();
    // size_t n_ct_coeff_packing = ((s2 * s3 + POLY_MOD - 1) / POLY_MOD) * s4;
    // size_t n_ct_bfv_packing = ((s2 * s3 * s4 + POLY_MOD - 1) / POLY_MOD) * 3;
    // if (n_ct_coeff_packing >= n_ct_bfv_packing)
    //     return Client::perform_proto(client, context, hom_conv, meta, A2, B2, threads);

    client[0]->sync();

    Tensor<uint64_t> C2;

#if PROTO == 1
    auto measures = Client::Protocol1(client, context, hom_conv, meta, A2, B2, C2, threads);
#else
    auto measures = Client::Protocol2(client, context, hom_conv, meta, A2, B2, C2, threads);
#endif

    for (size_t i = 0; i < threads; ++i) client[i]->counter = 0;

#ifdef VERIFY
    Verify_Conv(*(client[0]), A2, B2, C2);
#endif
    return measures;
}

#ifdef VERIFY
template <class T>
void Server::Verify_Conv(IO::NetIO& io, const gemini::HomBNSS::Meta& meta,
                         const gemini::HomBNSS& conv, const Tensor<T>& A1, const Tensor<T>& B1,
                         const Tensor<T>& C1) {
    Utils::log(Utils::Level::INFO, "VERIFYING BN");
    Tensor<T> A2(A1.shape());
    Tensor<T> B2(meta.vec_shape);
    Tensor<T> C2(C1.shape());

    io.recv_data(A2.data(), A2.NumElements() * sizeof(T));
    io.recv_data(B2.data(), B2.NumElements() * sizeof(T));
    io.recv_data(C2.data(), C2.NumElements() * sizeof(T));

    Utils::op_inplace<T>(C2, C1, [&conv](T a, T b) -> T { return add(conv, a, b); }); // C
    Utils::op_inplace<T>(A2, A1, [&conv](T a, T b) -> T { return add(conv, a, b); }); // A1 + A2

#if PROTO == 1
    Utils::op_inplace<T>(B2, B1, [&conv](T a, T b) -> T { return add(conv, a, b); });
#endif

    Tensor<T> test;                              // (A1 + A2) (B1 + B2)
    conv.idealFunctionality(A2, B2, meta, test); // (A1 + A2) (B1 + B2)

    bool same = C2.shape() == test.shape();
    for (long c = 0; c < C2.channels(); ++c)
        for (long h = 0; h < C2.height(); ++h)
            for (long w = 0; w < C2.width(); ++w)
                if (!same || test(c, h, w) != C2(c, h, w)) {
                    Utils::log(Utils::Level::FAILED, c, ", ", h, ", ", w, ": ", test(c, h, w), ", ",
                               C2(c, h, w));
                    same = false;
                    goto end;
                }
end:
    if (same)
        Utils::log(Utils::Level::PASSED, "BN: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "BN: FAILED");
}

// template <class T>
// void Client::Verify_Conv(IO::NetIO& io, const Tensor<T>& A2, const Tensor<T>& B2,
//                          const Tensor<T>& C2) {
//     log(Utils::Level::INFO, "SENDING");
//     io.send_data(A2.data(), A2.NumElements() * sizeof(T), false);
//     io.send_data(B2.data(), B2.NumElements() * sizeof(T), false);
//     io.send_data(C2.data(), C2.NumElements() * sizeof(T), false);
//     io.flush();
// }
#endif

#endif