#ifndef CONV_PROTO_HPP
#define CONV_PROTO_HPP

#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/shape_inference.h>
#include <gemini/cheetah/tensor_encoder.h>
#include <gemini/cheetah/tensor_shape.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"

using gemini::Tensor;
using Utils::Result;

static uint64_t add(const gemini::HomConv2DSS& conv, const uint64_t& a, const uint64_t& b) {
    uint64_t sum;
    seal::util::add_uint(&a, 1, b, &sum);
    return seal::util::barrett_reduce_64(sum, conv.plain_modulus());
}

namespace Server {

template <class Channel>
Result Protocol2(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                 const seal::SEALContext& context, const gemini::HomConv2DSS& conv,
                 const Tensor<uint64_t>& A1, Tensor<uint64_t>& C1, const size_t& threads = 1);

template <class Channel>
Result Protocol1(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                 const seal::SEALContext& context, const gemini::HomConv2DSS& conv,
                 const Tensor<uint64_t>& A1, const std::vector<Tensor<uint64_t>>& B1,
                 Tensor<uint64_t>& C1, const size_t& threads = 1);

template <class Channel>
Result perform_proto(gemini::HomConv2DSS::Meta& meta, Channel** server,
                     const seal::SEALContext& context, const gemini::HomConv2DSS& hom_conv,
                     const size_t& threads = 1);

#ifdef VERIFY
template <class T>
void Verify_Conv(IO::NetIO& io, const gemini::HomConv2DSS::Meta& meta,
                 const gemini::HomConv2DSS& conv, const Tensor<T>& A1,
                 const std::vector<Tensor<T>>& B1, const Tensor<T>& C1);
#endif

} // namespace Server

namespace Client {

template <class Channel>
Result Protocol1(Channel** client, const seal::SEALContext& context,
                 const gemini::HomConv2DSS& hom_conv, const gemini::HomConv2DSS::Meta& meta,
                 const Tensor<uint64_t>& A2, const std::vector<Tensor<uint64_t>>& B2,
                 Tensor<uint64_t>& C2, const size_t& threads = 1);

template <class Channel>
Result Protocol2(Channel** client, const seal::SEALContext& context,
                 const gemini::HomConv2DSS& hom_conv, const gemini::HomConv2DSS::Meta& meta,
                 const Tensor<uint64_t>& A2, const std::vector<Tensor<uint64_t>>& B2,
                 Tensor<uint64_t>& C2, const size_t& threads = 1);

template <class Channel>
Result perform_proto(gemini::HomConv2DSS::Meta& meta, Channel** client,
                     const seal::SEALContext& context, const gemini::HomConv2DSS& hom_conv,
                     const size_t& threads);

#ifdef VERIFY
template <class T>
void Verify_Conv(IO::NetIO& io, const Tensor<T>& A1, const std::vector<Tensor<T>>& B1,
                 const Tensor<T>& C1);
#endif

} // namespace Client

template <class Channel>
Result Client::Protocol2(Channel** client, const seal::SEALContext& context,
                         const gemini::HomConv2DSS& hom_conv, const gemini::HomConv2DSS::Meta& meta,
                         const Tensor<uint64_t>& A2, const std::vector<Tensor<uint64_t>>& B2,
                         Tensor<uint64_t>& C2, const size_t& threads) {
    Result measures;

    auto start = measure::now();

    std::vector<seal::Plaintext> enc_A2;
    measures.ret = hom_conv.encodeImage(A2, meta, enc_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Plaintext>> enc_B2;
    measures.ret = hom_conv.encodeFilters(B2, meta, enc_B2, threads);
    if (measures.ret != Code::OK)
        return measures;

    hom_conv.filtersToNtt(enc_B2, threads);

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
    measures.ret = hom_conv.conv2DSS(enc_A1, enc_A2, enc_B2, meta, M2, C2, threads);
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
                         const gemini::HomConv2DSS& hom_conv, const gemini::HomConv2DSS::Meta& meta,
                         const Tensor<uint64_t>& A2, const std::vector<Tensor<uint64_t>>& B2,
                         Tensor<uint64_t>& C2, const size_t& threads) {
    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // Receive A1' and enc/send A2'
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Plaintext> encoded_A2;
    std::vector<seal::Serializable<seal::Ciphertext>> enc_A2;
    measures.ret = hom_conv.encryptImage(A2, meta, enc_A2, encoded_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Plaintext>> enc_B2;
    measures.ret = hom_conv.encodeFilters(B2, meta, enc_B2, threads);
    if (measures.ret != Code::OK)
        return measures;

    hom_conv.filtersToNtt(enc_B2, threads);

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
    measures.ret = hom_conv.conv2DSS(enc_A1, encoded_A2, enc_B2, meta, enc_M2, R2, threads);
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
Result Server::Protocol2(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                         const seal::SEALContext& context, const gemini::HomConv2DSS& conv,
                         const Tensor<uint64_t>& A1, Tensor<uint64_t>& C1, const size_t& threads) {

    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // send Enc(A1)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, threads);
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
Result Server::Protocol1(const gemini::HomConv2DSS::Meta& meta, Channel** server,
                         const seal::SEALContext& context, const gemini::HomConv2DSS& conv,
                         const Tensor<uint64_t>& A1, const std::vector<Tensor<uint64_t>>& B1,
                         Tensor<uint64_t>& C1, const size_t& threads) {
    Result measures;
    measures.send_recv = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Enc(A1), enc(B1), send(A1), recv(A2)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();
    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    std::vector<seal::Plaintext> encoded_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, encoded_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Plaintext>> enc_B1;
    measures.ret = conv.encodeFilters(B1, meta, enc_B1, threads);
    if (measures.ret != Code::OK)
        return measures;

    conv.filtersToNtt(enc_B1, threads);

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
    measures.ret = conv.conv2DSS(enc_A2, encoded_A1, enc_B1, meta, M1, R1, threads);
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
Result Server::perform_proto(gemini::HomConv2DSS::Meta& meta, Channel** server,
                             const seal::SEALContext& context, const gemini::HomConv2DSS& hom_conv,
                             const size_t& threads) {
    auto A1 = Utils::init_image(meta, 5);
    auto B1 = Utils::init_filter(meta, 2.0);

    Tensor<uint64_t> C1;

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
Result Client::perform_proto(gemini::HomConv2DSS::Meta& meta, Channel** client,
                             const seal::SEALContext& context, const gemini::HomConv2DSS& hom_conv,
                             const size_t& threads) {
    auto A2 = Utils::init_image(meta, 5);
    auto B2 = Utils::init_filter(meta, 2.0);

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
void Server::Verify_Conv(IO::NetIO& io, const gemini::HomConv2DSS::Meta& meta,
                         const gemini::HomConv2DSS& conv, const Tensor<T>& A1,
                         const std::vector<Tensor<T>>& B1, const Tensor<T>& C1) {
    Utils::log(Utils::Level::INFO, "VERIFYING CONV");
    Tensor<T> A2(A1.shape());
    std::vector<Tensor<T>> B2(meta.n_filters, Tensor<T>(meta.fshape));
    Tensor<T> C2(C1.shape());

    io.recv_data(A2.data(), A2.NumElements() * sizeof(T));
    for (auto& filter : B2) io.recv_data(filter.data(), filter.NumElements() * sizeof(T));
    io.recv_data(C2.data(), C2.NumElements() * sizeof(T));

    Utils::op_inplace<T>(C2, C1, [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; }); // C
    Utils::op_inplace<T>(A2, A1, [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; }); // A1 + A2

#if PROTO == 1
    for (size_t i = 0; i < B1.size(); ++i) // B1 + B2
        Utils::op_inplace<T>(B2[i], B1[i], [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; });
#endif

    Tensor<T> test;                              // (A1 + A2) (B1 + B2)
    conv.idealFunctionality(A2, B2, meta, test); // (A1 + A2) (B1 + B2)

    bool same = C2.shape() == test.shape();
    for (long c = 0; c < C2.channels(); ++c)
        for (long h = 0; h < C2.height(); ++h)
            for (long w = 0; w < C2.width(); ++w)
                if (!same || test(c, h, w) != C2(c, h, w)) {
                    Utils::log(Utils::Level::FAILED, test(c, h, w), ", ", C2(c, h, w));
                    same = false;
                    goto end;
                }
end:
    if (same)
        Utils::log(Utils::Level::PASSED, "CONV: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "CONV: FAILED");
}

template <class T>
void Client::Verify_Conv(IO::NetIO& io, const Tensor<T>& A2, const std::vector<Tensor<T>>& B2,
                         const Tensor<T>& C2) {
    log(Utils::Level::INFO, "SENDING");
    io.send_data(A2.data(), A2.NumElements() * sizeof(T), false);
    for (auto& filter : B2) io.send_data(filter.data(), filter.NumElements() * sizeof(T), false);
    io.send_data(C2.data(), C2.NumElements() * sizeof(T), false);
    io.flush();
}
#endif

#endif