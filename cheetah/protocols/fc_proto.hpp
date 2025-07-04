#ifndef FC_PROTO_HPP
#define FC_PROTO_HPP

#include <gemini/cheetah/hom_fc_ss.h>
#include <gemini/cheetah/shape_inference.h>
#include <gemini/cheetah/tensor_encoder.h>
#include <gemini/cheetah/tensor_shape.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"

#ifndef VERIFY
#define VERIFY 1
#endif

using gemini::HomFCSS;
using gemini::Tensor;
using Utils::Result;

uint64_t add(const HomFCSS& conv, const uint64_t& a, const uint64_t& b) {
    uint64_t sum;
    seal::util::add_uint(&a, 1, b, &sum);
    return seal::util::barrett_reduce_64(sum, conv.plain_modulus());
}

namespace Server {

template <class Channel>
Result Protocol2(const HomFCSS::Meta& meta, Channel& server, const seal::SEALContext& context,
                 const HomFCSS& conv, const Tensor<uint64_t>& A1, Tensor<uint64_t>& C1,
                 const size_t& threads = 1);

template <class Channel>
Result Protocol1(const HomFCSS::Meta& meta, Channel& server, const seal::SEALContext& context,
                 const HomFCSS& conv, const Tensor<uint64_t>& A1, const Tensor<uint64_t>& B1,
                 Tensor<uint64_t>& C1, const size_t& threads = 1);

template <class Channel>
Result perform_proto(HomFCSS::Meta& meta, Channel& server, const seal::SEALContext& context,
                     const HomFCSS& hom_conv, const size_t& threads = 1);

#if VERIFY == 1
template <class T>
void Verify_Conv(IO::NetIO& io, const HomFCSS::Meta& meta, const HomFCSS& conv, const Tensor<T>& A1,
                 const Tensor<T>& B1, const Tensor<T>& C1);
#endif

} // namespace Server

namespace Client {

template <class Channel>
Result Protocol1(Channel& client, const seal::SEALContext& context, const HomFCSS& hom_conv,
                 const HomFCSS::Meta& meta, const Tensor<uint64_t>& A2, const Tensor<uint64_t>& B2,
                 Tensor<uint64_t>& C2, const size_t& threads = 1);

template <class Channel>
Result Protocol2(Channel& client, const seal::SEALContext& context, const HomFCSS& hom_conv,
                 const HomFCSS::Meta& meta, const Tensor<uint64_t>& A2, const Tensor<uint64_t>& B2,
                 Tensor<uint64_t>& C2, const size_t& threads = 1);

template <class Channel>
Result perform_proto(HomFCSS::Meta& meta, Channel& client, const seal::SEALContext& context,
                     const HomFCSS& hom_conv, const size_t& threads);

#if VERIFY == 1
template <class T>
void Verify_Conv(IO::NetIO& io, const Tensor<T>& A1, const Tensor<T>& B1, const Tensor<T>& C1);
#endif

} // namespace Client

template <class Channel>
Result Client::Protocol2(Channel& client, const seal::SEALContext& context, const HomFCSS& hom_fc,
                         const HomFCSS::Meta& meta, const Tensor<uint64_t>& A2,
                         const Tensor<uint64_t>& B2, Tensor<uint64_t>& C2, const size_t& threads) {
    Result measures;

    auto start = measure::now();

    std::vector<seal::Plaintext> enc_A2;
    measures.ret = hom_fc.encodeInputVector(A2, meta, enc_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Plaintext>> enc_B2;
    measures.ret = hom_fc.encodeWeightMatrix(B2, meta, enc_B2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // recv A' + deserialize
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    IO::recv_encrypted_vector(client, context, enc_A1);

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M2;
    measures.ret = hom_fc.MatVecMul(enc_A1, enc_A2, enc_B2, meta, M2, C2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // serialize + send M2'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    IO::send_encrypted_vector(client, M2);

    measures.send_recv += Utils::time_diff(start);

    for (auto& ele : client) measures.bytes += ele.counter;
    return measures;
}

template <class Channel>
Result Client::Protocol1(Channel& client, const seal::SEALContext& context, const HomFCSS& hom_conv,
                         const HomFCSS::Meta& meta, const Tensor<uint64_t>& A2,
                         const Tensor<uint64_t>& B2, Tensor<uint64_t>& C2, const size_t& threads) {
    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // Receive A1' and enc/send A2'
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Plaintext> encoded_A2;
    std::vector<seal::Serializable<seal::Ciphertext>> enc_A2;
    measures.ret = hom_conv.encryptInputVector(A2, meta, enc_A2, encoded_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Plaintext>> enc_B2;
    measures.ret = hom_conv.encodeWeightMatrix(B2, meta, enc_B2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);
    start               = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    measures.ret = IO::recv_send(context, client, enc_A2, enc_A1);
    if (measures.ret != Code::OK)
        return measures;

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M2;
    Tensor<uint64_t> R2;
    measures.ret = hom_conv.MatVecMul(enc_A1, encoded_A2, enc_B2, meta, enc_M2, R2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // send M2' + recv M1'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M1;
    measures.ret = IO::recv_send(context, client, enc_M2, enc_M1);
    if (measures.ret != Code::OK)
        return measures;

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // dec(M1') + R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    hom_conv.decryptToVector(enc_M1, meta, C2, threads);

    measures.decryption = Utils::time_diff(start);

    start = measure::now();

    Utils::op_inplace<uint64_t>(
        C2, R2, [&hom_conv](uint64_t a, uint64_t b) -> uint64_t { return add(hom_conv, a, b); });

    measures.plain_op = Utils::time_diff(start);

    for (auto& ele : client) measures.bytes += ele.counter;
    measures.ret = Code::OK;

    return measures;
}

template <class Channel>
Result Server::Protocol2(const HomFCSS::Meta& meta, Channel& server,
                         const seal::SEALContext& context, const HomFCSS& conv,
                         const Tensor<uint64_t>& A1, Tensor<uint64_t>& C1, const size_t& threads) {

    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // send Enc(A1)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    measures.ret = conv.encryptInputVector(A1, meta, enc_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    start = measure::now();

    IO::send_encrypted_vector(server, enc_A1);

    measures.send_recv = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // recv C1 = dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_C1;
    IO::recv_encrypted_vector(server, context, enc_C1);

    measures.send_recv += Utils::time_diff(start);
    start = measure::now();

    measures.ret        = conv.decryptToVector(enc_C1, meta, C1, threads);
    measures.decryption = Utils::time_diff(start);

    for (auto& ele : server) measures.bytes += ele.counter;
    return measures;
}

template <class Channel>
Result Server::Protocol1(const HomFCSS::Meta& meta, Channel& server,
                         const seal::SEALContext& context, const HomFCSS& conv,
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
    measures.ret = conv.encryptInputVector(A1, meta, enc_A1, encoded_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Plaintext>> enc_B1;
    measures.ret = conv.encodeWeightMatrix(B1, meta, enc_B1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    start = measure::now();

    std::vector<seal::Ciphertext> enc_A2;
    IO::send_recv(context, server, enc_A1, enc_A2);

    measures.send_recv = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M1 = (A1 + A2') ⊙ B1 - R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M1;
    Tensor<uint64_t> R1;
    measures.ret = conv.MatVecMul(enc_A2, encoded_A1, enc_B1, meta, M1, R1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // Send(M1), Recv(M2), Dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M2;
    IO::send_recv(context, server, M1, enc_M2);

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // Dec(M2) + R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    measures.ret = conv.decryptToVector(enc_M2, meta, C1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.decryption = Utils::time_diff(start);
    start               = measure::now();

    Utils::op_inplace<uint64_t>(C1, R1,
                                [&conv](uint64_t a, uint64_t b) { return add(conv, a, b); });

    measures.plain_op = Utils::time_diff(start);

    for (auto& ele : server) measures.bytes += ele.counter;
    measures.ret = Code::OK;
    return measures;
}

template <class Channel>
Result Server::perform_proto(HomFCSS::Meta& meta, Channel& server, const seal::SEALContext& context,
                             const HomFCSS& hom_conv, const size_t& threads) {
    Tensor<uint64_t> vec(meta.input_shape);
    for (long i = 0; i < vec.length(); i++) vec(i) = 2;
    Tensor<uint64_t> weight(meta.weight_shape);
    for (long i = 0; i < weight.rows(); i++)
        for (long j = 0; j < weight.cols(); j++) weight(i, j) = 5;

    Tensor<uint64_t> C1;

    server[0].sync();

#if PROTO == 1
    auto measures = Server::Protocol1(meta, server, context, hom_conv, vec, weight, C1, threads);
#else
    auto measures = Server::Protocol2(meta, server, context, hom_conv, vec, weight, threads);
#endif
    for (auto& ele : server) ele.counter = 0;

#if VERIFY == 1
    Verify_Conv(server[0], meta, hom_conv, vec, weight, C1);
#endif
    return measures;
}

template <class Channel>
Result Client::perform_proto(HomFCSS::Meta& meta, Channel& client, const seal::SEALContext& context,
                             const HomFCSS& hom_conv, const size_t& threads) {
    Tensor<uint64_t> vec(meta.input_shape);
    for (long i = 0; i < vec.length(); i++) vec(i) = 2;
    Tensor<uint64_t> weight(meta.weight_shape);
    for (long i = 0; i < weight.rows(); i++)
        for (long j = 0; j < weight.cols(); j++) weight(i, j) = 5;

    client[0].sync();

    Tensor<uint64_t> C2;

#if PROTO == 1
    auto measures = Client::Protocol1(client, context, hom_conv, meta, vec, weight, C2, threads);
#else
    auto measures = Client::Protocol2(client, context, hom_conv, meta, vec, weight, C2, threads);
#endif

    for (auto& ele : client) ele.counter = 0;

#if VERIFY == 1
    Verify_Conv(client[0], vec, weight, C2);
#endif
    return measures;
}

#if VERIFY == 1
template <class T>
void Server::Verify_Conv(IO::NetIO& io, const HomFCSS::Meta& meta, const HomFCSS& conv,
                         const Tensor<T>& A1, const Tensor<T>& B1, const Tensor<T>& C1) {
    Utils::log(Utils::Level::INFO, "VERIFYING FC");
    Tensor<T> A2(A1.shape());
    Tensor<T> B2(B1.shape());
    Tensor<T> C2(C1.shape());

    io.recv_data(A2.data(), A2.NumElements() * sizeof(T));
    io.recv_data(B2.data(), B2.NumElements() * sizeof(T));
    io.recv_data(C2.data(), C2.NumElements() * sizeof(T));

    Utils::op_inplace<T>(C2, C1, [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; }); // C
    Utils::op_inplace<T>(A2, A1, [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; }); // A1 + A2

#if PROTO == 1
    Utils::op_inplace<T>(B2, B1, [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; });
#endif

    Tensor<T> test;                              // (A1 + A2) (B1 + B2)
    conv.idealFunctionality(A2, B2, meta, test); // (A1 + A2) (B1 + B2)

    bool same = C2.shape() == test.shape();

    for (long i = 0; i < C2.length(); ++i) {
        if (!same || test(i) != C2(i)) {
            same = false;
            break;
        }
    }

    if (same)
        Utils::log(Utils::Level::PASSED, "FC: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "FC: FAILED");
}

template <class T>
void Client::Verify_Conv(IO::NetIO& io, const Tensor<T>& A2, const Tensor<T>& B2,
                         const Tensor<T>& C2) {
    log(Utils::Level::INFO, "SENDING");
    io.send_data(A2.data(), A2.NumElements() * sizeof(T));
    io.send_data(B2.data(), B2.NumElements() * sizeof(T));
    io.send_data(C2.data(), C2.NumElements() * sizeof(T));
    io.flush();
}
#endif

#endif