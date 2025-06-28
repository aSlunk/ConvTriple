#ifndef PROTO_HPP
#define PROTO_HPP

#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/shape_inference.h>
#include <gemini/cheetah/tensor_encoder.h>
#include <gemini/cheetah/tensor_shape.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"

using namespace gemini;

uint64_t add(const HomConv2DSS& conv, const uint64_t& a, const uint64_t& b) {
    uint64_t sum;
    seal::util::add_uint(&a, 1, b, &sum);
    return seal::util::barrett_reduce_64(sum, conv.plain_modulus());
}

namespace Server {

template <class Channel>
Result Protocol3(const HomConv2DSS::Meta& meta, Channel& server, const seal::SEALContext& context,
                 const HomConv2DSS& conv, const Tensor<uint64_t>& A1, const size_t& threads = 1);

template <class Channel>
Result Protocol2(const HomConv2DSS::Meta& meta, Channel& server, const seal::SEALContext& context,
                 const HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                 const std::vector<Tensor<uint64_t>>& B1, const size_t& threads = 1);

template <class Channel>
Result perform_proto(HomConv2DSS::Meta& meta, Channel& server, const seal::SEALContext& context,
                     const HomConv2DSS& hom_conv, const size_t& threads = 1);

#if VERIFY == 1
template <class T>
void Verify_Conv(IO::NetIO& io, const HomConv2DSS::Meta& meta, const HomConv2DSS& conv,
                 const Tensor<T>& A1, const std::vector<Tensor<T>>& B1, const Tensor<T>& C1);
#endif

} // namespace Server

namespace Client {

template <class Channel>
Result Protocol2(Channel& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                 const HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<Tensor<uint64_t>>& B2, const size_t& threads = 1);

template <class Channel>
Result Protocol3(Channel& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                 const HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<Tensor<uint64_t>>& B2, const size_t& threads = 1);

template <class Channel>
Result perform_proto(HomConv2DSS::Meta& meta, Channel& client, const seal::SEALContext& context,
                     const HomConv2DSS& hom_conv, const size_t& threads);

#if VERIFY == 1
template <class T>
void Verify_Conv(IO::NetIO& io, const Tensor<T>& A1, const std::vector<Tensor<T>>& B1,
                 const Tensor<T>& C1);
#endif

} // namespace Client

template <class Channel>
Result Client::Protocol3(Channel& client, const seal::SEALContext& context,
                         const HomConv2DSS& hom_conv, const HomConv2DSS::Meta& meta,
                         const Tensor<uint64_t>& A2, const std::vector<Tensor<uint64_t>>& B2,
                         const size_t& threads) {
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

    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // recv A' + deserialize
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    IO::recv_encrypted_vector(client, context, enc_A1);

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M2;
    Tensor<uint64_t> R;
    measures.ret = hom_conv.conv2DSS(enc_A1, enc_A2, enc_B2, meta, M2, R, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // serialize + send M2'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    IO::send_encrypted_vector(client, M2);

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    for (auto& ele : client) measures.bytes += ele.counter;

#if VERIFY == 1
    Verify_Conv(client[0], A2, B2, R);
#endif
    return measures;
}

template <class Channel>
Result Client::Protocol2(Channel& client, const seal::SEALContext& context,
                         const HomConv2DSS& hom_conv, const HomConv2DSS::Meta& meta,
                         const Tensor<uint64_t>& A2, const std::vector<Tensor<uint64_t>>& B2,
                         const size_t& threads) {
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

    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start               = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    measures.ret = recv_send(context, client, enc_A2, enc_A1);
    if (measures.ret != Code::OK)
        return measures;

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M2;
    Tensor<uint64_t> R2;
    measures.ret = hom_conv.conv2DSS(enc_A1, encoded_A2, enc_B2, meta, enc_M2, R2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // send M2' + recv M1'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M1;
    measures.ret = recv_send(context, client, enc_M2, enc_M1);
    if (measures.ret != Code::OK)
        return measures;

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // dec(M1') + R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    Tensor<uint64_t> M1;
    hom_conv.decryptToTensor(enc_M1, meta, M1, threads);

    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    Utils::op_inplace<uint64_t>(
        M1, R2, [&hom_conv](uint64_t a, uint64_t b) -> uint64_t { return add(hom_conv, a, b); });

    measures.plain_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    for (auto& ele : client) measures.bytes += ele.counter;
    measures.ret = Code::OK;

#if VERIFY == 1
    Verify_Conv(client[0], A2, B2, M1);
#endif
    return measures;
}

template <class Channel>
Result Server::Protocol3(const HomConv2DSS::Meta& meta, Channel& server,
                         const seal::SEALContext& context, const HomConv2DSS& conv,
                         const Tensor<uint64_t>& A1, const size_t& threads) {

    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // send Enc(A1)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    IO::send_encrypted_vector(server, enc_A1);

    measures.send_recv = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // recv C1 = dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_C1;
    IO::recv_encrypted_vector(server, context, enc_C1);

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start = measure::now();

    Tensor<uint64_t> C1;
    measures.ret        = conv.decryptToTensor(enc_C1, meta, C1, threads);
    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    std::cerr << C1.channels() << " x " << C1.height() << " x " << C1.width() << "\n";

    for (auto& ele : server) measures.bytes += ele.counter;
#if VERIFY == 1
    Verify_Conv(server[0], meta, conv, A1, std::vector<Tensor<uint64_t>>(0), C1);
#endif
    return measures;
}

template <class Channel>
Result Server::Protocol2(const HomConv2DSS::Meta& meta, Channel& server,
                         const seal::SEALContext& context, const HomConv2DSS& conv,
                         const Tensor<uint64_t>& A1, const std::vector<Tensor<uint64_t>>& B1,
                         const size_t& threads) {
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

    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    std::vector<seal::Ciphertext> enc_A2;
    send_recv(context, server, enc_A1, enc_A2);

    measures.send_recv = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // M1 = (A1 + A2') ⊙ B1 - R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> M1;
    Tensor<uint64_t> R1;
    measures.ret = conv.conv2DSS(enc_A2, encoded_A1, enc_B1, meta, M1, R1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // Send(M1), Recv(M2), Dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M2;
    send_recv(context, server, M1, enc_M2);

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // Dec(M2) + R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    Tensor<uint64_t> M2;
    measures.ret = conv.decryptToTensor(enc_M2, meta, M2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start               = measure::now();

    Utils::op_inplace<uint64_t>(M2, R1,
                                [&conv](uint64_t a, uint64_t b) { return add(conv, a, b); });

    measures.plain_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    std::cerr << M2.channels() << " x " << M2.height() << " x " << M2.width() << "\n";

    for (auto& ele : server) measures.bytes += ele.counter;
    measures.ret = Code::OK;
#if VERIFY == 1
    Verify_Conv(server[0], meta, conv, A1, B1, M2);
#endif
    return measures;
}

template <class Channel>
Result Server::perform_proto(HomConv2DSS::Meta& meta, Channel& server,
                             const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                             const size_t& threads) {
    auto A1 = Utils::init_image(meta, 5);
    auto B1 = Utils::init_filter(meta, 2.0);

    server[0].sync();

#if PROTO == 2
    auto measures = Server::Protocol2(meta, server, context, hom_conv, A1, B1, threads);
#elif PROTO == 3
    auto measures = Server::Protocol3(meta, server, context, hom_conv, A1, threads);
#endif
    for (auto& ele : server) ele.counter = 0;
    return measures;
}

template <class Channel>
Result Client::perform_proto(HomConv2DSS::Meta& meta, Channel& client,
                             const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                             const size_t& threads) {
    auto A2 = Utils::init_image(meta, 5);
    auto B2 = Utils::init_filter(meta, 2.0);

    client[0].sync();

#if PROTO == 3
    auto measures = Client::Protocol3(client, context, hom_conv, meta, A2, B2, threads);
#elif PROTO == 2
    auto measures = Client::Protocol2(client, context, hom_conv, meta, A2, B2, threads);
#endif

    for (auto& ele : client) ele.counter = 0;
    return measures;
}

#if VERIFY == 1
template <class T>
void Server::Verify_Conv(IO::NetIO& io, const HomConv2DSS::Meta& meta, const HomConv2DSS& conv,
                         const Tensor<T>& A1, const std::vector<Tensor<T>>& B1,
                         const Tensor<T>& C1) {
    std::cerr << "VERIFYING\n";
    Tensor<T> A2(A1.shape());
    std::vector<Tensor<T>> B2(meta.n_filters, Tensor<T>(meta.fshape));
    Tensor<T> C2(C1.shape());

    io.recv_data(A2.data(), A2.NumElements() * sizeof(T));
    for (auto& filter : B2) io.recv_data(filter.data(), filter.NumElements() * sizeof(T));
    io.recv_data(C2.data(), C2.NumElements() * sizeof(T));

    Utils::op_inplace<T>(C2, C1, [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; }); // C
    Utils::op_inplace<T>(A2, A1, [&conv](T a, T b) -> T { return (a + b) % PLAIN_MOD; }); // A1 + A2

#if PROTO == 2
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
                    same = false;
                    goto end;
                }
end:
    if (same)
        std::cerr << GREEN << "PASSED\n" << NC;
    else
        std::cerr << RED << "FAILED\n" << NC;
    std::cerr << "FINISHED VERIFYING\n";
}

template <class T>
void Client::Verify_Conv(IO::NetIO& io, const Tensor<T>& A2, const std::vector<Tensor<T>>& B2,
                         const Tensor<T>& C2) {
    std::cerr << "SENDING\n";
    io.send_data(A2.data(), A2.NumElements() * sizeof(T));
    for (auto& filter : B2) io.send_data(filter.data(), filter.NumElements() * sizeof(T));
    io.send_data(C2.data(), C2.NumElements() * sizeof(T));
    io.flush();
}
#endif

#endif