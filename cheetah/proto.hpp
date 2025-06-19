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

namespace Server {

Result Protocol3(HomConv2DSS::Meta& meta, IO::NetIO& server, const seal::SEALContext& context,
                      const HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                      const size_t& threads = 1);

Result Protocol2(HomConv2DSS::Meta& meta, IO::NetIO& server, const seal::SEALContext& context,
                      const HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                      const std::vector<Tensor<uint64_t>>& B1, const size_t& threads = 1);

}

namespace Client {

Result Protocol2(IO::NetIO& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                 HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<Tensor<uint64_t>>& B2, const size_t& threads = 1);
Result Protocol3(IO::NetIO& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                 HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<Tensor<uint64_t>>& B2, const size_t& threads = 1);
}

Result Client::Protocol3(IO::NetIO& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                 HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<Tensor<uint64_t>>& B2, const size_t& threads) {
    meta.is_shared_input = true;
    Result measures;
    measures.send_recv  = 0;
    measures.decryption = 0;
    measures.plain_op   = 0;
    measures.serial     = 0;

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
    // recv + deserialize
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::stringstream is;
    std::vector<seal::Ciphertext> enc_A1(IO::recv_encrypted_vector(client, is));

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start = measure::now();

    Utils::deserialize(context, is, enc_A1);

    measures.serial += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start = measure::now();

    // hom_conv.add_plain_inplace(enc_A1, enc_A2, N_THREADS);
    std::vector<seal::Ciphertext> M2;
    Tensor<uint64_t> R;
    measures.ret = hom_conv.conv2DSS(enc_A1, enc_A2, enc_B2, meta, M2, R, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // serialize + send
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    HomConv2DSS::serialize(M2, is, threads);

    measures.serial += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start = measure::now();

    IO::send_encrypted_vector(client, is, M2.size());

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    measures.bytes = client.counter;
    return measures;
}

Result Client::Protocol2(IO::NetIO& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                 HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<Tensor<uint64_t>>& B2, const size_t& threads) {
    meta.is_shared_input = true;
    Result measures;
    measures.send_recv = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Receive enc(A1) and enc/send A2
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

    start = measure::now();

    // auto ser = Utils::serialize(enc_A2);
    std::stringstream ser;
    HomConv2DSS::serialize(enc_A2, ser, threads);

    measures.serial = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    std::stringstream is;
    start = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    enc_A1.resize(IO::recv_encrypted_vector(client, is));
    IO::send_encrypted_vector(client, ser, enc_A2.size());

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start = measure::now();

    Utils::deserialize(context, is, enc_A1);

    measures.serial += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // (A1' + A2) ⊙ B2 - R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    // hom_conv.add_plain_inplace(enc_A1, encoded_A2);
    std::vector<seal::Ciphertext> enc_M2;
    Tensor<uint64_t> R2;
    measures.ret = hom_conv.conv2DSS(enc_A1, encoded_A2, enc_B2, meta, enc_M2, R2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // Send result
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    // ser = Utils::serialize(enc_M2);
    HomConv2DSS::serialize(enc_M2, ser, threads);

    measures.serial += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M1;
    enc_M1.resize(IO::recv_encrypted_vector(client, is));
    IO::send_encrypted_vector(client, ser, enc_M2.size());

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start = measure::now();

    Utils::deserialize(context, is, enc_M1);

    measures.serial += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // A2 ⊙ B2 + M1 - R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    Tensor<uint64_t> M1;
    hom_conv.decryptToTensor(enc_M1, meta, M1, threads);

    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    Utils::op_inplace<uint64_t>(M1, R2, [](uint64_t a, uint64_t b) -> uint64_t { return a + b; });

    measures.plain_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    measures.bytes = client.counter;
    measures.ret   = Code::OK;
    return measures;
}

Result Server::Protocol3(HomConv2DSS::Meta& meta, IO::NetIO& server, const seal::SEALContext& context,
                      const HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                      const size_t& threads) {

    meta.is_shared_input = true;
    Result measures;
    measures.plain_op  = 0;
    measures.cipher_op = 0;

    auto start = measure::now();

    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    std::stringstream is;
    HomConv2DSS::serialize(enc_A1, is, threads);

    measures.serial = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start           = measure::now();

    IO::send_encrypted_vector(server, is, enc_A1.size());

    measures.send_recv = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    std::vector<seal::Ciphertext> enc_C1(IO::recv_encrypted_vector(server, is));
    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    Utils::deserialize(context, is, enc_C1);

    measures.serial += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();
    Tensor<uint64_t> C1;
    measures.ret        = conv.decryptToTensor(enc_C1, meta, C1, threads);
    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    std::cerr << C1.channels() << " x " << C1.height() << " x " << C1.width() << "\n";

    measures.bytes = server.counter;
    return measures;
}

Result Server::Protocol2(HomConv2DSS::Meta& meta, IO::NetIO& server, const seal::SEALContext& context,
                      const HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                      const std::vector<Tensor<uint64_t>>& B1, const size_t& threads) {
    meta.is_shared_input = true;

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

    // auto ser = Utils::serialize(enc_A1);
    std::stringstream ser;
    HomConv2DSS::serialize(enc_A1, ser, threads);

    measures.serial = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    std::stringstream is;
    start = measure::now();

    IO::send_encrypted_vector(server, ser, enc_A1.size());
    std::vector<seal::Ciphertext> enc_A2;
    enc_A2.resize(IO::recv_encrypted_vector(server, is));

    measures.send_recv = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start              = measure::now();

    Utils::deserialize(context, is, enc_A2);

    measures.serial += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // M1 = (A1 + A2') ⊙ B1 - R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    // conv.add_plain_inplace(enc_A2, encoded_A1);
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

    // ser = Utils::serialize(M1);
    HomConv2DSS::serialize(M1, ser, threads);

    measures.serial += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start = measure::now();

    IO::send_encrypted_vector(server, ser, M1.size());
    std::vector<seal::Ciphertext> enc_M2;
    enc_M2.resize(IO::recv_encrypted_vector(server, is));

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    start = measure::now();

    Utils::deserialize(context, is, enc_M2);

    measures.serial += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // Dec(M2) + R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    Tensor<uint64_t> M2;
    measures.ret = conv.decryptToTensor(enc_M2, meta, M2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    Utils::op_inplace<uint64_t>(M2, R1, [](uint64_t a, uint64_t b) { return a - b; });

    measures.plain_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    std::cerr << M2.channels() << " x " << M2.height() << " x " << M2.width() << "\n";

    measures.bytes = server.counter;
    measures.ret   = Code::OK;
    return measures;
}

#endif