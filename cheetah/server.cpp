#include <iostream>
#include <memory>
#include <seal/seal.h>
#include <seal/util/uintcore.h>
#include <vector>

#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/shape_inference.h>
#include <gemini/cheetah/tensor_encoder.h>
#include <gemini/cheetah/tensor_shape.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"

using namespace gemini;
namespace {

struct Result {
    size_t encryption;
    size_t cipher_op;
    size_t plain_op;
    size_t decryption;
    size_t sending_recv;
    size_t bytes;
    Code ret;
};

double print_results(const Result& res, const int& layer = 0) {
    if (!layer)
        std::cout << "Encryption [ms],Cipher Calculations [s],Decryption [ms],Plain Calculations "
                     "[ms],Sending and Receiving[s],Total [s],Bytes Send [MB]\n";

    double total = res.encryption / 1'000.0 + res.cipher_op / 1'000.0 + res.decryption / 1'000.0
                   + res.plain_op / 1'000.0 + res.sending_recv / 1'000.0;
    total /= 1000.0;
    std::cout << res.encryption / 1'000.0 << ", " << res.cipher_op / 1'000'000. << ", "
              << res.decryption / 1'000. << ", " << res.plain_op / 1'000.0 << ", "
              << res.sending_recv / 1'000'000.0 << ", " << total << ", " << res.bytes / 1'000'000.0
              << "\n";

    return total;
}

Result conv2d
    [[maybe_unused]] (const IO::NetIO& io, const HomConv2DSS& conv, const HomConv2DSS::Meta& META,
                      const Tensor<uint64_t>& image, const std::vector<Tensor<uint64_t>>& filters) {
    Result measures;

    auto start = measure::now();
    std::vector<seal::Ciphertext> ctxts;
    measures.ret        = conv.encryptImage(image, META, ctxts);
    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    if (measures.ret != Code::OK)
        return measures;

    std::stringstream stream;
    for (auto& cipher : ctxts) {
        cipher.save(stream);
    }

    measures.bytes = stream.str().length();
    std::cerr << ctxts.size() << "\n";
    std::cerr << "send bytes: " << stream.str().length() << "\n";

    std::vector<std::vector<seal::Plaintext>> p_filter;
    measures.ret = conv.encodeFilters(filters, META, p_filter);
    if (measures.ret != Code::OK)
        return measures;

    start = measure::now();
    std::vector<seal::Ciphertext> result;
    Tensor<uint64_t> out;
    measures.ret
        = conv.conv2DSS(ctxts, std::vector<seal::Plaintext>(), p_filter, META, result, out);
    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    if (measures.ret != Code::OK)
        return measures;

    stream.clear();
    for (auto& cipher : result) {
        cipher.save(stream);
    }

    measures.bytes += stream.str().length();
    std::cerr << "sending result: " << stream.str().length() << "\n";

    start = measure::now();
    Tensor<uint64_t> out_tensor;
    measures.ret        = conv.decryptToTensor(result, META, out_tensor);
    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    return measures;
}

Result conv2D_online2(const HomConv2DSS::Meta& meta, IO::NetIO& server,
                      const seal::SEALContext& context, const HomConv2DSS& conv,
                      const Tensor<uint64_t>& A1, const std::vector<Tensor<uint64_t>>& B1) {
    Result measures;
    measures.sending_recv = 0;
    Tensor<uint64_t> R1(HomConv2DSS::GetConv2DOutShape(meta));
    R1.Randomize(1000);

    ////////////////////////////////////////////////////////////////////////////
    // Enc(A1), enc(B1), send(A1), recv(A2)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();
    std::vector<seal::Ciphertext> enc_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<seal::Plaintext> encoded_A1;
    measures.ret = conv.encodeImage(A1, meta, encoded_A1, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Plaintext>> enc_B1;
    measures.ret = conv.encodeFilters(B1, meta, enc_B1, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    IO::send_encrypted_vector(server, enc_A1);
    std::vector<seal::Ciphertext> enc_A2;
    IO::recv_encrypted_vector(server, context, enc_A2);

    measures.sending_recv = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // M1 = A2 ⊙ B1 - R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    conv.add_plain_inplace(enc_A2, encoded_A1);
    std::vector<seal::Ciphertext> M1;
    measures.ret = conv.conv2DSS(enc_A2, enc_B1, meta, R1, M1, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // Send(M1), Recv(M2), Dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    IO::send_encrypted_vector(server, M1);
    std::vector<seal::Ciphertext> enc_M2;
    IO::recv_encrypted_vector(server, context, enc_M2);

    measures.sending_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // A ⊙ B + Dec(M2) - R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    Tensor<uint64_t> M2;
    measures.ret = conv.decryptToTensor(enc_M2, meta, M2, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();
    // Tensor<uint64_t> AB;
    // conv.idealFunctionality(A1, B1, meta, AB);

    // Utils::op_inplace<uint64_t>(AB, M2, [](uint64_t a, uint64_t b) { return a + b; });
    Utils::op_inplace<uint64_t>(M2, R1, [](uint64_t a, uint64_t b) { return a - b; });

    measures.plain_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    std::cerr << M2.channels() << " x " << M2.height() << " x " << M2.width() << "\n";

    measures.bytes = server.counter;
    measures.ret   = Code::OK;
    return measures;
}

Result conv2D_online(const HomConv2DSS::Meta& meta, IO::NetIO& server,
                     const seal::SEALContext& context, const HomConv2DSS& conv,
                     const Tensor<uint64_t>& A1, const std::vector<Tensor<uint64_t>>& B1) {
    Result measures;

    auto start = measure::now();
    std::vector<seal::Ciphertext> enc_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Ciphertext>> enc_B1;
    measures.ret = conv.encryptFilters(B1, meta, enc_B1, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;
    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();
    IO::send_encrypted_vector(server, enc_A1);
    IO::send_encrypted_filters(server, enc_B1);

    ////////////////////////////////////////////////////////////////////////////
    // A ⊙ B
    ////////////////////////////////////////////////////////////////////////////
    std::vector<seal::Ciphertext> result;
    IO::recv_encrypted_vector(server, context, result);
    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // Dec(M)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();
    Tensor<uint64_t> out_tensor;
    measures.ret        = conv.decryptToTensor(result, meta, out_tensor, N_THREADS);
    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    if (measures.ret != Code::OK)
        return measures;

    std::cerr << out_tensor.channels() << " x " << out_tensor.height() << " x "
              << out_tensor.width() << "\n";

    ////////////////////////////////////////////////////////////////////////////
    // A ⊙ B + Dec(M)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();
    Tensor<uint64_t> AB;
    conv.idealFunctionality(A1, B1, meta, AB);

    Utils::op_inplace<uint64_t>(AB, out_tensor, [](uint64_t a, uint64_t b) { return a + b; });
    measures.plain_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    measures.bytes = server.counter;
    measures.ret   = Code::OK;
    return measures;
}

Result perform_proto(const HomConv2DSS::Meta& meta, IO::NetIO& server,
                     const seal::SEALContext& context, const HomConv2DSS& hom_conv) {
    auto A1 = Utils::init_image(meta, 5);
    auto B1 = Utils::init_filter(meta, 2.0);

    Tensor<uint64_t> R(HomConv2DSS::GetConv2DOutShape(meta));
    R.Randomize(1000 << filter_prec);
    // for (int i = 0; i < R.channels(); ++i)
    //     for (int j = 0; j < R.height(); ++j)
    //         for (int k = 0; k < R.width(); ++k) R(i, j, k) = 1ULL << filter_prec;

    server.sync();
    auto measures  = conv2D_online2(meta, server, context, hom_conv, A1, B1);
    server.counter = 0;
    return measures;
}

} // anonymous namespace

int main() {
    seal::SEALContext context = Utils::init_he_context();

    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);

    HomConv2DSS conv;
    conv.setUp(context, skey, pkey);

    IO::NetIO server(nullptr, PORT, true);

    double total_time = 0;
    double total_data = 0;

    auto layers = Utils::init_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cerr << "Current layer: " << i << std::endl;
        auto res = perform_proto(layers[i], server, context, conv);
        if (res.ret != Code::OK) {
            std::cerr << CodeMessage(res.ret) << "\n";
            return EXEC_FAILED;
        }
        total_time += print_results(res, i);
        total_data += res.bytes / 1'000'000.0;
    }

    total_data /= 1'000.0;

    std::cout << "Party 1: total time [s]: " << total_time << "\n";
    std::cout << "Party 1: total data [GB]: " << total_data << "\n";
}
