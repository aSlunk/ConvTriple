#include <vector>

#include <gemini/cheetah/tensor.h>
#include <gemini/cheetah/tensor_encoder.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"

using namespace gemini;

namespace {

// constexpr char ADDRESS[11] = "127.0.0.1\0";

struct Result {
    size_t encryption;
    size_t cipher_op;
    size_t decryption;
    size_t plain_op;
    size_t send_recv;
    size_t bytes;
    Code ret;
};

double print_results(const Result& res, const int& layer = 0, std::ostream& out = std::cout) {
    if (!layer)
        out << "Encryption [ms],Cipher Calculations [s],Decryption [ms],Plain Calculations [ms], "
               "Sending and Receiving [s],Total [s],Bytes Send [MB]\n";

    double total = res.encryption / 1'000.0 + res.cipher_op / 1'000.0 + res.send_recv / 1'000.0
                   + res.decryption / 1'000.0 + res.plain_op / 1'000.0;
    total /= 1'000.0;

    out << res.encryption / 1'000.0 << ", " << res.cipher_op / 1'000'000.0 << ", "
        << res.decryption / 1'000. << ", " << res.plain_op / 1'000.0 << ", "
        << res.send_recv / 1'000'000.0 << ", " << total << ", " << res.bytes / 1'000'000.0 << "\n";

    return total;
}

Result Protocol2(IO::NetIO& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                 const HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<Tensor<uint64_t>>& B2, const Tensor<uint64_t>& R2) {
    Result measures;
    measures.send_recv = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Receive enc(A1) and enc/send A2
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    std::vector<seal::Plaintext> encoded_A2;
    std::vector<seal::Ciphertext> enc_A2;
    measures.ret = hom_conv.encryptImage(A2, meta, enc_A2, encoded_A2, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Plaintext>> enc_B2;
    measures.ret = hom_conv.encodeFilters(B2, meta, enc_B2, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    IO::recv_encrypted_vector(client, context, enc_A1);
    IO::send_encrypted_vector(client, enc_A2);

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // (A1' + A2) ⊙ B2 - R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    hom_conv.add_plain_inplace(enc_A1, encoded_A2);
    std::vector<seal::Ciphertext> enc_M2;
    measures.ret = hom_conv.conv2DSS(enc_A1, enc_B2, meta, R2, enc_M2, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    ////////////////////////////////////////////////////////////////////////////
    // Send result
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    std::vector<seal::Ciphertext> enc_M1;
    IO::recv_encrypted_vector(client, context, enc_M1);
    IO::send_encrypted_vector(client, enc_M2);

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // A2 ⊙ B2 + M1 - R2
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    Tensor<uint64_t> M1;
    hom_conv.decryptToTensor(enc_M1, meta, M1, N_THREADS);

    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    Utils::op_inplace<uint64_t>(M1, R2, [](uint64_t a, uint64_t b) -> uint64_t { return a + b; });

    measures.plain_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    measures.bytes = client.counter;
    measures.ret   = Code::OK;
    return measures;
}

Result Protocol(IO::NetIO& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                const HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                const std::vector<Tensor<uint64_t>>& B2, const Tensor<uint64_t>& R) {
    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // Receive enc(A1), enc(B1)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now(); // MEASURE_START

    std::vector<seal::Ciphertext> enc_A1;
    IO::recv_encrypted_vector(client, context, enc_A1);

    std::vector<std::vector<seal::Ciphertext>> enc_B1;
    IO::recv_encrypted_filters(client, context, enc_B1);

    measures.send_recv
        = std::chrono::duration_cast<Unit>(measure::now() - start).count(); // MEASURE_END

    ////////////////////////////////////////////////////////////////////////////
    // Encode A2, B2
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::vector<seal::Plaintext>> enc_B2;
    measures.ret = hom_conv.encodeFilters(B2, meta, enc_B2, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<seal::Plaintext> enc_A2;
    measures.ret = hom_conv.encodeImage(A2, meta, enc_A2, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    ////////////////////////////////////////////////////////////////////////////
    // A1 ⊙ B2 - R + A2 ⊙ B1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now(); // MEASURE_START
    std::vector<seal::Ciphertext> result;
    Tensor<uint64_t> out;

    measures.ret = hom_conv.conv2DSS(enc_A1, enc_B2, meta, R, result, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<seal::Ciphertext> result2;
    measures.ret = hom_conv.conv2DSS(enc_A2, enc_B1, meta, result2, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    hom_conv.add_inplace(result, result2, N_THREADS);

    measures.cipher_op
        = std::chrono::duration_cast<Unit>(measure::now() - start).count(); // MEASURE_END

    ////////////////////////////////////////////////////////////////////////////
    // Send result
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now(); // MEASURE_START
    IO::send_encrypted_vector(client, result);
    measures.send_recv
        += std::chrono::duration_cast<Unit>(measure::now() - start).count(); // MEASURE END

    ////////////////////////////////////////////////////////////////////////////
    // A2 ⊙ B2 + R
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();
    Tensor<uint64_t> final;
    hom_conv.idealFunctionality(A2, B2, meta, final);
    Utils::op_inplace<uint64_t>(final, R, [](uint64_t a, uint64_t b) -> uint64_t { return a + b; });
    measures.plain_op
        = std::chrono::duration_cast<Unit>(measure::now() - start).count(); // MEASURE END

    measures.bytes      = client.counter;
    measures.decryption = 0;
    measures.ret        = Code::OK;
    return measures;
}

Result perform_proto(const HomConv2DSS::Meta& meta, IO::NetIO& client,
                     const seal::SEALContext& context, const HomConv2DSS& hom_conv) {
    auto A2 = Utils::init_image(meta, 5);
    auto B2 = Utils::init_filter(meta, 2.0);

    Tensor<uint64_t> R(HomConv2DSS::GetConv2DOutShape(meta));
    // R.Randomize(5 << filter_prec);
    for (int i = 0; i < R.channels(); ++i)
        for (int j = 0; j < R.height(); ++j)
            for (int k = 0; k < R.width(); ++k) R(i, j, k) = 1ULL << filter_prec;

    client.sync();
    auto measures  = Protocol2(client, context, hom_conv, meta, A2, B2, R);
    client.counter = 0;
    return measures;
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return EXEC_FAILED;
    }

    auto context = Utils::init_he_context();

    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);

    HomConv2DSS hom_conv;
    hom_conv.setUp(context, skey, pkey);

    ////////////////////////////////////////////////////////////////////////////
    // Sample R
    ////////////////////////////////////////////////////////////////////////////
    IO::NetIO client(argv[1], PORT, true);

    double total_time = 0;
    double total_data = 0;

    auto layers = Utils::init_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        auto measures = perform_proto(layers[i], client, context, hom_conv);
        if (measures.ret != Code::OK) {
            std::cerr << CodeMessage(measures.ret) << "\n";
            return EXEC_FAILED;
        }

        total_time += print_results(measures, i);
        total_data += measures.bytes / 1'000'000.0;
    }

    total_data /= 1000.0;

    std::cout << "Party 2: total time [s]: " << total_time << "\n";
    std::cout << "Party 2: total data [GB]: " << total_data << "\n";
}
