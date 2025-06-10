#include <vector>

#include <gemini/cheetah/tensor.h>
#include <gemini/cheetah/tensor_encoder.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"

using namespace gemini;

namespace {

// constexpr char ADDRESS[11] = "127.0.0.1\0";

Result Protocol3(IO::NetIO& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                 HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                 const std::vector<Tensor<uint64_t>>& B2, const size_t& threads) {
    meta.is_shared_input = true;
    Result measures;
    measures.send_recv  = 0;
    measures.decryption = 0;
    measures.plain_op   = 0;

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

    start = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    IO::recv_encrypted_vector(client, context, enc_A1);

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    // hom_conv.add_plain_inplace(enc_A1, enc_A2, N_THREADS);
    std::vector<seal::Ciphertext> M2;
    Tensor<uint64_t> R;
    measures.ret = hom_conv.conv2DSS(enc_A1, enc_A2, enc_B2, meta, M2, R, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    IO::send_encrypted_vector(client, M2);

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    measures.bytes = client.counter;
    return measures;
}

Result Protocol2(IO::NetIO& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
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
    std::vector<seal::Ciphertext> enc_A2;
    measures.ret = hom_conv.encryptImage(A2, meta, enc_A2, encoded_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Plaintext>> enc_B2;
    measures.ret = hom_conv.encodeFilters(B2, meta, enc_B2, threads);
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

    std::vector<seal::Ciphertext> enc_M1;
    IO::recv_encrypted_vector(client, context, enc_M1);
    IO::send_encrypted_vector(client, enc_M2);

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();

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

Result Protocol(IO::NetIO& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                const HomConv2DSS::Meta& meta, const Tensor<uint64_t>& A2,
                const std::vector<Tensor<uint64_t>>& B2, const Tensor<uint64_t>& R,
                const size_t& threads) {
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
    measures.ret = hom_conv.encodeFilters(B2, meta, enc_B2, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<seal::Plaintext> enc_A2;
    measures.ret = hom_conv.encodeImage(A2, meta, enc_A2, threads);
    if (measures.ret != Code::OK)
        return measures;

    ////////////////////////////////////////////////////////////////////////////
    // A1 ⊙ B2 - R + A2 ⊙ B1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now(); // MEASURE_START
    std::vector<seal::Ciphertext> result;
    Tensor<uint64_t> out;

    measures.ret = hom_conv.conv2DSS(enc_A1, enc_B2, meta, R, result, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<seal::Ciphertext> result2;
    measures.ret = hom_conv.conv2DSS(enc_A2, enc_B1, meta, result2, threads);
    if (measures.ret != Code::OK)
        return measures;

    hom_conv.add_inplace(result, result2, threads);

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

Result perform_proto(HomConv2DSS::Meta& meta, IO::NetIO& client, const seal::SEALContext& context,
                     const HomConv2DSS& hom_conv, const size_t& threads) {
    auto A2 = Utils::init_image(meta, 5);
    auto B2 = Utils::init_filter(meta, 2.0);

    client.sync();
#if PROTO == 3
    auto measures = Protocol3(client, context, hom_conv, meta, A2, B2, threads);
#elif PROTO == 2
    auto measures = Protocol2(client, context, hom_conv, meta, A2, B2, threads);
#endif
    client.counter = 0;
    return measures;
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cout << argv[0] << " <host> <samples> <batchSize> (<threads>)\n";
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

    double totalTime = 0;
    double totalData = 0;

    int threads;
    if (argc == 3)
        threads = N_THREADS;
    else
        threads = strtol(argv[4], NULL, 10);

    int samples   = strtol(argv[2], NULL, 10);
    int batchSize = strtol(argv[3], NULL, 10);

    std::vector<Result> results(samples);
    std::cerr << "Samples: " << samples << "\n";
    std::cerr << "BatchSize: " << batchSize << "\n";
    std::cerr << "threads: " << threads << "\n";

    auto layers = Utils::init_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        for (int round = 0; round < samples; ++round) {
            Result res = {.encryption = 0,
                          .cipher_op  = 0,
                          .plain_op   = 0,
                          .decryption = 0,
                          .send_recv  = 0,
                          .bytes      = 0,
                          .ret        = Code::OK};
            for (int batch = 0; batch < batchSize; ++batch) {
                auto cur = perform_proto(layers[i], client, context, hom_conv, threads);
                if (cur.ret != Code::OK) {
                    std::cerr << CodeMessage(cur.ret) << "\n";
                    return EXEC_FAILED;
                }

                res.encryption += cur.encryption;
                res.cipher_op += cur.cipher_op;
                res.plain_op += cur.plain_op;
                res.decryption += cur.decryption;
                res.send_recv += cur.send_recv;
                res.bytes += cur.bytes;
            }
            results[round] = res;
        }

        auto measures = average(results);
        totalTime += print_results(measures, i, batchSize, threads);
        totalData += measures.bytes / 1'000'000.0;
    }

    totalData /= 1000.0;

    std::cout << "Party 2: total time [s]: " << totalTime << "\n";
    std::cout << "Party 2: total data [GB]: " << totalData << "\n";
}
