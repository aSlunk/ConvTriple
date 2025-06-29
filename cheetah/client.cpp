#include <utility>
#include <vector>

#include <gemini/cheetah/tensor.h>
#include <gemini/cheetah/tensor_encoder.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"
#include "proto.hpp"

using namespace gemini;
using Utils::Result;

namespace {

Result Protocol
    [[maybe_unused]] (IO::NetIO& client, const seal::SEALContext& context,
                      const HomConv2DSS& hom_conv, const HomConv2DSS::Meta& meta,
                      const Tensor<uint64_t>& A2, const std::vector<Tensor<uint64_t>>& B2,
                      const Tensor<uint64_t>& R, const size_t& threads) {
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

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cout << argv[0] << " <port> <host> <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
    }

    long port     = strtol(argv[1], NULL, 10);
    char* addr    = argv[2];
    int samples   = strtol(argv[3], NULL, 10);
    int batchSize = strtol(argv[4], NULL, 10);
    int threads;
    if (argc == 3)
        threads = N_THREADS;
    else
        threads = strtol(argv[5], NULL, 10);

    auto context = Utils::init_he_context();

    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);

    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads / batch_threads;
    auto ioss = Utils::init_ios<IO::NetIO>(addr, port, batch_threads, threads_per_thread);

    IO::send_pkey(ioss[0][0], *pkey);
    IO::recv_pkey(ioss[0][0], context, *pkey);

    HomConv2DSS hom_conv;
    hom_conv.setUp(context, skey, pkey);

    double totalTime = 0;
    double totalData = 0;

    std::vector<Result> results(samples);
    double total = 0;

    auto layers = Utils::init_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        for (int round = 0; round < samples; ++round) {
            ThreadPool tpool(batch_threads);
            std::vector<Result> batches_results(batch_threads);
            auto batch = [&](long wid, size_t start, size_t end) -> Code {
                auto& ios = ioss[wid];
                for (size_t cur = start; cur < end; ++cur) {
                    Result result;
                    if (PROTO == 3 || (cur + wid) % 2 == 0) {
                        result = (Client::perform_proto(layers[i], ios, context, hom_conv,
                                                        threads_per_thread));
                    } else {
                        result = (Server::perform_proto(layers[i], ios, context, hom_conv,
                                                        threads_per_thread));
                    }

                    if (result.ret != Code::OK)
                        return result.ret;

                    Utils::add_result(batches_results[wid], result);
                }
                return Code::OK;
            };

            auto start = measure::now();
            auto code  = gemini::LaunchWorks(tpool, batchSize, batch);
            total += std::chrono::duration_cast<Unit>(measure::now() - start).count() / 1'000'000.0;
            if (code != Code::OK) {
                std::cerr << CodeMessage(code) << "\n";
                return EXEC_FAILED;
            }
            results[round] = Utils::average(batches_results, false);
        }
        auto measures = Utils::average(results, true);
        totalTime += Utils::print_results(measures, i, batchSize, threads);
        totalData += measures.bytes / 1'000'000.0;
    }

    totalData /= 1000.0;

    std::cout << "Party 2: total time [s]: " << totalTime << "\n";
    std::cout << "Party 2: total data [GB]: " << totalData << "\n";
    std::cout << "TOTAL: " << total << "\n";
}
