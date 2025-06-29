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
                    if (PROTO == 2 || (cur + wid) % 2 == 0) {
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
