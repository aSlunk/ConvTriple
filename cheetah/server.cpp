#include <cstdlib>
#include <iostream>
#include <vector>

#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/hom_fc_ss.h>
#include <seal/seal.h>

#include <io/net_io_channel.hpp>

#include "defs.hpp"
#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"

using namespace gemini;
using Utils::Result;

int main(int argc, char** argv) {
    if (argc != 5 && argc != 4) {
        std::cout << argv[0] << " <port> <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
    }

    switch (PROTO) {
    case 1:
        Utils::log(Utils::Level::DEBUG, "RUNNING AB");
        break;
    case 2:
        Utils::log(Utils::Level::DEBUG, "RUNNING AB2");
        break;
    default:
        Utils::log(Utils::Level::ERROR, "Unknown <PROTO>: ", PROTO);
        break;
    }

    int port      = strtol(argv[1], NULL, 10);
    int samples   = strtol(argv[2], NULL, 10);
    int batchSize = strtol(argv[3], NULL, 10);
    int threads;
    if (argc == 4)
        threads = N_THREADS;
    else
        threads = std::min(atoi(argv[4]), N_THREADS);

    auto context = Utils::init_he_context();

    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);

    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads / batch_threads;
    auto ioss = Utils::init_ios<IO::NetIO>(nullptr, port, batch_threads, threads_per_thread);

    IO::send_pkey(ioss[0][0], *pkey);
    IO::recv_pkey(ioss[0][0], context, *pkey);

    HomConv2DSS conv;
    HomFCSS fc;
    conv.setUp(context, skey, pkey);
    fc.setUp(context, skey, pkey);

    auto m = Utils::init_meta_fc(10, 5, 10);
    Server::perform_proto(m, ioss[0], context, fc, threads_per_thread);

    double total_time = 0;
    double total_data = 0;

    Utils::log(Utils::Level::DEBUG, "Samples: ", samples);
    Utils::log(Utils::Level::DEBUG, "batchSize: ", batchSize);
    Utils::log(Utils::Level::DEBUG, "threads: ", threads);
    Utils::log(Utils::Level::DEBUG, "#threads: ", batch_threads);
    Utils::log(Utils::Level::DEBUG, "threads per thread: ", threads_per_thread);

    std::vector<Result> results(samples);
    double total = 0;

    auto layers = Utils::init_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        Utils::log(Utils::Level::DEBUG, "Current layer: ", i);

        for (int round = 0; round < samples; ++round) {
            ThreadPool tpool(batch_threads);
            std::vector<Result> batches_results(batch_threads);
            auto batch = [&](long wid, size_t start, size_t end) -> Code {
                auto& ios = ioss[wid];
                for (size_t cur = start; cur < end; ++cur) {
                    Result result;
                    if (PROTO == 2 || cur % 2 == 0) {
                        Utils::log(Utils::Level::DEBUG, "Server", cur, " ", wid);
                        result = (Server::perform_proto(layers[i], ios, context, conv,
                                                        threads_per_thread));
                    } else {
                        Utils::log(Utils::Level::DEBUG, "Client");
                        result = (Client::perform_proto(layers[i], ios, context, conv,
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
            if (code != Code::OK)
                Utils::log(Utils::Level::ERROR, CodeMessage(code));

            results[round] = Utils::average(batches_results, false);
        }
        auto res = Utils::average(results, true);
        total_time += Utils::print_results(res, i, batchSize, threads);
        total_data += res.bytes / 1'000'000.0;
    }

    total_data /= 1'000.0;

    std::cout << "Party 1: total time [s]: " << total_time << "\n";
    std::cout << "Party 1: total data [GB]: " << total_data << "\n";
    std::cout << "TOTAL: " << total << "\n";
}
