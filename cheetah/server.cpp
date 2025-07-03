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
#include "protocols/ot_proto.hpp"

#define PARTY 1

using Utils::Result;

namespace {

Result run_conv(std::vector<std::vector<IO::NetIO>>& ioss, const size_t& batchSize,
                const seal::SEALContext& context, const gemini::HomConv2DSS& conv,
                gemini::HomConv2DSS::Meta& layer, double& total) {
    gemini::ThreadPool tpool(ioss.size());
    std::vector<Result> batches_results(ioss.size());
    auto batch = [&](long wid, size_t start, size_t end) -> Code {
        auto& ios = ioss[wid];
        for (size_t cur = start; cur < end; ++cur) {
            Result result;
            if (PROTO == 2 || cur % 2 == 0) {
                result = (Server::perform_proto(layer, ios, context, conv, ios.size()));
            } else {
                result = (Client::perform_proto(layer, ios, context, conv, ios.size()));
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

    return Utils::average(batches_results, false);
}

} // namespace

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

    size_t port      = strtoul(argv[1], NULL, 10);
    size_t samples   = strtoul(argv[2], NULL, 10);
    size_t batchSize = strtoul(argv[3], NULL, 10);
    size_t threads;
    if (argc == 4)
        threads = N_THREADS;
    else
        threads = std::min(strtoul(argv[4], NULL, 10), (size_t)N_THREADS);

    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads / batch_threads;
    auto ioss = Utils::init_ios<IO::NetIO>(nullptr, port, batch_threads, threads_per_thread);

    auto context = Utils::init_he_context();
    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);

    IO::send_pkey(ioss[0][0], *pkey);
    IO::recv_pkey(ioss[0][0], context, *pkey);

    gemini::HomConv2DSS conv;
    gemini::HomFCSS fc;
    conv.setUp(context, skey, pkey);
    fc.setUp(context, skey, pkey);

    auto m = Utils::init_meta_fc(10, 5);
    Server::perform_proto(m, ioss[0], context, fc, threads_per_thread);

    IO::NetIO* ios[threads];
    for (size_t i = 0; i < batch_threads; ++i)
        for (size_t j = 0; j < threads_per_thread; ++j)
            ios[i * threads_per_thread + j] = &ioss[i][j];

    cheetah::SilentOT<IO::NetIO> ot(PARTY, threads_per_thread, ios, true, true, "preot-server");
    Server::Test<IO::NetIO, uint64_t>(ot, 100);

    Utils::log(Utils::Level::DEBUG, "Samples: ", samples);
    Utils::log(Utils::Level::DEBUG, "batchSize: ", batchSize);
    Utils::log(Utils::Level::DEBUG, "threads: ", threads);
    Utils::log(Utils::Level::DEBUG, "#threads: ", batch_threads);
    Utils::log(Utils::Level::DEBUG, "threads per thread: ", threads_per_thread);

    double total      = 0;
    double total_data = 0;

    auto layers = Utils::init_layers();
    std::vector<Result> results(samples);           // all samples
    std::vector<Result> all_results(layers.size()); // averaged samples

    for (size_t i = 0; i < layers.size(); ++i) {
        Utils::log(Utils::Level::DEBUG, "Current layer: ", i);
        double tmp_total = 0;

        for (size_t round = 0; round < samples; ++round) {
            results[round] = run_conv(ioss, batchSize, context, conv, layers[i], tmp_total);
        }

        total += tmp_total / samples;

        all_results[i] = Utils::average(results, true);
        total_data += all_results[i].bytes / 1'000'000.0;
    }

    total_data /= 1'000.0;

    Utils::make_csv(all_results, batchSize, threads, "server.csv");
    std::cout << "Party 1: total time [s]: " << total << "\n";
    std::cout << "Party 1: total data [GB]: " << total_data << "\n";
}
