#include <cstdlib>
#include <iostream>
#include <vector>

#include <io/net_io_channel.hpp>

#include "defs.hpp"
#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"
#include "protocols/ot_proto.hpp"

#define PARTY 2

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
                result = (Client::perform_proto(layer, ios, context, conv, ios.size()));
            } else {
                result = (Server::perform_proto(layer, ios, context, conv, ios.size()));
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
    if (argc != 5 && argc != 6) {
        std::cout << argv[0] << " <port> <host> <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
    }

    if (PROTO != 1 && PROTO != 2) {
        Utils::log(Utils::Level::ERROR, "Unknown <PROTO>: ", PROTO);
    }

    long port     = strtol(argv[1], NULL, 10);
    char* addr    = argv[2];
    int samples   = strtol(argv[3], NULL, 10);
    int batchSize = strtol(argv[4], NULL, 10);
    int threads;
    if (argc == 5)
        threads = N_THREADS;
    else
        threads = std::min(atoi(argv[5]), N_THREADS);

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

    gemini::HomConv2DSS hom_conv;
    gemini::HomFCSS fc;
    hom_conv.setUp(context, skey, pkey);
    fc.setUp(context, skey, pkey);

    auto m = Utils::init_meta_fc(10, 5);
    Client::perform_proto(m, ioss[0], context, fc, threads_per_thread);

    IO::NetIO* ios[threads];
    for (size_t i = 0; i < batch_threads; ++i)
        for (size_t j = 0; j < threads_per_thread; ++j)
            ios[i * threads_per_thread + j] = &ioss[i][j];

    cheetah::SilentOT<IO::NetIO> ot(PARTY, threads_per_thread, ios);

    Client::Test<IO::NetIO, uint64_t>(ot, 100);

    double totalTime = 0;
    double totalData = 0;

    std::vector<Result> results(samples);
    double total = 0;

    auto layers = Utils::init_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        for (int round = 0; round < samples; ++round) {
            results[round] = run_conv(ioss, batchSize, context, hom_conv, layers[i], total);
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
