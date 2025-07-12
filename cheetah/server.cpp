#include <cstdlib>
#include <iostream>
#include <vector>

#include "all.hpp"

#define PARTY 1

int main(int argc, char** argv) {
    if (argc != 5 && argc != 4) {
        std::cout << argv[0] << " <port> <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
    }

    switch (PROTO) {
    case 1:
        Utils::log(Utils::Level::DEBUG, "Protcol: AB");
        break;
    case 2:
        Utils::log(Utils::Level::DEBUG, "Protcol: AB2");
        break;
    default:
        Utils::log(Utils::Level::ERROR, "Unknown <PROTO>: ", PROTO);
        break;
    }

    Utils::log(Utils::Level::DEBUG, "BIT_LEN: ", BIT_LEN);

    size_t port      = strtoul(argv[1], NULL, 10);
    size_t samples   = strtoul(argv[2], NULL, 10);
    size_t batchSize = strtoul(argv[3], NULL, 10);
    size_t threads;
    if (argc == 4)
        threads = N_THREADS;
    else
        threads = std::min(strtoul(argv[4], NULL, 10), (size_t)N_THREADS);

    HE_OT::HE<IO::NetIO> all(PARTY, nullptr, port, threads, batchSize, samples, false);
    auto start = measure::now();
    all.test_bn();
    std::cerr << Utils::to_sec(Utils::time_diff(start)) << "\n";

    start = measure::now();
    all.alt_bn();
    std::cerr << Utils::to_sec(Utils::time_diff(start)) << "\n";

    // all.test();
    // auto layers_fc = Utils::init_layers_fc();
    // all.run_he(layers_fc, all.get_fc());

    // all.run_ot(20'000'000, false);

    auto layers = Utils::init_layers();
    all.run_he(layers, all.get_conv());
}