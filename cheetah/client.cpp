#include <cstdlib>
#include <iostream>
#include <vector>

#include "all.hpp"

#define PARTY 2

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cout << argv[0] << " <port> <host> <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
    }

    if (PROTO != 1 && PROTO != 2) {
        Utils::log(Utils::Level::ERROR, "Unknown <PROTO>: ", PROTO);
    }

    size_t port      = strtoul(argv[1], NULL, 10);
    char* addr       = argv[2];
    size_t samples   = strtoul(argv[3], NULL, 10);
    size_t batchSize = strtoul(argv[4], NULL, 10);
    size_t threads;
    if (argc == 5)
        threads = N_THREADS;
    else
        threads = std::min(strtoul(argv[5], NULL, 10), (size_t)N_THREADS);

    HE_OT::HE<IO::NetIO> all(PARTY, addr, port, threads, batchSize, samples, false);
    // {
    //     auto layers = Utils::init_layers_fc();
    //     all.run_he(layers, all.get_fc());
    // }

    // {
    //     auto layers = Utils::init_layers();
    //     all.run_he(layers, all.get_conv());
    // }

    {
        auto layers = Utils::init_layers_bn();
        all.run_he(layers, all.get_bn());
        // double time = 0;
        // double data = 0;
        // for (auto& layer : layers) {
        //     time += all.alt_bn(layer, data);
        // }
        // Utils::log(Utils::Level::INFO, "Total time [s]: ", time);
        // Utils::log(Utils::Level::INFO, "Total data [MB]: ", data);
    }
}
