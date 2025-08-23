#include <cstdlib>
#include <iostream>
#include <vector>

#include <cheetah_interface.hpp>
#include <networks/resnet50.hpp>
#include <core/hpmpc_interface.hpp>

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

    int num_triples = 3;
    
    {
        uint8_t a[3] = {1, 1, 1};
        uint8_t b[3] = {0, 1, 0};
        uint8_t* c = new uint8_t[num_triples];

        Iface::generateTripleCheetah(a, b, c, 1, num_triples, std::string(addr), port, PARTY);

        for (int i = 0; i < num_triples; ++i) {
            std::cerr << i << ": " << static_cast<int>(a[i]) << ", " << static_cast<int>(b[i]) << ", " << static_cast<int>(c[i]) << std::endl;
        }
        delete[] c;
    }

    {
        uint64_t a[3] = {1, 1, 1};
        uint64_t b[3] = {0, 1, 0};
        uint64_t* c = new uint64_t[num_triples];

        Iface::generateArithTripleCheetah(a, b, c, 32, num_triples, std::string(addr), port, PARTY);

        for (int i = 0; i < num_triples; ++i) {
            std::cerr << i << ": " << a[i] << ", " << b[i] << ", " << c[i] << std::endl;
        }

        delete[] c;
    }
    
    // HE_OT::HE<IO::NetIO> all(PARTY, addr, port, threads, samples, true);
    // all.run_ot(20'000'000);

    // {
    //     auto layers = Utils::init_layers_fc();
    //     all.test_he(layers, all.get_fc(), batchSize);
    // }

    // {
    //     auto layers = ResNet50::init_layers_conv_cheetah();
    //     all.test_he(layers, all.get_conv(), batchSize);
    // }

    // {
    //     auto layers = ResNet50::init_layers_bn_cheetah();
    //     all.test_he(layers, all.get_bn(), batchSize);
    // }
}
