#include <cstdlib>
#include <iostream>
#include <vector>

#include <core/cheetah_interface.hpp>
#include <core/hpmpc_interface.hpp>
#include <core/networks/resnet50.hpp>

#define PARTY 2

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cout << argv[0] << " <port> <host> <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
    }

    if (PROTO != 1 && PROTO != 2) {
        Utils::log(Utils::Level::ERROR, "Unknown <PROTO>: ", PROTO);
    }

    size_t port                       = strtoul(argv[1], NULL, 10);
    char* addr                        = argv[2];
    [[maybe_unused]] size_t samples   = strtoul(argv[3], NULL, 10);
    [[maybe_unused]] size_t batchSize = strtoul(argv[4], NULL, 10);
    size_t threads;
    if (argc == 5)
        threads = N_THREADS;
    else
        threads = std::min(strtoul(argv[5], NULL, 10), (size_t)N_THREADS);

    int num_triples = 320608;

    // {
    //     uint8_t* a = new uint8_t[num_triples];
    //     uint8_t* b = new uint8_t[num_triples];
    //     uint8_t* c = new uint8_t[num_triples];

    //     Iface::generateBoolTriplesCheetah(a, b, c, 1, num_triples, std::string(addr), port,
    //     PARTY);

    //     delete[] a;
    //     delete[] b;
    //     delete[] c;
    // }

    {
        // num_triples = 30'000'000;
        num_triples = 48'168'448;
        std::vector<uint32_t> a(num_triples, 1);
        std::vector<uint32_t> b(num_triples, 1);
        std::vector<uint32_t> c(num_triples, 1);

        Iface::generateArithTriplesCheetah(a.data(), b.data(), c.data(), 32, num_triples,
                                           std::string(addr), port, PARTY, threads);
    }

    Iface::generateFCTriplesCheetah(10, PARTY, std::string(addr), port);

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
