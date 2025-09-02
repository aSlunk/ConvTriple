#include <cstdlib>
#include <iostream>
#include <vector>

#include <core/cheetah_interface.hpp>
#include <core/hpmpc_interface.hpp>
#include <core/networks/resnet50.hpp>

#define PARTY 2

void print_m128i(__m128i var) {
    int32_t* vals = (int32_t*)&var;
    for (int i = 0; i < 4; ++i) std::cout << "Element " << i << ": " << vals[i] << "\n";
}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cout << argv[0] << " <port> <host> <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
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

    int num_triples = 10;

    {
        int tmp    = 16;
        __m128i* a = new __m128i[tmp];
        __m128i* b = new __m128i[tmp];
        __m128i* c = new __m128i[tmp];

        a[15] = _mm_set_epi32(-1, -2, -3, -4);

        Iface::generateBoolTriplesCheetah((uint8_t*)a, (uint8_t*)b, (uint8_t*)c, 1,
                                          tmp * sizeof(*a), std::string(addr), port, PARTY,
                                          threads);

        print_m128i(a[15]);

        delete[] a;
        delete[] b;
        delete[] c;
    }

    for (int i = 0; i < 2; ++i) {
        {
            // num_triples = 48'168'448;
            num_triples = 22;
            std::vector<uint32_t> a(num_triples, 0);
            std::vector<uint32_t> b(num_triples, 0);
            std::vector<uint32_t> c(num_triples, 0);

            Iface::generateArithTriplesCheetah(a.data(), b.data(), c.data(), 32, num_triples,
                                               std::string(addr), port, PARTY, threads,
                                               Utils::PROTO::AB);
        }
    }

    Iface::generateFCTriplesCheetah(10, PARTY, std::string(addr), port, Utils::PROTO::AB);

    {
        Iface::ConvParm conv{
            .ic        = 3,
            .iw        = 224,
            .ih        = 224,
            .fc        = 3,
            .fw        = 7,
            .fh        = 7,
            .n_filters = 64,
            .stride    = 2,
            .padding   = 3,
        };
        Iface::generateConvTriplesCheetah(conv, 1, std::string(addr), port, PARTY,
                                          Utils::PROTO::AB);
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
