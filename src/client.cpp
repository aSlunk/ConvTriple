#include <cstdlib>
#include <iostream>
#include <vector>

#include <core/benching.hpp>
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

    size_t port                     = strtoul(argv[1], NULL, 10);
    char* addr                      = argv[2];
    [[maybe_unused]] size_t samples = strtoul(argv[3], NULL, 10);
    size_t batchSize                = strtoul(argv[4], NULL, 10);
    size_t threads;
    if (argc == 5)
        threads = N_THREADS;
    else
        threads = std::min(strtoul(argv[5], NULL, 10), (size_t)N_THREADS);

    int num_triples = 10;

    {
        int tmp = 37'996'272;
        tmp = 3;
        // tmp = 37'500'000;
        uint8_t* a = new uint8_t[tmp];
        uint8_t* b = new uint8_t[tmp];
        uint8_t* c = new uint8_t[tmp];

        Iface::generateBoolTriplesCheetah((uint8_t*)a, (uint8_t*)b, (uint8_t*)c, 1,
                                          tmp * sizeof(*a), std::string(addr), port, PARTY,
                                          threads);

        delete[] a;
        delete[] b;
        delete[] c;
    }

    for (int i = 0; i < 1; ++i) {
        {
            // num_triples = 48'168'448;
            num_triples = 9'006'592;
            std::vector<uint32_t> a(num_triples, 1);
            std::vector<uint32_t> b(num_triples, 1);
            std::vector<uint32_t> c(num_triples, 1);

            Iface::generateArithTriplesCheetah(a.data(), b.data(), c.data(), 32, num_triples,
                                               std::string(addr), port, PARTY, threads,
                                               Utils::PROTO::AB);
        }
    }

    {
        int n       = 3;
        int out     = 2;
        uint32_t* a = new uint32_t[n * batchSize];
        uint32_t* b = new uint32_t[n * batchSize * out];

        for (size_t j = 0; j < batchSize; ++j) {
            for (int i = 0; i < n * out; ++i) {
                a[i % n + n * j]   = 1;
                b[i + n * out * j] = 0;
            }
        }

        uint32_t* c = new uint32_t[out * batchSize];

        Iface::generateFCTriplesCheetah(a, nullptr, c, batchSize, n, out, PARTY, std::string(addr),
                                        port, threads, Utils::PROTO::AB2);

        for (size_t i = 0; i < batchSize; ++i) {
            for (int j = 0; j < out; ++j) {
                std::cout << "P" << PARTY << ": " << j << " " << c[i * out + j] << "\n";
            }
        }

        delete[] a;
        delete[] b;
        delete[] c;
    }

    {
        Utils::ConvParm conv{
            .ic        = 1,
            .iw        = 100,
            .ih        = 100,
            .fc        = 1,
            .fw        = 3,
            .fh        = 3,
            .n_filters = 1,
            .stride    = 1,
            .padding   = 1,
        };

        auto meta = Utils::init_meta_conv(conv.ic, conv.ih, conv.iw, conv.fc, conv.fh, conv.fw,
                                          conv.n_filters, conv.stride, conv.padding);

        uint32_t* a = new uint32_t[meta.ishape.num_elements() * batchSize];
        for (size_t i = 0; i < meta.ishape.num_elements() * batchSize; ++i) a[i] = i;
        uint32_t* b = new uint32_t[meta.n_filters * meta.fshape.num_elements() * batchSize];
        for (size_t i = 0; i < meta.n_filters; ++i)
            for (size_t j = 0; j < meta.fshape.num_elements() * batchSize; ++j)
                b[i * meta.fshape.num_elements() + j] = 3;

        uint32_t* c = new uint32_t[Utils::getOutDim(conv).num_elements() * batchSize];

        Iface::generateConvTriplesCheetahWrapper(a, nullptr, c, conv, batchSize, std::string(addr),
                                                 port, PARTY, threads, Utils::PROTO::AB);

        delete[] a;
        delete[] b;
        delete[] c;
    }
    {
        int rows = 2;
        int h    = 2;
        int w    = 2;
        std::vector<uint32_t> A(rows * h * w * batchSize);
        for (size_t i = 0; i < A.size(); ++i) A[i] = i;
        std::vector<uint32_t> B(rows * batchSize, 1);
        std::vector<uint32_t> C(rows * h * w * batchSize);

        Iface::generateBNTriplesCheetah(A.data(), B.data(), C.data(), batchSize, rows, h, w,
                                        std::string(addr), port, PARTY, threads, Utils::PROTO::AB2);
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
