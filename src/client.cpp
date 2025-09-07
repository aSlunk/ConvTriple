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

    {
        int n       = 3;
        int batch   = 1;
        uint32_t* a = new uint32_t[n];
        uint32_t* b = new uint32_t[n];

        for (int i = 0; i < n; ++i) {
            a[i] = 1;
            b[i] = i;
        }

        uint32_t* c = new uint32_t[1];

        Iface::generateFCTriplesCheetah(a, b, c, batch, n, PARTY, std::string(addr), port,
                                        Utils::PROTO::AB);

        std::cout << c[0] << "\n";

        delete[] a;
        delete[] b;
        delete[] c;
    }

    {
        Iface::ConvParm conv{
            .ic        = 1,
            .iw        = 7,
            .ih        = 7,
            .fc        = 1,
            .fw        = 3,
            .fh        = 3,
            .n_filters = 1,
            .stride    = 1,
            .padding   = 0,
        };

        auto meta   = Utils::init_meta_conv(conv.ic, conv.ih, conv.iw, conv.fc, conv.fh, conv.fw,
                                            conv.n_filters, conv.stride, conv.padding);
        uint32_t* a = new uint32_t[meta.ishape.num_elements()];
        memset(a, 0, meta.ishape.num_elements() * 4);
        uint32_t* b = new uint32_t[meta.n_filters * meta.fshape.num_elements()];
        memset(b, 0, meta.n_filters * meta.fshape.num_elements() * 4);
        uint32_t* c = new uint32_t[gemini::GetConv2DOutShape(meta).num_elements()];

        Iface::generateConvTriplesCheetah(a, b, c, conv, 1, std::string(addr), port, PARTY, threads,
                                          Utils::PROTO::AB);

        delete[] a;
        delete[] b;
        delete[] c;
    }
    {
        int rows = 2;
        int cols = 4;
        std::vector<uint32_t> A(rows * cols);
        for (size_t i = 0; i < A.size(); ++i) A[i] = i;
        std::vector<uint32_t> B(rows, 1);
        std::vector<uint32_t> C(rows * cols);

        Iface::generateBNTriplesCheetah(A.data(), B.data(), C.data(), rows, cols, 1,
                                        std::string(addr), port, PARTY, Utils::PROTO::AB);
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
