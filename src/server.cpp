#include <cstdlib>
#include <iostream>
#include <vector>

#include <core/benching.hpp>
#include <core/hpmpc_interface.hpp>
#include <core/networks/resnet50.hpp>

#define PARTY 1

int main(int argc, char** argv) {
    if (argc != 5 && argc != 4) {
        std::cout << argv[0] << " <port> <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
    }

    size_t port                       = strtoul(argv[1], NULL, 10);
    [[maybe_unused]] size_t samples   = strtoul(argv[2], NULL, 10);
    [[maybe_unused]] size_t batchSize = strtoul(argv[3], NULL, 10);
    size_t threads;
    if (argc == 4)
        threads = N_THREADS;
    else
        threads = std::min(strtoul(argv[4], NULL, 10), (size_t)N_THREADS);

    int num_triples = 10;

    {
        int tmp    = 256;
        uint8_t* a = new uint8_t[tmp];
        uint8_t* b = new uint8_t[tmp];
        uint8_t* c = new uint8_t[tmp];

        Iface::generateBoolTriplesCheetah(a, b, c, 1, tmp, std::string(""), port, PARTY, threads);

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

            a[2] = 2;
            b[2] = 2;

            Iface::generateArithTriplesCheetah(a.data(), b.data(), c.data(), 1, num_triples,
                                               std::string(""), port, PARTY, threads,
                                               Utils::PROTO::AB);

            if (!Utils::save_to_file("arith.triple", a.data(), b.data(), c.data(), num_triples)) {
                Utils::log(Utils::Level::FAILED, "Failed to save triples");
            } else {
                Utils::log(Utils::Level::PASSED, "Saved triples");
            }

            std::vector<uint32_t> a_cp(num_triples);
            std::vector<uint32_t> b_cp(num_triples);
            std::vector<uint32_t> c_cp(num_triples);

            bool passed = true;
            if (!Utils::read_from_file("arith.triple", a_cp.data(), b_cp.data(), c_cp.data(),
                                       num_triples)) {
                Utils::log(Utils::Level::FAILED, "Failed to read triples");
            } else {
                for (int i = 0; i < num_triples; ++i) {
                    if (a_cp[i] != a[i] || b_cp[i] != b[i] || c_cp[i] != c[i]) {
                        passed = false;
                        break;
                    }
                }
                if (passed)
                    Utils::log(Utils::Level::PASSED, "Numbers match");
                else
                    Utils::log(Utils::Level::FAILED, "Read wrong numbers");
            }
        }
    }

    {
        int n       = 3;
        int batch   = 2;
        uint32_t* a = new uint32_t[n * batch];
        uint32_t* b = new uint32_t[n * batch];
        uint32_t* c = new uint32_t[1 * batch];

        for (int j = 0; j < batch; ++j) {
            for (int i = 0; i < n; ++i) {
                a[i + n * j] = 0;
                b[i + n * j] = 0;
            }
        }

        Iface::generateFCTriplesCheetah(a, b, c, batch, n, PARTY, std::string(""), port,
                                        Utils::PROTO::AB);

        for (int j = 0; j < batch; ++j) {
            std::cout << j << " " << c[j] << "\n";
        }

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

        Iface::generateConvTriplesCheetah(a, b, c, conv, 1, std::string(""), port, PARTY, threads,
                                          Utils::PROTO::AB);
        delete[] a;
        delete[] b;
        delete[] c;
    }
    {
        int batch = 2;
        int rows = 2;
        int cols = 3;

        std::vector<uint32_t> A(rows * cols * batch, 3);
        std::vector<uint32_t> B(rows * batch, 1);
        std::vector<uint32_t> C(rows * cols * batch);

        Iface::generateBNTriplesCheetah(A.data(), B.data(), C.data(), batch, rows, cols,
                                        std::string(""), port, PARTY, threads, Utils::PROTO::AB);
    }

    // HE_OT::HE<IO::NetIO> all(PARTY, nullptr, port, threads, samples, true);
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