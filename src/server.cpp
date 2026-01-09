#include "core/utils.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>

#include <core/benching.hpp>
#include <core/hpmpc_interface.hpp>
#include <core/networks/resnet50.hpp>
#include <io/file_io.hpp>

#define PARTY 1

int main(int argc, char** argv) {
    if (argc != 5 && argc != 4) {
        std::cout << argv[0] << " <port> <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
    }

    size_t port                     = strtoul(argv[1], NULL, 10);
    [[maybe_unused]] size_t samples = strtoul(argv[2], NULL, 10);
    size_t batchSize                = strtoul(argv[3], NULL, 10);
    size_t threads;
    if (argc == 4)
        threads = N_THREADS;
    else
        threads = std::min(strtoul(argv[4], NULL, 10), (size_t)N_THREADS);

    int num_triples = 1;

    {
        uint32_t a[num_triples * 8];
        uint8_t b[num_triples];
        uint32_t c[num_triples * 8];

        for (int i = 0; i < num_triples; ++i) {
            b[i] = 0xff;
            for (size_t j = 0; j < 8; ++j) {
                a[i * 8 + j] = 10;
            }
        }

        Iface::do_multiplex(num_triples * 8, a, b, c, PARTY, std::string(""), port, 1, threads);
        Iface::generateCOT(PARTY, a, nullptr, c, num_triples * 8, std::string(""), port, threads,
                           1);
    }

    {
        int tmp = 37'996'272;
        tmp     = 9'000'000 / 8;
        // tmp = 37'500'000;
        uint8_t* a = new uint8_t[tmp];
        uint8_t* b = new uint8_t[tmp];
        uint8_t* c = new uint8_t[tmp];

        Iface::generateBoolTriplesCheetah(a, b, c, 1, tmp, std::string("127.0.0.1"), port, PARTY,
                                          threads, _16KKOT_to_4OT);

        delete[] a;
        delete[] b;
        delete[] c;
    }

    auto& keys = Iface::Keys<IO::NetIO>::instance(PARTY, std::string(), port, threads, 1);
    keys.disconnect();

    {
        // num_triples = 9'006'592;
        num_triples = 22;
        std::vector<uint32_t> a_cp(num_triples * 1);
        std::vector<uint32_t> b_cp(num_triples * 1);
        std::vector<uint32_t> c_cp(num_triples * 1);

        for (int i = 0; i < 1; ++i) {
            // num_triples = 48'168'448;
            std::vector<uint32_t> a(num_triples, 0);
            std::vector<uint32_t> b(num_triples, 0);
            std::vector<uint32_t> c(num_triples, 0);

            Iface::generateArithTriplesCheetah(a.data(), b.data(), c.data(), 32, num_triples,
                                               std::string(""), port, PARTY, threads,
                                               Utils::PROTO::AB);

            for (size_t i = 0; i < 10; ++i)
                std::cout << c[i] << "\n";

            if (!IO::save_to_file("arith.triple", a.data(), b.data(), c.data(), num_triples)) {
                Utils::log(Utils::Level::FAILED, "Failed to save triples");
            } else {
                Utils::log(Utils::Level::PASSED, "Saved triples");
            }
            std::memcpy(a_cp.data() + num_triples * i, a.data(), num_triples * sizeof(uint32_t));
            std::memcpy(b_cp.data() + num_triples * i, b.data(), num_triples * sizeof(uint32_t));
            std::memcpy(c_cp.data() + num_triples * i, c.data(), num_triples * sizeof(uint32_t));
        }

        for (size_t round = 0; round < 1; ++round) {
            std::vector<uint32_t> a(num_triples);
            std::vector<uint32_t> b(num_triples);
            std::vector<uint32_t> c(num_triples);
            if (IO::read_from_file("arith.triple", a.data(), b.data(), c.data(), num_triples,
                                   true)) {
                Utils::log(Utils::Level::PASSED, "Read triples");

                bool passed = true;
                for (int i = 0; i < num_triples; ++i) {
                    if (a[i] != a_cp[i + round * num_triples]
                        || b[i] != b_cp[i + round * num_triples]
                        || c[i] != c_cp[i + round * num_triples]) {
                        passed = false;
                        break;
                    }
                }
                if (passed)
                    Utils::log(Utils::Level::PASSED, "Read CORRECT triples");
                else
                    Utils::log(Utils::Level::FAILED, "Triples mismatch");
            }
        }
    }

    {
        int n       = 3;
        int out     = 2;
        uint32_t* a = new uint32_t[n * batchSize];
        uint32_t* b = new uint32_t[n * batchSize * out];
        uint32_t* c = new uint32_t[out * batchSize];

        for (size_t j = 0; j < batchSize; ++j) {
            for (int i = 0; i < n * out; ++i) {
                a[i % n + n * j]   = 0;
                b[i + n * out * j] = i + j;
            }
        }

        Iface::generateFCTriplesCheetah(keys, nullptr, b, c, batchSize, n, out, PARTY, threads,
                                        Utils::PROTO::AB2);

        delete[] a;
        delete[] b;
        delete[] c;
    }

    {
        Utils::ConvParm conv{
            .batchsize = static_cast<int>(batchSize),
            .ic        = 1,
            .iw        = 10,
            .ih        = 10,
            .fc        = 1,
            .fw        = 7,
            .fh        = 7,
            .n_filters = 1,
            .stride    = 2,
            .padding   = 0,
        };

        auto meta   = Utils::init_meta_conv(conv.ic, conv.ih, conv.iw, conv.fc, conv.fh, conv.fw,
                                            conv.n_filters, conv.stride, conv.padding);
        uint32_t* a = new uint32_t[meta.ishape.num_elements() * batchSize];
        memset(a, 0, meta.ishape.num_elements() * sizeof(uint32_t) * batchSize);
        for (size_t i = 0; i < meta.ishape.num_elements() * batchSize; ++i) {
            a[i] = 1;
        }
        uint32_t* b = new uint32_t[meta.n_filters * meta.fshape.num_elements()];
        memset(b, 0, meta.n_filters * meta.fshape.num_elements() * sizeof(uint32_t));
        for (size_t i = 0; i < meta.n_filters; ++i)
            for (int j = 0; j < meta.fshape.num_elements(); ++j)
                b[i * meta.fshape.num_elements() + j] = 3;
        uint32_t* c = new uint32_t[Utils::getOutDim(conv).num_elements() * batchSize];

        std::vector<Utils::ConvParm> vec = {conv};
        std::vector<uint32_t*> bb        = {b};
        Iface::generateConvTriplesCheetahWrapper(keys, a, b, c, conv, PARTY, threads,
                                                 Utils::PROTO::AB, 1, true);

        for (size_t i = 0; i < Utils::getOutDim(conv).num_elements() * batchSize; ++i) {
            std::cout << "P" << PARTY << ": res" << c[i] << "\n";
        }

        delete[] a;
        delete[] b;
        delete[] c;
    }
    {
        int rows = 2;
        int h    = 2;
        int w    = 2;

        std::vector<uint32_t> A(rows * h * w * batchSize, 3);
        std::vector<uint32_t> B(rows * batchSize, 0);
        std::vector<uint32_t> C(rows * h * w * batchSize);

        Iface::generateBNTriplesCheetah(keys, A.data(), B.data(), C.data(), batchSize, rows, h, w,
                                        PARTY, threads, Utils::PROTO::AB2);
    }
    keys.disconnect();
}
