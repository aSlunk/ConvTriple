#include <cstdlib>
#include <iostream>
#include <vector>

#include <core/benching.hpp>
#include <core/hpmpc_interface.hpp>
#include <core/networks/resnet50.hpp>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>

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
    [[maybe_unused]] size_t samples = strtoul(argv[3], NULL, 10);
    size_t batchSize                = strtoul(argv[4], NULL, 10);
    size_t threads;
    if (argc == 5)
        threads = N_THREADS;
    else
        threads = std::min(strtoul(argv[5], NULL, 10), (size_t)N_THREADS);

    std::string ip;
    {
        struct addrinfo hints, *res;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family   = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        int status = getaddrinfo(argv[2], argv[1], &hints, &res);
        if (status != 0) {
            std::cerr << "getaddrinfo error: " << gai_strerror(status) << endl;
            return -1;
        }

        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &((struct sockaddr_in*)(res->ai_addr))->sin_addr, ip_str,
                  INET_ADDRSTRLEN);
        freeaddrinfo(res);
        ip = ip_str;
    }

    int num_triples = 1;

    {
        uint32_t a[num_triples * 8];
        uint8_t b[num_triples];
        uint32_t c[num_triples * 8];

        for (int i = 0; i < num_triples; ++i) {
            b[i] = 0xaa;
            for (size_t j = 0; j < 8; ++j) {
                a[i * 8 + j] = 10;
            }
        }

        Iface::do_multiplex(num_triples * 8, a, b, c, PARTY, ip, port, 1, threads);
        Iface::generateCOT(PARTY, nullptr, b, c, num_triples * 8, ip, port, threads, 1);
    }

    {
        int tmp = 37'996'272;
        tmp     = 9'000'000 / 8;
        // tmp = 37'500'000;
        uint8_t* a = new uint8_t[tmp];
        uint8_t* b = new uint8_t[tmp];
        uint8_t* c = new uint8_t[tmp];

        Iface::generateBoolTriplesCheetah((uint8_t*)a, (uint8_t*)b, (uint8_t*)c, 1,
                                          tmp * sizeof(*a), ip, port, PARTY, threads,
                                          _16KKOT_to_4OT);

        delete[] a;
        delete[] b;
        delete[] c;
    }

    auto& keys = Iface::Keys<IO::NetIO>::instance(PARTY, ip, port, threads, 0);
    keys.disconnect();

    for (int i = 0; i < 1; ++i) {
        {
            // num_triples = 9'006'592;
            num_triples = 22;
            std::vector<uint32_t> a(num_triples, 1);
            std::vector<uint32_t> b(num_triples, 1);
            std::vector<uint32_t> c(num_triples, 1);

            Iface::generateArithTriplesCheetah(a.data(), b.data(), c.data(), 32, num_triples, ip,
                                               port, PARTY, threads, Utils::PROTO::AB);
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

        Iface::generateFCTriplesCheetah(keys, a, nullptr, c, batchSize, n, out, PARTY, threads,
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

        auto meta = Utils::init_meta_conv(conv.ic, conv.ih, conv.iw, conv.fc, conv.fh, conv.fw,
                                          conv.n_filters, conv.stride, conv.padding);

        uint32_t* a = new uint32_t[meta.ishape.num_elements() * batchSize];
        for (size_t i = 0; i < meta.ishape.num_elements() * batchSize; ++i) a[i] = i;
        uint32_t* b = new uint32_t[meta.n_filters * meta.fshape.num_elements()];
        for (size_t i = 0; i < meta.n_filters; ++i)
            for (int j = 0; j < meta.fshape.num_elements(); ++j)
                b[i * meta.fshape.num_elements() + j] = 3;

        uint32_t* c = new uint32_t[Utils::getOutDim(conv).num_elements() * batchSize];

        std::vector<Utils::ConvParm> vec = {conv};
        std::vector<uint32_t*> aa        = {a};
        // Iface::generateConvTriplesCheetah(keys, batchSize, vec, aa.data(), nullptr, c,
        // Utils::PROTO::AB2, PARTY, threads, 1);
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
        std::vector<uint32_t> A(rows * h * w * batchSize);
        for (size_t i = 0; i < A.size(); ++i) A[i] = i;
        std::vector<uint32_t> B(rows * batchSize, 1);
        std::vector<uint32_t> C(rows * h * w * batchSize);

        Iface::generateBNTriplesCheetah(keys, A.data(), B.data(), C.data(), batchSize, rows, h, w,
                                        PARTY, threads, Utils::PROTO::AB2);
    }

    keys.disconnect();
}
