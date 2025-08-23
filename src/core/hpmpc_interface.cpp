#include "hpmpc_interface.hpp"

#include "protocols/ot_proto.hpp"

void Iface::generateBoolTriplesCheetah(uint32_t a[], uint32_t b[], uint32_t c[], int bitlength,
                                       uint64_t num_triples, std::string ip, int port, int party,
                                       int threads) {
    const char* addr = ip.c_str();
    if (ip == "")
        addr = nullptr;

    uint8_t* ai = new uint8_t[num_triples];
    uint8_t* bi = new uint8_t[num_triples];
    uint8_t* ci = new uint8_t[num_triples];

    for (size_t i = 0; i < num_triples; ++i) {
        ai[i] = static_cast<uint8_t>(a[i]);
        bi[i] = static_cast<uint8_t>(b[i]);
    }

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads);
    sci::OTPack<IO::NetIO> ot_pack(ios, threads, party, true, false);
    TripleGenerator<IO::NetIO> triple_gen(party, ios[0], &ot_pack);

    switch (party) {
    case emp::ALICE:
        Server::triple_gen(triple_gen, ai, bi, ci, num_triples, false);
        break;
    case emp::BOB:
        Client::triple_gen(triple_gen, ai, bi, ci, num_triples, false);
        break;
    }

    for (int i = 0; i < threads; ++i) delete ios[i];

    delete[] ios;

    for (size_t i = 0; i < num_triples; ++i) {
        a[i] = static_cast<uint32_t>(ai[i]);
        b[i] = static_cast<uint32_t>(bi[i]);
        c[i] = static_cast<uint32_t>(ci[i]);
    }

    delete[] ai;
    delete[] bi;
    delete[] ci;
}

template void Iface::generateArithTriplesCheetah<uint32_t>(uint32_t[], uint32_t[], uint32_t[], int,
                                                           uint64_t, std::string, int, int, int);