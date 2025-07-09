#ifndef OT_PROTO_HPP
#define OT_PROTO_HPP

#include <iostream>

#include <io/net_io_channel.hpp>

#include <ot/bit-triple-generator.h>
#include <ot/silent_ot.h>

#include "defs.hpp"

#define BIT_LEN 1

static constexpr TripleGenMethod METHOD = TripleGenMethod::_2ROT;

template <class T>
T bitmask(int l) {
    int bits = sizeof(T) * 8;

    if (l >= bits)
        return ~0ULL;
    else
        return ~(~0ULL << l);
}

namespace Server {

template <class Channel>
void RunGen(TripleGenerator<Channel>& triple_gen, const size_t& batchSize, const bool& packed);

template <class Channel>
void Test(cheetah::SilentOT<Channel>& ot, cheetah::SilentOT<Channel>& rev_ot,
          const size_t& batchsize);

} // namespace Server

namespace Client {

template <class Channel>
void RunGen(TripleGenerator<Channel>& triple_gen, const size_t& batchSize, const bool& packed);

template <class Channel>
void Test(cheetah::SilentOT<Channel>& ot, cheetah::SilentOT<Channel>& rev_ot,
          const size_t& batchsize);

} // namespace Client

template <class Channel>
void Server::RunGen(TripleGenerator<Channel>& triple_gen, const size_t& batchSize,
                    const bool& packed) {
    size_t len = batchSize / (packed ? 8 : 1);
    uint8_t a[len];
    uint8_t b[len];
    uint8_t c[len];

    triple_gen.generate(emp::ALICE, a, b, c, batchSize, METHOD, packed);

#if VERIFY == 1
    Utils::log(Utils::Level::INFO, "VERIFYING OT");
    uint8_t a2[len];
    uint8_t b2[len];
    uint8_t c2[len];

    triple_gen.io->recv_data(a2, sizeof(uint8_t) * len);
    triple_gen.io->recv_data(b2, sizeof(uint8_t) * len);
    triple_gen.io->recv_data(c2, sizeof(uint8_t) * len);

    bool same = true;
    for (size_t i = 0; i < len; ++i) {
        if (((b2[i] ^ b[i]) & (a[i] ^ a2[i])) != (c2[i] ^ c[i])) {
            same = false;
            break;
        }
    }

    if (same)
        Utils::log(Utils::Level::PASSED, "OT: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "OT: FAILED");
#endif
}

template <class Channel>
void Server::Test(cheetah::SilentOT<Channel>& ot, cheetah::SilentOT<Channel>& rev_ot,
                  const size_t& batchsize) {
    uint8_t* a = new uint8_t[batchsize];
    uint8_t* b = new uint8_t[batchsize];
    uint8_t* c = new uint8_t[batchsize];
    uint8_t* u = new uint8_t[batchsize];
    uint8_t* v = new uint8_t[batchsize];

    rev_ot.recv_ot_rm_rc(u, (bool*)a, batchsize, BIT_LEN);
    ot.send_ot_rm_rc(v, b, batchsize, BIT_LEN);

    for (size_t i = 0; i < batchsize; ++i) {
        b[i] = b[i] ^ v[i];
        c[i] = (a[i] & b[i]) ^ u[i] ^ v[i];
    }

#if VERIFY == 1
    Utils::log(Utils::Level::INFO, "VERIFYING OT");
    uint8_t A2[batchsize];
    uint8_t B2[batchsize];
    uint8_t C2[batchsize];

    ot.ferret->io->recv_data(A2, sizeof(uint8_t) * batchsize);
    ot.ferret->io->recv_data(B2, sizeof(uint8_t) * batchsize);
    ot.ferret->io->recv_data(C2, sizeof(uint8_t) * batchsize);

    bool same   = true;
    size_t ones = 0;
    for (size_t i = 0; i < batchsize; ++i) {
        if (((B2[i] ^ b[i]) & (a[i] ^ A2[i])) != (C2[i] ^ c[i])) {
            same = false;
            break;
        }

        if (B2[i])
            ones++;
    }

    if (same)
        Utils::log(Utils::Level::PASSED, "OT: PASSED, Ones: ", ones);
    else
        Utils::log(Utils::Level::FAILED, "OT: FAILED");
#endif
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] u;
    delete[] v;
}

template <class Channel>
void Client::RunGen(TripleGenerator<Channel>& triple_gen, const size_t& batchSize,
                    const bool& packed) {
    size_t len = batchSize / (packed ? 8 : 1);
    uint8_t a[len];
    uint8_t b[len];
    uint8_t c[len];

    triple_gen.generate(emp::BOB, a, b, c, batchSize, METHOD, packed);

#if VERIFY == 1
    triple_gen.io->send_data(a, sizeof(uint8_t) * len, false);
    triple_gen.io->send_data(b, sizeof(uint8_t) * len, false);
    triple_gen.io->send_data(c, sizeof(uint8_t) * len, false);
    triple_gen.io->flush();
#endif
}

template <class Channel>
void Client::Test(cheetah::SilentOT<Channel>& ot, cheetah::SilentOT<Channel>& rev_ot,
                  const size_t& batchsize) {
    assert(sizeof(uint8_t) == sizeof(bool));
    uint8_t* b = new uint8_t[batchsize];

    uint8_t* a = new uint8_t[batchsize];
    uint8_t* c = new uint8_t[batchsize];
    uint8_t* u = new uint8_t[batchsize];
    uint8_t* v = new uint8_t[batchsize];

    rev_ot.send_ot_rm_rc(v, b, batchsize, BIT_LEN);
    ot.recv_ot_rm_rc(u, (bool*)a, batchsize, BIT_LEN);

    for (size_t i = 0; i < batchsize; ++i) {
        b[i] = b[i] ^ v[i];
        c[i] = (a[i] & b[i]) ^ u[i] ^ v[i];
    }

#if VERIFY == 1
    ot.ferret->io->send_data(a, sizeof(uint8_t) * batchsize);
    ot.ferret->io->send_data(b, sizeof(uint8_t) * batchsize);
    ot.ferret->io->send_data(c, sizeof(uint8_t) * batchsize);
    ot.ferret->io->flush();
#endif
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] u;
    delete[] v;
}

#endif