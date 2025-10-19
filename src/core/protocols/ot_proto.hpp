#ifndef OT_PROTO_HPP_
#define OT_PROTO_HPP_

#include <iostream>

#include <io/net_io_channel.hpp>

#include <ot/bit-triple-generator.h>
#include <ot/silent_ot.h>

#include "core/utils.hpp"

static constexpr size_t LEN(const size_t& numTriple, const bool& packed) {
    return numTriple / (packed ? 8 : 1);
}

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
void triple_gen(TripleGenerator<Channel>& triple, uint8_t* a, uint8_t* b, uint8_t* c,
                size_t numTriple, const bool& packed, TripleGenMethod method);

template <class Channel>
void RunGen(TripleGenerator<Channel>& triple, const size_t& numTriple, const bool& packed);

} // namespace Server

namespace Client {

template <class Channel>
void triple_gen(TripleGenerator<Channel>& triple, uint8_t* a, uint8_t* b, uint8_t* c,
                size_t numTriple, const bool& packed, TripleGenMethod method);

template <class Channel>
void RunGen(TripleGenerator<Channel>& triple, const size_t& numTriple, const bool& packed);

} // namespace Client

template <class Channel>
void Server::triple_gen(TripleGenerator<Channel>& triple, uint8_t* a, uint8_t* b, uint8_t* c,
                        size_t numTriple, const bool& packed, TripleGenMethod method) {

    if (packed) {
        numTriple *= 8;
    }

    Triple trips(a, b, c, numTriple, packed);
    triple.get(emp::ALICE, &trips, method);

#ifdef VERIFY
    size_t len = numTriple / 8;
    Utils::log(Utils::Level::DEBUG, "VERIFYING OT");
    Utils::log(Utils::Level::DEBUG, numTriple);

    uint8_t* a2 = new uint8_t[len];
    uint8_t* b2 = new uint8_t[len];
    uint8_t* c2 = new uint8_t[len];

    triple.io->recv_data(a2, sizeof(uint8_t) * len);
    triple.io->recv_data(b2, sizeof(uint8_t) * len);
    triple.io->recv_data(c2, sizeof(uint8_t) * len);

    bool same = true;
    for (size_t i = 0; i < len; ++i) {
        if (((b2[i] ^ b[i]) & (a[i] ^ a2[i])) != (c2[i] ^ c[i])) {
            same = false;
            std::cout << i << "\n";
            break;
        }
    }

    if (same)
        Utils::log(Utils::Level::PASSED, "OT: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "OT: FAILED");
    delete[] a2;
    delete[] b2;
    delete[] c2;
#endif
}

template <class Channel>
void Server::RunGen(TripleGenerator<Channel>& triple, const size_t& numTriple, const bool& packed) {
    size_t len = LEN(numTriple, packed);
    uint8_t* a = new uint8_t[len];
    uint8_t* b = new uint8_t[len];
    uint8_t* c = new uint8_t[len];

    triple_gen(triple, a, b, c, numTriple, packed);

    delete[] a;
    delete[] b;
    delete[] c;
}

template <class Channel>
void Client::triple_gen(TripleGenerator<Channel>& triple, uint8_t* a, uint8_t* b, uint8_t* c,
                        size_t numTriple, const bool& packed, TripleGenMethod method) {

    if (packed) {
        numTriple *= 8;
    }

    Triple trips(a, b, c, numTriple, packed);
    triple.get(emp::BOB, &trips, method);

#ifdef VERIFY
    size_t len = numTriple / 8;
    triple.io->send_data(a, sizeof(uint8_t) * len, false);
    triple.io->send_data(b, sizeof(uint8_t) * len, false);
    triple.io->send_data(c, sizeof(uint8_t) * len, false);
    triple.io->flush();
#endif
}

template <class Channel>
void Client::RunGen(TripleGenerator<Channel>& triple, const size_t& numTriple, const bool& packed) {
    size_t len = LEN(numTriple, packed);
    uint8_t* a = new uint8_t[len];
    uint8_t* b = new uint8_t[len];
    uint8_t* c = new uint8_t[len];

    triple_gen(triple, a, b, c, numTriple, packed);

    delete[] a;
    delete[] b;
    delete[] c;
}

#endif