#ifndef OT_PROTO_HPP_
#define OT_PROTO_HPP_

#include <iostream>
#include <memory>

#include "core/utils.hpp"

#include <io/net_io_channel.hpp>

#include <ot/bit-triple-generator.h>
#include <ot/silent_ot.h>

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

/// Generate random (a, u), (b, v) with ab = u ^ v. See also `Client::mul_gen`.
/// Implements Algorithm 1 from https://ia.cr/2013/552.
template <class IO>
void mul_gen(const sci::OTPack<IO>* otpack, uint8_t* a, uint8_t* u, size_t num_muls);

} // namespace Server

namespace Client {

template <class Channel>
void triple_gen(TripleGenerator<Channel>& triple, uint8_t* a, uint8_t* b, uint8_t* c,
                size_t numTriple, const bool& packed, TripleGenMethod method);

template <class Channel>
void RunGen(TripleGenerator<Channel>& triple, const size_t& numTriple, const bool& packed);

/// Generate random (a, u), (b, v) with ab = u ^ v. See also `Server::mul_gen`.
/// Implements Algorithm 1 from https://ia.cr/2013/552.
template <class IO>
void mul_gen(const sci::OTPack<IO>* otpack, uint8_t* b, uint8_t* v, size_t num_muls);

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

template <class IO>
void Server::mul_gen(const sci::OTPack<IO>* otpack, uint8_t* a, uint8_t* u, size_t num_muls){
    auto a_buf = std::make_unique<bool[]>(num_muls);
    auto x_a = std::make_unique<uint8_t[]>(num_muls);
    otpack->silent_ot_reversed->template recv_ot_rm_rc<uint8_t>(x_a.get(), a_buf.get(), num_muls, 1);
    otpack->io->flush();

    // pack `bool`s
    for (size_t i = 0; i < num_muls; i++) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        u[byte_idx] |= x_a[i] << bit_idx;
        a[byte_idx] |= static_cast<uint8_t>(a_buf[i]) << bit_idx;
    }
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

template <class IO>
void Client::mul_gen(const sci::OTPack<IO>* otpack, uint8_t* b, uint8_t* v, size_t num_muls){
    auto x0 = std::make_unique<uint8_t[]>(num_muls);
    auto x1 = std::make_unique<uint8_t[]>(num_muls);
    otpack->silent_ot_reversed->template send_ot_rm_rc<uint8_t>(x0.get(), x1.get(), num_muls, 1);
    otpack->io->flush();

    for (size_t i = 0; i < num_muls; ++i) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        uint8_t x0_bit = x0[i] << bit_idx;
        b[byte_idx] |= x0_bit ^ (x1[i] << bit_idx); 
        v[byte_idx] |= x0_bit;
    }
}

#endif
