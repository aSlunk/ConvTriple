#ifndef OT_PROTO_HPP
#define OT_PROTO_HPP

#include <iostream>

#include <ot/silent_ot.h>

#define BIT_LEN 64

template <class T>
T bitmask(int l) {
    int bits = sizeof(T) * 8;

    if (l >= bits)
        return ~0ULL;
    else
        return ~(~0ULL << l);
}

namespace Server {

template <class Channel, class T>
void Test(cheetah::SilentOT<Channel>& ot, const size_t& batchsize) {
    T* M[batchsize];
    T RA  = 10;
    T RA2 = 32;
    for (size_t i = 0; i < batchsize; ++i) {
        M[i]    = new T[2];
        M[i][0] = RA2;
        M[i][1] = RA ^ RA2;
    }

    ot.send_impl(M, batchsize, BIT_LEN);

    T A1 = RA;
    T C1 = RA2;

#if VERIFY == 1
    T B2[batchsize];
    T C2[batchsize];

    ot.ferret->io->recv_data(B2, sizeof(T) * batchsize);
    ot.ferret->io->recv_data(C2, sizeof(T) * batchsize);

    Utils::log(Utils::Level::INFO, "VERIFYING OT");
    bool same = true;
    for (size_t i = 0; i < batchsize; ++i) {
        if ((B2[i] & A1) != (C2[i] ^ C1)) {
            same = false;
            break;
        }
    }

    if (same)
        Utils::log(Utils::Level::PASSED, "OT: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "OT: FAILED");
#endif

    for (size_t i = 0; i < batchsize; ++i) delete[] M[i];
}

} // namespace Server

namespace Client {

template <class Channel, class T>
void Test(cheetah::SilentOT<Channel>& ot, const size_t& batchsize) {
    uint8_t RB[batchsize];
#if VERIFY == 1
    T B2[batchsize];
#endif
    for (size_t i = 0; i < batchsize; ++i) {
        RB[i] = i % 2;
#if VERIFY == 1
        if (RB[i])
            B2[i] = bitmask<T>(BIT_LEN);
        else
            B2[i] = 0;
#endif
    }
    T C2[batchsize];
    ot.recv_impl(C2, RB, batchsize, BIT_LEN);

#if VERIFY == 1
    ot.ferret->io->send_data(B2, sizeof(T) * batchsize);
    ot.ferret->io->send_data(C2, sizeof(T) * batchsize);
#endif
}

} // namespace Client

#endif