#ifndef HPMPC_INTERFACE_HPP_
#define HPMPC_INTERFACE_HPP_

#include "defs.hpp"
#include "protocols/bn_direct_proto.hpp"

#include "ot/bit-triple-generator.h"

#include <string>

#include <io/net_io_channel.hpp>

#include <gemini/cheetah/hom_bn_ss.h>

namespace Iface {

template <class Channel, class SerKey>
void exchange_keys(Channel** ios, const SerKey& pkey, seal::PublicKey& o_pkey,
                   const seal::SEALContext& ctx, int party) {
    switch (party) {
    case emp::ALICE:
        IO::send_pkey(*(ios[0]), pkey);
        IO::recv_pkey(*(ios[0]), ctx, o_pkey);
        break;
    case emp::BOB:
        IO::recv_pkey(*(ios[0]), ctx, o_pkey);
        IO::send_pkey(*(ios[0]), pkey);
        break;
    }
}

void setupBn(IO::NetIO** ios, gemini::HomBNSS& bn, const seal::SEALContext& ctx, const int& party);

void generateBoolTriplesCheetah(uint8_t a[], uint8_t b[], uint8_t c[], int bitlength,
                                uint64_t num_triples, std::string ip, int port, int party,
                                int threads = 1, TripleGenMethod method = _16KKOT_to_4OT,
                                unsigned io_offset = 1);

void generateArithTriplesCheetah(const uint32_t a[], const uint32_t b[], uint32_t c[],
                                 int bitlength, uint64_t num_triples, std::string ip, int port,
                                 int party, int threads = 1, Utils::PROTO proto = Utils::PROTO::AB,
                                 unsigned io_offset = 1);

void generateFCTriplesCheetah(const uint32_t* a, const uint32_t* b, uint32_t* c, int batch,
                              uint64_t com_dim, uint64_t dim2, int party, std::string ip, int port,
                              int threads, Utils::PROTO proto, int factor = 1,
                              unsigned io_offset = 1);

void generateConvTriplesCheetahWrapper(const uint32_t* a, const uint32_t* b, uint32_t* c,
                                       Utils::ConvParm parm, int batch, std::string ip, int port,
                                       int party, int threads, Utils::PROTO proto, int factor = 1,
                                       unsigned io_offset = 1);

void generateConvTriplesCheetah(const uint32_t* a, const uint32_t* b, uint32_t* c,
                                const gemini::HomConv2DSS::Meta& meta, int batch, std::string ip,
                                int port, int party, int threads, Utils::PROTO proto, int factor,
                                unsigned io_offset);

void generateBNTriplesCheetah(const uint32_t* a, const uint32_t* b, uint32_t* c, int batch,
                              size_t num_ele, size_t h, size_t w, std::string ip, int port,
                              int party, int threads, Utils::PROTO proto, int factor = 1,
                              unsigned io_offset = 1);
// void tmp(int party);

} // namespace Iface

#endif