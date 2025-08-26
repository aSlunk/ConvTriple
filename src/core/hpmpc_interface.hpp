#ifndef HPMPC_INTERFACE_HPP_
#define HPMPC_INTERFACE_HPP_

#include "defs.hpp"
#include "protocols/bn_direct_proto.hpp"

#include <string>

#include <io/net_io_channel.hpp>

#include <ot/bit-triple-generator.h>
#include <ot/cheetah-ot_pack.h>

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

void generateBoolTriplesCheetah(uint8_t a[], uint8_t b[], uint8_t c[], int bitlength,
                                uint64_t num_triples, std::string ip, int port, int party,
                                int threads = 1);

void generateArithTriplesCheetah(uint32_t a[], uint32_t b[], uint32_t c[], int bitlength,
                                 uint64_t num_triples, std::string ip, int port, int party,
                                 int threads = 1);

} // namespace Iface

#endif