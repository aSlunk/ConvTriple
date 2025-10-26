#ifndef HPMPC_INTERFACE_HPP_
#define HPMPC_INTERFACE_HPP_

#include "protocols/bn_direct_proto.hpp"
#include "utils.hpp"

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

void generateFCTriplesCheetah(IO::NetIO** ios, const uint32_t* a, const uint32_t* b, uint32_t* c,
                              int batch, uint64_t com_dim, uint64_t dim2, int party, int threads,
                              Utils::PROTO proto, int factor = 1);

template <class Channel>
gemini::HomConv2DSS setupConv(Channel** ios, int party);

void generateConvTriplesCheetahWrapper(IO::NetIO** ios, const uint32_t* a, const uint32_t* b,
                                       uint32_t* c, Utils::ConvParm parm, int party, int threads,
                                       Utils::PROTO proto, int factor = 1);

void generateConvTriplesCheetah(IO::NetIO** ios, gemini::HomConv2DSS& hom_conv,
                                size_t total_batches, std::vector<Utils::ConvParm>& parms,
                                uint32_t** a, uint32_t** b, uint32_t* c, Utils::PROTO proto,
                                int party, int threads, int factor);

void generateConvTriplesCheetahPhase1(IO::NetIO** ios, const gemini::HomConv2DSS& hom_conv,
                                      const uint32_t* a, const uint32_t* b, Utils::ConvParm parm,
                                      vector<vector<seal::Plaintext>>& enc_a,
                                      vector<vector<vector<seal::Plaintext>>>& enc_b,
                                      vector<vector<seal::Ciphertext>>& enc_a2, int party,
                                      int threads, Utils::PROTO proto, int factor);

void generateConvTriplesCheetahPhase2(IO::NetIO** ios, const gemini::HomConv2DSS& hom_conv,
                                      vector<vector<seal::Ciphertext>>& enc_A1,
                                      vector<vector<seal::Plaintext>>& enc_A2,
                                      vector<vector<vector<seal::Plaintext>>>& enc_B2,
                                      vector<Tensor<uint64_t>>& C,
                                      vector<vector<seal::Ciphertext>>& M, Utils::ConvParm parm,
                                      int party, int threads, Utils::PROTO proto, int factor);

void generateConvTriplesCheetahPhase3(IO::NetIO** ios, const gemini::HomConv2DSS& hom_conv,
                                      vector<vector<seal::Ciphertext>>& M, uint32_t* c,
                                      vector<Tensor<uint64_t>>& C, Utils::ConvParm parm, int party,
                                      int threads, Utils::PROTO proto, int factor);

void generateConvTriplesCheetah(IO::NetIO** ios, const uint32_t* a, const uint32_t* b, uint32_t* c,
                                const gemini::HomConv2DSS::Meta& meta, int batch, int party,
                                int threads, Utils::PROTO proto, int factor);

void generateBNTriplesCheetah(IO::NetIO** ios, const uint32_t* a, const uint32_t* b, uint32_t* c,
                              int batch, size_t num_ele, size_t h, size_t w, int party, int threads,
                              Utils::PROTO proto, int factor = 1);

void tmp(int party, int threads);

} // namespace Iface

template <class Channel>
gemini::HomConv2DSS Iface::setupConv(Channel** ios, int party) {
    gemini::HomConv2DSS conv;
    seal::SEALContext ctx = Utils::init_he_context();

    seal::KeyGenerator keygen(ctx);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    auto o_pkey          = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);
    exchange_keys(ios, *pkey, *o_pkey, ctx, party);

    conv.setUp(ctx, skey, o_pkey);
    return conv;
}

#endif