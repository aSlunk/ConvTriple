#ifndef HPMPC_INTERFACE_HPP_
#define HPMPC_INTERFACE_HPP_

#include "core/defs.hpp"
#include "protocols/bn_direct_proto.hpp"

#include <string>

#include <io/net_io_channel.hpp>

#include <ot/cheetah-ot_pack.h>
#include <ot/bit-triple-generator.h>

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

void generateTripleCheetah(uint8_t a[], uint8_t b[], uint8_t c[], int bitlength, uint64_t num_triples, std::string ip, int port, int party, int threads = 1) {
    const char* addr = ip.c_str();
    if (ip == "")
        addr = nullptr;

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads);
    sci::OTPack<IO::NetIO> ot_pack(ios, threads, party, true, false);
    TripleGenerator<IO::NetIO> triple_gen(party, ios[0], &ot_pack);

    triple_gen.generate(party, a, b, c, num_triples, TripleGenMethod::_2COT, false);

    for (int i = 0; i < threads; ++i)
        delete ios[i];

    delete[] ios;
}

template <class T>
void generateArithTripleCheetah(T a[], T b[], T c[], int bitlength, uint64_t num_triples, std::string ip, int port, int party, int threads = 1) {
    using namespace seal;
    const char* addr = ip.c_str();
    if (ip == "")
        addr = nullptr;

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads);
    
    SEALContext ctx = Utils::init_he_context();

    KeyGenerator keygen(ctx);
    SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<PublicKey>();
    auto o_pkey          = std::make_shared<PublicKey>();
    keygen.create_public_key(*pkey);
    exchange_keys(ios, *pkey, *o_pkey, ctx, party);

    size_t ntarget_bits = std::ceil(std::log2(PLAIN_MOD));
    size_t crt_bits     = 2 * ntarget_bits + 1 + gemini::HomBNSS::kStatBits;

    const size_t nbits_per_crt_plain = [](size_t crt_bits) {
        constexpr size_t kMaxCRTPrime = 50;
        for (size_t nCRT = 1;; ++nCRT) {
            size_t np = gemini::CeilDiv(crt_bits, nCRT);
            if (np <= kMaxCRTPrime)
                return np;
        }
    }(crt_bits + 1);

    const size_t nCRT = gemini::CeilDiv<size_t>(crt_bits, nbits_per_crt_plain);
    std::vector<int> crt_primes_bits(nCRT, nbits_per_crt_plain);

    const size_t N  = POLY_MOD;
    auto plain_crts = CoeffModulus::Create(N, crt_primes_bits);
    EncryptionParameters seal_parms(scheme_type::bfv);
    seal_parms.set_n_special_primes(0);
    // We are not exporting the pk/ct with more than 109-bit.
    std::vector<int> cipher_moduli_bits{60, 49};
    seal_parms.set_poly_modulus_degree(N);
    seal_parms.set_coeff_modulus(CoeffModulus::Create(N, cipher_moduli_bits));

    std::vector<std::shared_ptr<seal::SEALContext>> bn_contexts_(nCRT);
    for (size_t i = 0; i < nCRT; ++i) {
        seal_parms.set_plain_modulus(plain_crts[i]);
        bn_contexts_[i] = std::make_shared<SEALContext>(seal_parms, true, sec_level_type::tc128);
    }

    std::vector<seal::SEALContext> contexts;
    std::vector<std::optional<SecretKey>> opt_sks;

    std::vector<std::shared_ptr<seal::PublicKey>> bn_pks_(nCRT); // public keys (BN)
    std::vector<std::shared_ptr<seal::SecretKey>> bn_sks_(nCRT); // secret keys (BN)

    for (size_t i = 0; i < nCRT; ++i) {
        KeyGenerator keygen(*bn_contexts_[i]);
        bn_sks_[i]                   = std::make_shared<SecretKey>(keygen.secret_key());
        bn_pks_[i]                   = std::make_shared<PublicKey>();
        Serializable<PublicKey> s_pk = keygen.create_public_key();

        exchange_keys(ios, s_pk, *bn_pks_[i], *bn_contexts_[i], party);
        contexts.emplace_back(*bn_contexts_[i]);
        opt_sks.emplace_back(*bn_sks_[i]);
    }

    gemini::HomBNSS bn;

    Code code;
    code = bn.setUp(PLAIN_MOD, ctx, skey, o_pkey);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party), ": ", CodeMessage(code));
    code = bn.setUp(PLAIN_MOD, contexts, opt_sks, bn_pks_);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party), ": ", CodeMessage(code));


    gemini::HomBNSS::Meta meta;
    meta.is_shared_input = true;
    meta.vec_shape = {static_cast<long>(num_triples)};
    meta.target_base_mod = PLAIN_MOD;

    Tensor<uint64_t> A(meta.vec_shape);
    Tensor<uint64_t> B(meta.vec_shape);

    for (uint64_t i = 0; i < num_triples; ++i) {
        A(i) = static_cast<uint64_t>(a[i]);
        B(i) = static_cast<uint64_t>(b[i]);
    }

    Tensor<uint64_t> C(meta.vec_shape);

    switch (party) {
        case emp::ALICE: {
            Server::perform_elem(ios, ctx, bn, meta, A, B, C, threads);
            break;
        }
        case emp::BOB: {
            Client::perform_elem(ios, ctx, bn, meta, A, B, C, threads);
            break;
        }
        default: {
            Utils::log(Utils::Level::ERROR, "Unknown party: P", party);
        }
    }

    for (uint64_t i = 0; i < num_triples; ++i)
        c[i] = static_cast<T>(C(i));

    for (int i = 0; i < threads; ++i)
        delete ios[i];

    delete[] ios;
}

}

#endif