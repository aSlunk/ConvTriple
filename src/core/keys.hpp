
#ifndef KEYS_HPP_
#define KEYS_HPP_

#include "core/utils.hpp"
#include "io/send.hpp"

#include <iostream>

namespace Iface {

// not thread safe
class Keys {
  public:
    template <class Channel>
    static Keys& instance(Channel** ios, int party) {
        static Keys k(ios, party);
        return k;
    }

    gemini::HomFCSS& get_fc() { return _fc; }
    gemini::HomBNSS& get_bn() { return _bn; }
    gemini::HomConv2DSS& get_conv() { return _hom_conv; }

  private:
    gemini::HomFCSS _fc;
    gemini::HomConv2DSS _hom_conv;
    gemini::HomBNSS _bn;

    template <class Channel>
    Keys(Channel** ios, int party) {
        auto start = measure::now();

        seal::SEALContext ctx = Utils::init_he_context();

        seal::KeyGenerator keygen(ctx);
        seal::SecretKey skey = keygen.secret_key();
        auto pkey            = std::make_shared<seal::PublicKey>();
        auto o_pkey          = std::make_shared<seal::PublicKey>();
        keygen.create_public_key(*pkey);
        exchange_keys(ios, *pkey, *o_pkey, ctx, party);

        _fc.setUp(ctx, skey, o_pkey);
        _hom_conv.setUp(ctx, skey, o_pkey);
        _bn.setUp(PLAIN_MOD, ctx, skey, o_pkey);
        setupBn(ios, ctx, party);

        auto time = Utils::to_sec(Utils::time_diff(start));
        Utils::log(Utils::Level::INFO, "P", party - 1, ": Keyexchange took [s]: ", time);
    };

    ~Keys() noexcept {}

    template <class Channel, class SerKey>
    void exchange_keys(Channel** ios, const SerKey& pkey, seal::PublicKey& o_pkey,
                       const seal::SEALContext& ctx, int party);

    template <class Channel>
    void setupBn(Channel** ios, const seal::SEALContext& ctx, const int& party);

  public:
    Keys(Keys& copy)             = delete;
    Keys(Keys&& copy)            = delete;
    Keys& operator=(Keys& copy)  = delete;
    Keys& operator=(Keys&& copy) = delete;
};

template <class Channel, class SerKey>
void Keys::exchange_keys(Channel** ios, const SerKey& pkey, seal::PublicKey& o_pkey,
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

template <class Channel>
void Keys::setupBn(Channel** ios, const seal::SEALContext& ctx, const int& party) {
    using namespace seal;
    KeyGenerator keygen(ctx);

    size_t ntarget_bits = BIT_LEN;
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

    auto code = _bn.setUp(PLAIN_MOD, contexts, opt_sks, bn_pks_);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", party - 1, ": ", CodeMessage(code));
}
} // namespace Iface

#endif
