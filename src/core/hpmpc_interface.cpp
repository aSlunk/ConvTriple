#include "hpmpc_interface.hpp"

#include <algorithm>

#include "protocols/fc_proto.hpp"
#include "protocols/ot_proto.hpp"

constexpr uint64_t MAX_ARITH = 20'000'000;

namespace Iface {

void generateBoolTriplesCheetah(uint8_t a[], uint8_t b[], uint8_t c[],
                                int bitlength [[maybe_unused]], uint64_t num_triples,
                                std::string ip, int port, int party, int threads) {
    const char* addr = ip.c_str();
    if (party == emp::ALICE)
        addr = nullptr;

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads);
    sci::OTPack<IO::NetIO> ot_pack(ios, threads, party, true, false);
    TripleGenerator<IO::NetIO> triple_gen(party, ios[0], &ot_pack);

    switch (party) {
    case emp::ALICE:
        Server::triple_gen(triple_gen, a, b, c, num_triples, false);
        break;
    case emp::BOB:
        Client::triple_gen(triple_gen, a, b, c, num_triples, false);
        break;
    }

    for (int i = 0; i < threads; ++i) delete ios[i];
    delete[] ios;
}

void setUpBn(IO::NetIO** ios, gemini::HomBNSS& bn, const seal::SEALContext& ctx, const int& party) {
    using namespace seal;
    KeyGenerator keygen(ctx);
    SecretKey skey = keygen.secret_key();
    auto pkey      = std::make_shared<PublicKey>();
    auto o_pkey    = std::make_shared<PublicKey>();
    keygen.create_public_key(*pkey);
    exchange_keys(ios, *pkey, *o_pkey, ctx, party);

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

    Code code;
    code = bn.setUp(PLAIN_MOD, ctx, skey, o_pkey);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party), ": ", CodeMessage(code));
    code = bn.setUp(PLAIN_MOD, contexts, opt_sks, bn_pks_);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party), ": ", CodeMessage(code));
}

void generateArithTriplesCheetah(uint32_t a[], uint32_t b[], uint32_t c[],
                                 int bitlength [[maybe_unused]], uint64_t num_triples,
                                 std::string ip, int port, int party, int threads) {
    const char* addr = ip.c_str();
    if (party == emp::ALICE)
        addr = nullptr;

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads);

    seal::SEALContext ctx = Utils::init_he_context();
    gemini::HomBNSS bn;

    setUpBn(ios, bn, ctx, party);

    for (size_t total = num_triples; total > 0;) {
        size_t current = std::min(MAX_ARITH, total);

        gemini::HomBNSS::Meta meta;
        meta.is_shared_input = true;
        meta.vec_shape       = {static_cast<long>(current)};
        meta.target_base_mod = PLAIN_MOD;

        Tensor<uint64_t> A({static_cast<long>(std::min(MAX_ARITH, current))});
        Tensor<uint64_t> B({static_cast<long>(std::min(MAX_ARITH, current))});
        Tensor<uint64_t> C({static_cast<long>(std::min(MAX_ARITH, current))});

        A.Randomize();
        B.Randomize();

        for (uint64_t i = 0; i < current; ++i) {
            // A(i) = static_cast<uint64_t>(a[i]);
            // B(i) = static_cast<uint64_t>(b[i]);
            A(i) %= PLAIN_MOD;
            B(i) %= PLAIN_MOD;
            a[i + num_triples - total] = A(i);
            b[i + num_triples - total] = B(i);
        }

        switch (party) {
        case emp::ALICE: {
            Server::perform_elem(ios, bn, meta, A, B, C, threads);
            break;
        }
        case emp::BOB: {
            Client::perform_elem(ios, bn, meta, A, B, C, threads);
            break;
        }
        default: {
            Utils::log(Utils::Level::ERROR, "Unknown party: P", party);
        }
        }

        for (uint64_t i = 0; i < current; ++i)
            c[i + num_triples - total] = static_cast<uint32_t>(C(i));
        total -= current;
    }

    for (int i = 0; i < threads; ++i) delete ios[i];

    delete[] ios;
}

void generateFCTriplesCheetah(uint64_t num_triples, int party, std::string ip, int port) {
    int batch        = 1;
    const char* addr = ip.c_str();

    if (party == emp::ALICE) {
        addr = nullptr;
    }

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, 1);

    auto meta = Utils::init_meta_fc(num_triples, 1);
    gemini::HomFCSS fc;
    seal::SEALContext ctx = Utils::init_he_context();
    gemini::HomBNSS bn;

    seal::KeyGenerator keygen(ctx);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    auto o_pkey          = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);
    exchange_keys(ios, *pkey, *o_pkey, ctx, party);

    fc.setUp(ctx, skey, o_pkey);

    std::vector<Tensor<uint64_t>> A(batch, Tensor<uint64_t>(meta.input_shape));
    std::vector<Tensor<uint64_t>> B(batch, Tensor<uint64_t>(meta.weight_shape));

    for (int i = 0; i < batch; ++i) {
        A[i].Randomize(PLAIN_MOD);
        B[i].Randomize(PLAIN_MOD);
    }

    std::vector<Tensor<uint64_t>> C(batch);

    switch (party) {
    case emp::ALICE: {
        Server::perform_proto(meta, ios, ctx, fc, A, B, C, 1ul, batch);
        break;
    }
    case emp::BOB: {
        Client::perform_proto(meta, ios, ctx, fc, A, B, C, 1ul, batch);
        break;
    }
    }
}

} // namespace Iface