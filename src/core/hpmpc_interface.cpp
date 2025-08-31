#include "hpmpc_interface.hpp"

#include <algorithm>

#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"
#include "protocols/ot_proto.hpp"

#include "ot/bit-triple-generator.h"
#include "ot/cheetah-ot_pack.h"

constexpr uint64_t MAX_BOOL  = 20'000'000;
constexpr uint64_t MAX_ARITH = 20'000'000;

namespace Iface {

class PROF : public seal::MMProf {
    std::unique_ptr<seal::MemoryPoolHandle> handle;
    std::shared_ptr<seal::util::MemoryPoolMT> pool;

  public:
    PROF() {
        pool   = std::make_shared<seal::util::MemoryPoolMT>(true);
        handle = std::make_unique<seal::MemoryPoolHandle>(pool);
    }

    ~PROF() noexcept {
        handle.release();
        if (pool.unique()) {
            std::cout << "UNIQUE\n";
        } else {
            std::cout << "NOT UNIQUE: " << pool.use_count() << "\n";
        }
    }

    seal::MemoryPoolHandle get_pool(uint64_t) { return *handle; }
};

void generateBoolTriplesCheetah(uint8_t a[], uint8_t b[], uint8_t c[],
                                int bitlength [[maybe_unused]], uint64_t num_triples,
                                std::string ip, int port, int party, int threads) {
    const char* addr = ip.c_str();
    if (party == emp::ALICE)
        addr = nullptr;

    auto start = measure::now();

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads);
    static sci::OTPack<IO::NetIO> ot_pack(ios, threads, party, true, false);
    static TripleGenerator<IO::NetIO> triple_gen(party, ios[0], &ot_pack);

    for (size_t total = 0; total < num_triples;) {
        int current = std::min(num_triples - total, MAX_BOOL);
        switch (party) {
        case emp::ALICE:
            Server::triple_gen(triple_gen, a + total, b + total, c + total, current, true);
            break;
        case emp::BOB:
            Client::triple_gen(triple_gen, a + total, b + total, c + total, current, true);
            break;
        }
        total += current;
    }

    Utils::log(Utils::Level::INFO, "P", party,
               ": Bool triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    uint64_t data = Utils::to_MB(ios[0]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party, ": Bool triple data[", unit, "]: ", data);

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
                                 std::string ip, int port, int party, int threads,
                                 Utils::PROTO proto) {
    {
        const char* addr = ip.c_str();
        if (party == emp::ALICE)
            addr = nullptr;

        IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads);

        static gemini::HomBNSS bn = [&ios, &party] {
            gemini::HomBNSS bn;
            auto ctx = Utils::init_he_context();
            setUpBn(ios, bn, ctx, party);
            return bn;
        }();

        auto pool = seal::MemoryPoolHandle::New();
        auto pg   = seal::MMProfGuard(std::make_unique<seal::MMProfFixed>(std::move(pool)));

        Tensor<uint64_t> A({static_cast<long>(num_triples)});
        Tensor<uint64_t> B({static_cast<long>(num_triples)});

        for (uint64_t i = 0; i < num_triples; ++i) {
            A(i) = static_cast<uint64_t>(a[i]);
            B(i) = static_cast<uint64_t>(b[i]);
        }

        auto start = measure::now();

        gemini::HomBNSS::Meta meta;
        meta.is_shared_input = true;
        meta.target_base_mod = PLAIN_MOD;

        for (size_t total = 0; total < num_triples;) {
            size_t current = std::min(MAX_ARITH, num_triples - total);

            meta.vec_shape = {static_cast<long>(current)};

            Tensor<uint64_t> tmp_A = Tensor<uint64_t>::Wrap(A.data() + total, meta.vec_shape);
            Tensor<uint64_t> tmp_B = Tensor<uint64_t>::Wrap(B.data() + total, meta.vec_shape);
            Tensor<uint64_t> tmp_C(meta.vec_shape);

            Result res;
            switch (party) {
            case emp::ALICE: {
                res = Server::perform_elem(ios, bn, meta, tmp_A, tmp_B, tmp_C, threads, proto);
                break;
            }
            case emp::BOB: {
                res = Client::perform_elem(ios, bn, meta, tmp_A, tmp_B, tmp_C, threads, proto);
                break;
            }
            default: {
                Utils::log(Utils::Level::ERROR, "Unknown party: P", party);
            }
            }

            for (uint64_t i = 0; i < current; ++i) c[i + total] = static_cast<uint32_t>(tmp_C(i));
            total += current;
        }

        Utils::log(Utils::Level::INFO, "P", party,
                   ": Arith triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
        std::string unit;
        uint64_t data = Utils::to_MB(ios[0]->counter, unit);
        Utils::log(Utils::Level::INFO, "P", party, ": Arith triple data[", unit, "]: ", data);

        for (int i = 0; i < threads; ++i) delete ios[i];

        delete[] ios;
    }
}

void generateFCTriplesCheetah(uint64_t num_triples, int party, std::string ip, int port,
                              Utils::PROTO proto) {
    int batch        = 1;
    const char* addr = ip.c_str();

    if (party == emp::ALICE) {
        addr = nullptr;
    }

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, 1);

    auto meta                 = Utils::init_meta_fc(num_triples, 1);
    static gemini::HomFCSS fc = [&ios, &party] {
        gemini::HomFCSS fc;
        seal::SEALContext ctx = Utils::init_he_context();

        seal::KeyGenerator keygen(ctx);
        seal::SecretKey skey = keygen.secret_key();
        auto pkey            = std::make_shared<seal::PublicKey>();
        auto o_pkey          = std::make_shared<seal::PublicKey>();
        keygen.create_public_key(*pkey);
        exchange_keys(ios, *pkey, *o_pkey, ctx, party);

        fc.setUp(ctx, skey, o_pkey);
        return fc;
    }();

    std::vector<Tensor<uint64_t>> A(batch, Tensor<uint64_t>(meta.input_shape));
    std::vector<Tensor<uint64_t>> B(batch, Tensor<uint64_t>(meta.weight_shape));

    // for (int i = 0; i < batch; ++i) {
    //     A[i].Randomize(PLAIN_MOD);
    //     B[i].Randomize(PLAIN_MOD);
    // }

    std::vector<Tensor<uint64_t>> C(batch);

    switch (party) {
    case emp::ALICE: {
        Server::perform_proto(meta, ios, fc, A, B, C, 1ul, batch, proto);
        break;
    }
    case emp::BOB: {
        Client::perform_proto(meta, ios, fc, A, B, C, 1ul, batch, proto);
        break;
    }
    }
}

void generateConvTriplesCheetah(const ConvParm& parm, int batch, std::string ip, int port,
                                int party, Utils::PROTO proto) {
    int threads      = 1;
    const char* addr = ip.c_str();

    if (party == emp::ALICE) {
        addr = nullptr;
    }

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads);

    auto meta = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                      parm.n_filters, parm.stride, parm.padding);
    static gemini::HomConv2DSS conv = [&ios, &party] {
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
    }();

    Tensor<uint64_t> A(meta.ishape);
    std::vector<Tensor<uint64_t>> B(meta.n_filters, Tensor<uint64_t>(meta.fshape));

    Tensor<uint64_t> C(gemini::GetConv2DOutShape(meta));

    switch (party) {
    case emp::ALICE: {
        Server::perform_proto(meta, ios, conv, A, B, C, threads, proto);
        break;
    }
    case emp::BOB: {
        Client::perform_proto(meta, ios, conv, A, B, C, threads, proto);
        break;
    }
    }
}

// void tmp(int party) {
//     // auto context = Utils::init_he_context();
//     seal::EncryptionParameters parms(seal::scheme_type::bfv);
//     parms.set_poly_modulus_degree(POLY_MOD);
//     parms.set_coeff_modulus(seal::CoeffModulus::Create(POLY_MOD, {60, 60, 60, 49}));
//     size_t prime_mod = seal::PlainModulus::Batching(POLY_MOD, 20).value();
//     parms.set_plain_modulus(prime_mod);
//     seal::SEALContext context(parms, true, seal::sec_level_type::none);
//
//     auto io = Utils::init_ios<IO::NetIO>(party == emp::ALICE ? nullptr : "127.0.0.1", 6969, 1);
//
//     seal::KeyGenerator keygen(context);
//     seal::SecretKey skey = keygen.secret_key();
//     auto pkey            = std::make_shared<seal::PublicKey>();
//     auto o_pkey          = std::make_shared<seal::PublicKey>();
//     keygen.create_public_key(*pkey);
//     exchange_keys(io, *pkey, *o_pkey, context, party);
//
//     uint64_t num_triples = 20'000'000;
//
//     std::vector<uint64_t> A(num_triples);
//     std::vector<uint64_t> B(num_triples);
//     std::vector<uint64_t> C(num_triples);
//
//     seal::Encryptor enc(context, *o_pkey);
//     seal::Decryptor dec(context, skey);
//
//     elemwise_product(&context, io[0], &enc, &dec, num_triples, A, B, C, prime_mod, party);
//     std::cout << "okay\n";
//     std::cout << io[0]->counter << "\n";
// }

} // namespace Iface