#include "hpmpc_interface.hpp"

#include <algorithm>

#include "protocols/bn_direct_proto.hpp"
#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"
#include "protocols/ot_proto.hpp"

#include "ot/bit-triple-generator.h"
#include "ot/cheetah-ot_pack.h"

constexpr uint64_t MAX_BOOL  = 1'000'000;
constexpr uint64_t MAX_ARITH = 20'000'000;

#define OTHER_PARTY(party) (3 - party)

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
                                std::string ip, int port, int party, int threads,
                                TripleGenMethod method) {
    Utils::log(Utils::Level::INFO, "P", party, ": num_triples (BOOL): ", num_triples);
    const char* addr = ip.c_str();
    if (party == emp::ALICE)
        addr = nullptr;

    auto start = measure::now();

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads);

    std::vector<sci::OTPack<IO::NetIO>*> ot_packs(threads);
    for (int i = 0; i < threads; ++i)
        ot_packs[i] = new sci::OTPack<IO::NetIO>(ios + i, 1, i & 1 ? OTHER_PARTY(party) : party,
                                                 true, false);

    auto func = [&](int wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;

        int cur_party = wid & 1 ? OTHER_PARTY(party): party;
        TripleGenerator<IO::NetIO> triple_gen(cur_party, ios[wid], ot_packs[wid], false);

        for (int total = start; total < end;) {
            int current = std::min(end - total, static_cast<int>(MAX_BOOL / threads));
            switch (cur_party) {
            case emp::ALICE:
                Server::triple_gen(triple_gen, a + total, b + total, c + total, current, true,
                                   method);
                break;
            case emp::BOB:
                Client::triple_gen(triple_gen, a + total, b + total, c + total, current, true,
                                   method);
                break;
            }
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_triples, func);

    Utils::log(Utils::Level::INFO, "P", party,
               ": Bool triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party, ": Bool triple data[", unit, "]: ", data);

    for (int i = 0; i < threads; ++i) {
        delete ot_packs[i];
        delete ios[i];
    }
    delete[] ios;
}

void setupBn(IO::NetIO** ios, gemini::HomBNSS& bn, const seal::SEALContext& ctx, const int& party) {
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

void generateArithTriplesCheetah(const uint32_t a[], const uint32_t b[], uint32_t c[],
                                 int bitlength [[maybe_unused]], uint64_t num_triples,
                                 std::string ip, int port, int party, int threads,
                                 Utils::PROTO proto) {
    Utils::log(Utils::Level::INFO, "P", party, ": num_triples (ARITH): ", num_triples,
               " " + Utils::proto_str(proto));
    const char* addr = ip.c_str();
    if (party == emp::ALICE)
        addr = nullptr;

    auto start = measure::now();

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads);

    static gemini::HomBNSS bn = [&ios, &party] {
        gemini::HomBNSS bn;
        auto ctx = Utils::init_he_context();
        setupBn(ios, bn, ctx, party);
        return bn;
    }();

    auto pool = seal::MemoryPoolHandle::New();
    auto pg   = seal::MMProfGuard(std::make_unique<seal::MMProfFixed>(std::move(pool)));

    Tensor<uint64_t> A({static_cast<long>(num_triples)});
    Tensor<uint64_t> B({static_cast<long>(num_triples)});

    for (uint64_t i = 0; i < num_triples; ++i) {
        if (a)
            A(i) = static_cast<uint64_t>(a[i]);
        if (b)
            B(i) = static_cast<uint64_t>(b[i]);
    }

    gemini::HomBNSS::Meta meta;
    meta.is_shared_input = proto == Utils::PROTO::AB;
    meta.target_base_mod = PLAIN_MOD;

    auto func = [&](size_t wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;
        for (int total = start; total < end;) {
            size_t current = std::min(static_cast<int>(MAX_ARITH), end - total);

            gemini::HomBNSS::Meta m = meta;
            m.vec_shape = gemini::TensorShape({static_cast<long>(current)});

            Tensor<uint64_t> tmp_A = Tensor<uint64_t>::Wrap(A.data() + total, m.vec_shape);
            Tensor<uint64_t> tmp_B = Tensor<uint64_t>::Wrap(B.data() + total, m.vec_shape);
            Tensor<uint64_t> tmp_C(m.vec_shape);

            Result res;
            switch (party) {
            case emp::ALICE: {
                res = Server::perform_elem(ios + wid, bn, m, tmp_A, tmp_B, tmp_C, 1, proto);
                break;
            }
            case emp::BOB: {
                res = Client::perform_elem(ios + wid, bn, m, tmp_A, tmp_B, tmp_C, 1, proto);
                break;
            }
            default: {
                Utils::log(Utils::Level::ERROR, "Unknown party: P", party);
            }
            }

            for (uint64_t i = 0; i < current; ++i) c[i + total] = static_cast<uint32_t>(tmp_C(i));
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_triples, func);

    Utils::log(Utils::Level::INFO, "P", party,
               ": Arith triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party, ": Arith triple data[", unit, "]: ", data);

    for (int i = 0; i < 1; ++i) delete ios[i];
    delete[] ios;
}

void generateFCTriplesCheetah(const uint32_t* a, const uint32_t* b, uint32_t* c, int batch,
                              uint64_t com_dim, uint64_t dim2, int party, std::string ip, int port,
                              int threads, Utils::PROTO proto, int factor) {
    const char* addr = ip.c_str();

    if (party == emp::ALICE) {
        addr = nullptr;
    }

    auto meta = Utils::init_meta_fc(com_dim, dim2);
    Utils::log(Utils::Level::INFO, "P", party, " FC: ", meta.input_shape, " x ", meta.weight_shape,
               " ", Utils::proto_str(proto));

    auto start = measure::now();

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, 1);

    // meta.is_shared_input = proto == Utils::PROTO::AB;
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

    uint64_t* ai = new uint64_t[meta.input_shape.num_elements() * batch];
    for (uint i = 0; i < meta.input_shape.num_elements() * batch; ++i)
        ai[i] = a != nullptr ? a[i] : 0;
    std::vector<Tensor<uint64_t>> A(batch);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = Tensor<uint64_t>::Wrap(ai + meta.input_shape.num_elements() * i, meta.input_shape);

    uint64_t* bi = new uint64_t[meta.weight_shape.num_elements() * factor];
    for (uint i = 0; i < meta.weight_shape.num_elements() * factor; ++i)
        bi[i] = b == nullptr ? 0 : b[i];

    size_t tmp = batch / factor;
    std::vector<Tensor<uint64_t>> B(batch);
    for (int i = 0; i < factor; ++i)
        for (size_t j = 0; j < tmp; ++j)
            B[i * tmp + j] = Tensor<uint64_t>::Wrap(
                bi + meta.weight_shape.num_elements() * (i % factor), meta.weight_shape);

    std::vector<Tensor<uint64_t>> C(batch);

    switch (party) {
    case emp::ALICE: {
        Client::perform_proto(meta, ios, fc, A, B, C, threads, batch, proto);
        break;
    }
    case emp::BOB: {
        Server::perform_proto(meta, ios, fc, A, B, C, threads, batch, proto);
        break;
    }
    }

    for (size_t i = 0; i < C.size(); ++i)
        for (size_t j = 0; j < dim2; ++j) {
            c[i * dim2 + j] = C[i](j);
        }

    Utils::log(Utils::Level::INFO, "P", party,
               ": FC triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < 1; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party, ": FC triple data[", unit, "]: ", data);

    delete[] ai;
    delete[] bi;

    for (int i = 0; i < 1; ++i) delete ios[i];
    delete[] ios;
}

void generateConvTriplesCheetahWrapper(const uint32_t* a, const uint32_t* b, uint32_t* c,
                                       Utils::ConvParm parm, int batch, std::string ip, int port,
                                       int party, int threads, Utils::PROTO proto, int factor) {
    auto meta = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                      parm.n_filters, parm.stride, parm.padding);

    Utils::log(Utils::Level::INFO, "P", party, " CONV: ", meta.ishape, " x ", meta.fshape, " x ", parm.n_filters, ", ",
               parm.stride, ", ", parm.padding, ", ", Utils::proto_str(proto));

    meta.is_shared_input = proto == Utils::PROTO::AB;
    if (Utils::getOutDim(parm) == gemini::GetConv2DOutShape(meta)) {
        generateConvTriplesCheetah(a, b, c, meta, batch, ip, port, party, threads, proto, factor);
    } else {
        Utils::log(Utils::Level::INFO, "Adding padding manually");

        std::vector<uint32_t> ai;
        std::tuple<int, int> dim;

        dim = Utils::pad_zero(a, ai, parm.ic, parm.ih, parm.iw, parm.padding, batch);

        parm.ih      = std::get<0>(dim);
        parm.iw      = std::get<1>(dim);
        parm.padding = 0;

        meta = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                     parm.n_filters, parm.stride, parm.padding);
        generateConvTriplesCheetah(ai.data(), b, c, meta, batch, ip, port, party, threads, proto,
                                   factor);
    }
}

void generateConvTriplesCheetah(const uint32_t* a, const uint32_t* b, uint32_t* c,
                                const gemini::HomConv2DSS::Meta& meta, int batch, std::string ip,
                                int port, int party, int threads, Utils::PROTO proto, int factor) {
    const char* addr = ip.c_str();

    if (party == emp::ALICE) {
        addr = nullptr;
    }

    auto start = measure::now();

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, 1);

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

    uint64_t* ai = new uint64_t[meta.ishape.num_elements() * batch];
    for (long i = 0; i < meta.ishape.num_elements() * batch; ++i) ai[i] = a != nullptr ? a[i] : 0;

    uint64_t* bi = new uint64_t[meta.fshape.num_elements() * meta.n_filters * factor];
    if (b)
        for (size_t i = 0; i < meta.fshape.num_elements() * meta.n_filters * factor; ++i)
            bi[i] = b[i];

    int ac_batch_size = batch / factor;
    for (int cur_batch = 0; cur_batch < batch; ++cur_batch) {
        Tensor<uint64_t> A
            = Tensor<uint64_t>::Wrap(ai + meta.ishape.num_elements() * cur_batch, meta.ishape);

        std::vector<Tensor<uint64_t>> B(meta.n_filters);
        for (size_t i = 0; i < meta.n_filters; ++i)
            B[i] = Tensor<uint64_t>::Wrap(
                bi + meta.fshape.num_elements() * meta.n_filters * (cur_batch / ac_batch_size)
                    + meta.fshape.num_elements() * i,
                meta.fshape);

        Tensor<uint64_t> C;

        switch (party) {
        case emp::ALICE: {
            Client::perform_proto(meta, ios, conv, A, B, C, threads, proto);
            break;
        }
        case emp::BOB: {
            Server::perform_proto(meta, ios, conv, A, B, C, threads, proto);
            break;
        }
        }
        for (long i = 0; i < C.NumElements(); ++i) c[i + C.NumElements() * cur_batch] = C.data()[i];
    }

    Utils::log(Utils::Level::INFO, "P", party,
               ": CONV triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < 1; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party, ": CONV triple data[", unit, "]: ", data);

    delete[] ai;
    delete[] bi;

    for (int i = 0; i < 1; ++i) delete ios[i];
    delete[] ios;
}

void generateBNTriplesCheetah(const uint32_t* a, const uint32_t* b, uint32_t* c, int batch,
                              size_t num_ele, size_t h, size_t w, std::string ip, int port,
                              int party, int threads, Utils::PROTO proto, int factor) {
    const char* addr = ip.c_str();

    if (party == emp::ALICE) {
        addr = nullptr;
    }
    auto meta = Utils::init_meta_bn(num_ele, h, w);
    Utils::log(Utils::Level::INFO, "P", party, " BN: ", meta.ishape, " x ", meta.vec_shape, ", ",
               Utils::proto_str(proto));

    auto start = measure::now();

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, 1);

    meta.is_shared_input      = proto == Utils::PROTO::AB;
    static gemini::HomBNSS bn = [&ios, &party] {
        gemini::HomBNSS bn;
        seal::SEALContext ctx = Utils::init_he_context();
        setupBn(ios, bn, ctx, party);
        return bn;
    }();

    size_t ac_batch_size = batch / factor;
    for (int cur_batch = 0; cur_batch < batch; ++cur_batch) {
        Tensor<uint64_t> A(meta.ishape);
        for (long i = 0; i < A.channels(); i++)
            for (long j = 0; j < A.height(); j++)
                for (long k = 0; k < A.width(); k++)
                    A(i, j, k) = a != nullptr ? a[meta.ishape.num_elements() * cur_batch
                                                  + i * A.height() * A.width() + j * A.width() + k]
                                              : 0;

        Tensor<uint64_t> B(meta.vec_shape);
        for (long i = 0; i < B.NumElements(); i++)
            B(i) = b != nullptr ? b[i + B.NumElements() * (cur_batch / ac_batch_size)] : 0;

        Tensor<uint64_t> C;

        switch (party) {
        case emp::ALICE: {
            Client::perform_proto(meta, ios, bn, A, B, C, threads, proto);
            break;
        }
        case emp::BOB: {
            Server::perform_proto(meta, ios, bn, A, B, C, threads, proto);
            break;
        }
        }

        for (long i = 0; i < C.channels(); i++)
            for (long j = 0; j < C.height(); j++)
                for (long k = 0; k < C.width(); k++)
                    c[C.NumElements() * cur_batch + i * C.height() * C.width() + j * C.width() + k]
                        = C(i, j, k);
    }

    Utils::log(Utils::Level::INFO, "P", party,
               ": BN triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < 1; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party, ": BN triple data[", unit, "]: ", data);

    for (int i = 0; i < 1; ++i) delete ios[i];
    delete[] ios;
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