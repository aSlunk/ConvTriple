#ifndef CHEETAH_INTERFACE_HPP_
#define CHEETAH_INTERFACE_HPP_

#include <memory>

#include <seal/seal.h>

#include <gemini/cheetah/hom_bn_ss.h>
#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/hom_fc_ss.h>

#include <net_io_channel.hpp>
#include <send.hpp>

#include <ot/cheetah-ot_pack.h>

#include "protocols/bn_direct_proto.hpp"
#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"
#include "protocols/ot_proto.hpp"

#include "defs.hpp"

using Utils::Result;

namespace HE_OT {

template <class Channel>
class HE {
  public:
    explicit HE(const int& party, const char* addr, const int& port, const size_t& threads,
                size_t& samples, bool setup_ot = true, bool setup_bn = true);

    HE(const HE& other) = delete;
    HE(HE&& other)      = delete;

    HE& operator=(const HE& other) = delete;
    HE& operator=(HE&& other)      = delete;

    ~HE() { delete[] ios_c_; }

    const gemini::HomConv2DSS& get_conv() const { return conv_; }
    const gemini::HomFCSS& get_fc() const { return fc_; }
    const gemini::HomBNSS& get_bn() const { return bn_; }
    Channel** get_ios() const { return ios_c_; }

    template <class T>
    void test_he(std::vector<class T::Meta>& layers, const T& cheetah, const size_t& batchSize = 1);

    void run_conv(const Tensor<uint64_t>& A, const std::vector<Tensor<uint64_t>>& B,
                  Tensor<uint64_t>& C, const size_t& stride, const size_t& padding,
                  const size_t& batchSize = 1);

    void run_bn(const Tensor<uint64_t>& A, const Tensor<uint64_t>& B,
                           Tensor<uint64_t>& C,
                           const size_t& batchSize);

    template <class T, class Bs>
    void run_he(const T& cheetah, const class T::Meta& meta, const std::vector<Tensor<uint64_t>>& A, const std::vector<Bs>& B,
                  std::vector<Tensor<uint64_t>>& C,
                  const size_t& batchSize = 1);

    void run_ot(const size_t& numTriples, bool packed = false);

    double alt_bn(const gemini::HomBNSS::Meta& meta);

  private:
    gemini::HomBNSS bn_;
    gemini::HomConv2DSS conv_;
    gemini::HomFCSS fc_;

    std::unique_ptr<sci::OTPack<Channel>> ot_pack_;
    std::unique_ptr<TripleGenerator<Channel>> triple_gen_;

    seal::SEALContext context_ = Utils::init_he_context();
    std::vector<std::shared_ptr<seal::SEALContext>> bn_contexts_;
    std::vector<std::shared_ptr<seal::PublicKey>> bn_pks_;
    std::vector<std::shared_ptr<seal::SecretKey>> bn_sks_;
    std::unique_ptr<seal::Encryptor> encryptor_;

    size_t threads_;
    size_t samples_;

    std::vector<Channel> ios_;
    Channel** ios_c_;

    int party_;

    template <class SerKey>
    void exchange_keys_(const SerKey& pkey, seal::PublicKey& o_pkey, const seal::SEALContext& ctx);

    void setup_OT();
    void setup_BN(const std::optional<seal::SecretKey>& skey,
                  const std::shared_ptr<seal::PublicKey>& o_pkey);

    inline size_t counter_() {
        size_t counter = 0;
        for (size_t i = 0; i < threads_; ++i) counter += ios_c_[i]->counter;
        return counter;
    }

    inline void reset_counter_() {
        for (size_t i = 0; i < threads_; ++i) ios_c_[i]->counter = 0;
    }
};

template <class Channel>
HE<Channel>::HE(const int& party, const char* addr, const int& port, const size_t& threads,
                size_t& samples, bool setup_ot, bool setup_bn)
    : threads_(threads), samples_(samples), party_(party) {
    Code code;

    ios_   = Utils::init_ios<Channel>(addr, port, threads);
    ios_c_ = new Channel*[threads];

    for (size_t i = 0; i < threads; ++i) ios_c_[i] = &ios_[i];

    if (setup_ot)
        setup_OT();

    seal::KeyGenerator keygen(context_);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    auto o_pkey          = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);
    exchange_keys_(*pkey, *o_pkey, context_);

    code = conv_.setUp(context_, skey, o_pkey);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), ": ", CodeMessage(code));
    code = fc_.setUp(context_, skey, o_pkey);
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), ": ", CodeMessage(code));

    if (setup_bn)
        setup_BN(skey, o_pkey);

    encryptor_ = std::make_unique<seal::Encryptor>(context_, skey);

    reset_counter_();
}

template <class Channel>
void HE<Channel>::setup_BN(const std::optional<seal::SecretKey>& skey,
                           const std::shared_ptr<seal::PublicKey>& o_pkey) {
    using namespace seal;
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

    bn_contexts_.resize(nCRT);
    for (size_t i = 0; i < nCRT; ++i) {
        seal_parms.set_plain_modulus(plain_crts[i]);
        bn_contexts_[i] = std::make_shared<SEALContext>(seal_parms, true, sec_level_type::tc128);
    }

    std::vector<seal::SEALContext> contexts;
    std::vector<std::optional<SecretKey>> opt_sks;
    bn_sks_.resize(nCRT);
    bn_pks_.resize(nCRT);
    for (size_t i = 0; i < nCRT; ++i) {
        KeyGenerator keygen(*bn_contexts_[i]);
        bn_sks_[i]                   = std::make_shared<SecretKey>(keygen.secret_key());
        bn_pks_[i]                   = std::make_shared<PublicKey>();
        Serializable<PublicKey> s_pk = keygen.create_public_key();

        exchange_keys_(s_pk, *bn_pks_[i], *bn_contexts_[i]);
        contexts.emplace_back(*bn_contexts_[i]);
        opt_sks.emplace_back(*bn_sks_[i]);
    }

    Code code;
    if (party_ == 2) { // ALICE
        code = bn_.setUp(PLAIN_MOD, context_, skey, o_pkey);
        if (code != Code::OK)
            Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), ": ", CodeMessage(code));
        code = bn_.setUp(PLAIN_MOD, contexts, opt_sks, bn_pks_);
        if (code != Code::OK)
            Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), ": ", CodeMessage(code));
    } else { // BOB
        code = bn_.setUp(PLAIN_MOD, context_, skey, o_pkey);
        if (code != Code::OK)
            Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), ": ", CodeMessage(code));
        code = bn_.setUp(PLAIN_MOD, contexts, opt_sks, bn_pks_);
        if (code != Code::OK)
            Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), ": ", CodeMessage(code));
    }
}

template <class Channel>
void HE<Channel>::setup_OT() {
    std::string unit;
    auto start  = measure::now();
    ot_pack_    = std::make_unique<sci::OTPack<Channel>>(ios_c_, threads_, party_);
    triple_gen_ = std::make_unique<TripleGenerator<Channel>>(party_, ios_c_[0], ot_pack_.get());
    Utils::log(Utils::Level::INFO, "P", party_,
               ": OT startup time: ", Utils::to_sec(Utils::time_diff(start)));
    Utils::log(Utils::Level::INFO, "P", party_,
               ": OT startup data: ", Utils::to_MB(counter_(), unit), unit);
}

template <class Channel>
template <class SerKey>
void HE<Channel>::exchange_keys_(const SerKey& pkey, seal::PublicKey& o_pkey,
                                 const seal::SEALContext& ctx) {
    switch (party_) {
    case emp::ALICE:
        IO::send_pkey(*(ios_c_[0]), pkey);
        IO::recv_pkey(*(ios_c_[0]), ctx, o_pkey);
        break;
    case emp::BOB:
        IO::recv_pkey(*(ios_c_[0]), ctx, o_pkey);
        IO::send_pkey(*(ios_c_[0]), pkey);
        break;
    }
}

template <class Channel>
void HE<Channel>::run_ot(const size_t& numTriples, bool packed) {
    size_t len = numTriples / (packed ? 8 : 1);
    uint8_t* a = new uint8_t[len];
    uint8_t* b = new uint8_t[len];
    uint8_t* c = new uint8_t[len];

    Utils::log(Utils::Level::DEBUG, "Num Triples: ", numTriples);

    auto start = measure::now();
    for (size_t i = 0; i < samples_; ++i) {
        switch (party_) {
        case emp::ALICE:
            Server::triple_gen(*triple_gen_, a, b, c, numTriples, packed);
            break;
        case emp::BOB:
            Client::triple_gen(*triple_gen_, a, b, c, numTriples, packed);
            break;
        }
    }
    std::string unit;
    auto data = Utils::to_MB(counter_(), unit);
    Utils::log(Utils::Level::INFO, "P", party_,
               ": OT TIME[s]: ", Utils::to_sec(Utils::time_diff(start)) / samples_);
    Utils::log(Utils::Level::INFO, "P", party_, ": OT data[", unit,
               "]: ", (data / samples_));


    delete[] a;
    delete[] b;
    delete[] c;
    reset_counter_();
}

template <class Channel>
template <class T>
void HE<Channel>::test_he(std::vector<class T::Meta>& layers, const T& cheetah,
                          const size_t& batchSize) {
    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads_ / batch_threads;

    double total      = 0;
    double total_data = 0;
    std::string proto("AB");
    proto += PROTO > 1 ? std::to_string(PROTO) : "";

    std::vector<Result> results(samples_);          // all samples
    std::vector<Result> all_results(layers.size()); // averaged samples

    for (size_t i = 0; i < layers.size(); ++i) {
        ios_c_[0]->sync();
        Utils::log(Utils::Level::DEBUG, "Current layer: ", i);
        double tmp_total = 0;

        for (size_t round = 0; round < samples_; ++round) {
            std::vector<Result> batches_results(batch_threads);
            auto batch = [&](long wid, size_t start, size_t end) -> Code {
                for (size_t cur = start; cur < end; ++cur) {
                    Result result;
                    if ((PROTO == 2 && party_ == emp::ALICE)
                        || (PROTO == 1 && (cur + party_ - 1) % 2 == 0)) {
                        result = Server::perform_proto(layers[i], ios_c_ + wid * threads_per_thread,
                                                       context_, cheetah, threads_per_thread);
                    } else {
                        result = Client::perform_proto(layers[i], ios_c_ + wid * threads_per_thread,
                                                       context_, cheetah, threads_per_thread);
                    }

                    if (result.ret != Code::OK)
                        return result.ret;

                    Utils::add_result(batches_results[wid], result);
                }
                return Code::OK;
            };

            gemini::ThreadPool tpool(batch_threads);

            auto start = measure::now();
            auto code  = gemini::LaunchWorks(tpool, batchSize, batch);
            total += Utils::to_sec(Utils::time_diff(start));
            if (code != Code::OK)
                Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), " ",
                           CodeMessage(code));

            results[round] = Utils::average(batches_results, false);
        }

        total += tmp_total / samples_;

        all_results[i] = Utils::average(results, true);
        total_data += all_results[i].bytes;
    }

    switch (party_) {
    case emp::ALICE:
        Utils::make_csv(all_results, batchSize, threads_,
                        "party" + std::to_string(party_) + "_" + cheetah.get_str() + "_" + proto
                            + ".csv");
        break;
    case emp::BOB:
        Utils::make_csv(all_results, batchSize, threads_,
                        "party" + std::to_string(party_) + "_" + cheetah.get_str() + "_" + proto
                            + ".csv");
        break;
    }

    std::string unit;
    total_data = Utils::to_MB(total_data, unit);
    std::cout << "Party " << party_ << ": total time [s]: " << total << "\n";
    std::cout << "Party " << party_ << ": total data [" << unit << "]: " << total_data << "\n";

    reset_counter_();
}

template <class Channel>
double HE<Channel>::alt_bn(const gemini::HomBNSS::Meta& meta_bn) {
    auto meta = Utils::init_meta_fc(1, meta_bn.ishape.height());

    auto start = measure::now();

    Result res;
    switch (party_) {
    case emp::ALICE: {
        std::cerr << meta_bn.ishape.height() << " x " << meta_bn.vec_shape.num_elements() << "\n";
        res = Server::perform_proto(meta, ios_c_, context_, fc_, threads_,
                                    meta_bn.vec_shape.num_elements());
        break;
    }
    case emp::BOB: {
        res = Client::perform_proto(meta, ios_c_, context_, fc_, threads_,
                                    meta_bn.vec_shape.num_elements());
        break;
    }
    }

    double time = Utils::to_sec(Utils::time_diff(start));

    return time;
}

template <class Channel>
template <class T, class Bs>
void HE<Channel>::run_he(const T& cheetah, const class T::Meta& meta, const std::vector<Tensor<uint64_t>>& A, const std::vector<Bs>& B,
                  std::vector<Tensor<uint64_t>>& C,
                  const size_t& batchSize) {
    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads_ / batch_threads;

    double total      = 0;
    double total_data = 0;
    std::string proto("AB");
    proto += PROTO > 1 ? std::to_string(PROTO) : "";

    std::vector<Result> batches_results(batch_threads);
    auto batch = [&](long wid, size_t start, size_t end) -> Code {
        for (size_t cur = start; cur < end; ++cur) {
            Result result;
            if ((PROTO == 2 && party_ == emp::ALICE)
                || (PROTO == 1 && (cur + party_ - 1) % 2 == 0)) {
                result = Server::perform_proto(meta, ios_c_ + wid * threads_per_thread, context_,
                                               cheetah, A[cur], B[cur], C[cur], threads_per_thread);
            } else {
                result = Client::perform_proto(meta, ios_c_ + wid * threads_per_thread, context_,
                                               cheetah, A[cur], B[cur], C[cur], threads_per_thread);
            }

            if (result.ret != Code::OK)
                return result.ret;

            Utils::add_result(batches_results[wid], result);
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(batch_threads);

    auto start = measure::now();
    auto code  = gemini::LaunchWorks(tpool, batchSize, batch);
    total += Utils::to_sec(Utils::time_diff(start));
    if (code != Code::OK)
        Utils::log(Utils::Level::ERROR, "P", std::to_string(party_), " ", CodeMessage(code));

    std::string unit;
    total_data = Utils::to_MB(total_data, unit);
    std::cout << "Party " << party_ << ": total time [s]: " << total << "\n";
    std::cout << "Party " << party_ << ": total data [" << unit << "]: " << total_data << "\n";

    reset_counter_();
}

template <class Channel>
void HE<Channel>::run_conv(const Tensor<uint64_t>& A, const std::vector<Tensor<uint64_t>>& B,
                           Tensor<uint64_t>& C, const size_t& stride, const size_t& padding,
                           const size_t& batchSize) {
    gemini::HomConv2DSS::Meta meta
        = Utils::init_meta_conv(A.channels(), A.height(), A.width(), B[0].channels(), B[0].height(),
                                B[0].width(), B.size(), stride, padding);
    run_he(conv_, meta, A, B, C, batchSize);
}

template <class Channel>
void HE<Channel>::run_bn(const Tensor<uint64_t>& A, const Tensor<uint64_t>& B,
                           Tensor<uint64_t>& C,
                           const size_t& batchSize) {
    gemini::HomBNSS::Meta meta
        = Utils::init_meta_bn(B.NumElements(), A.shape().rows()* A.shape().rows());

    run_he(bn_, meta, A, B, C, batchSize);
}

} // namespace HE_OT

#endif