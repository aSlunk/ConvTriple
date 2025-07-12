#ifndef ALL_HPP
#define ALL_HPP

#include <memory>

#include <seal/seal.h>

#include <gemini/cheetah/hom_bn_ss.h>
#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/hom_fc_ss.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include <ot/cheetah-ot_pack.h>

#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"
#include "protocols/ot_proto.hpp"

#include "defs.hpp"

using Utils::Result;

namespace HE_OT {

template <class Channel>
class HE {
  private:
    gemini::HomBNSS bn_;
    gemini::HomConv2DSS conv_;
    gemini::HomFCSS fc_;

    std::unique_ptr<sci::OTPack<Channel>> ot_pack_;
    std::unique_ptr<TripleGenerator<Channel>> triple_gen_;

    seal::SEALContext context_ = Utils::init_he_context();
    std::unique_ptr<seal::Encryptor> encryptor_;

    size_t threads_;
    size_t batchSize_;
    size_t samples_;

    Channel** ios_c_;
    std::vector<std::vector<Channel>> ios_vec_;

    int party_;

    void exchange_keys_(const seal::PublicKey& pkey, seal::PublicKey& o_pkey);

    inline size_t counter_() {
        size_t counter = 0;
        for (size_t i = 0; i < threads_; ++i) counter += ios_c_[i]->counter;
        return counter;
    }

    inline void reset_counter_() {
        for (size_t i = 0; i < threads_; ++i) ios_c_[i]->counter = 0;
    }

  public:
    explicit HE(const int& party, const char* addr, const int& port, const size_t& threads,
                const size_t& batchSize, size_t& samples);

    HE(const HE& other) = delete;
    HE(HE&& other)      = delete;

    HE& operator=(const HE& other) = delete;
    HE& operator=(HE&& other)      = delete;

    ~HE() { delete[] ios_c_; }

    const gemini::HomConv2DSS& get_conv() const { return conv_; }
    const gemini::HomFCSS& get_fc() const { return fc_; }

    template <class T>
    void run_he(std::vector<class T::Meta>& layers, const T& conv);

    void run_ot(const size_t& batchSize, bool packed = false);

    void test_bn();
    void alt_bn();

    void test() {
        std::stringstream ss;
        for (size_t i = 0; i < 2048; ++i) {
            seal::Ciphertext ct;
            encryptor_->encrypt_symmetric(seal::Plaintext(10), ct);
            ct.save(ss);
        }
        std::cerr << "Total: " << Utils::to_MB(ss.str().size()) << "\n";
    }
};

template <class Channel>
HE<Channel>::HE(const int& party, const char* addr, const int& port, const size_t& threads,
                const size_t& batchSize, size_t& samples)
    : threads_(threads), batchSize_(batchSize), samples_(samples), party_(party) {
    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads / batch_threads;

    ios_vec_ = Utils::init_ios<Channel>(addr, port, batch_threads, threads_per_thread);
    ios_c_   = new Channel*[threads];

    for (size_t i = 0; i < batch_threads; ++i)
        for (size_t j = 0; j < threads_per_thread; ++j)
            ios_c_[i * threads_per_thread + j] = &ios_vec_[i][j];

    auto start  = measure::now();
    ot_pack_    = std::make_unique<sci::OTPack<Channel>>(ios_c_, threads, party);
    triple_gen_ = std::make_unique<TripleGenerator<Channel>>(party, ios_c_[0], ot_pack_.get());
    Utils::log(Utils::Level::INFO, "P", party,
               ": OT startup time: ", Utils::to_sec(Utils::time_diff(start)));

    seal::KeyGenerator keygen(context_);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    auto o_pkey          = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);
    exchange_keys_(*pkey, *o_pkey);

    conv_.setUp(context_, skey, o_pkey);
    fc_.setUp(context_, skey, pkey);
    bn_.setUp(PLAIN_MOD, context_, skey, pkey);

    encryptor_ = std::make_unique<seal::Encryptor>(context_, skey);

    reset_counter_();
}

template <class Channel>
void HE<Channel>::exchange_keys_(const seal::PublicKey& pkey, seal::PublicKey& o_pkey) {
    switch (party_) {
    case emp::ALICE:
        IO::send_pkey(ios_vec_[0][0], pkey);
        IO::recv_pkey(ios_vec_[0][0], context_, o_pkey);
        break;
    case emp::BOB:
        IO::recv_pkey(ios_vec_[0][0], context_, o_pkey);
        IO::send_pkey(ios_vec_[0][0], pkey);
        break;
    }
}

template <class Channel>
void HE<Channel>::run_ot(const size_t& batchSize, bool packed) {
    auto start = measure::now();
    for (size_t i = 0; i < samples_; ++i) {
        switch (party_) {
        case emp::ALICE:
            Server::RunGen(*triple_gen_, batchSize, packed);
            break;
        case emp::BOB:
            Client::RunGen(*triple_gen_, batchSize, packed);
            break;
        }
    }
    Utils::log(Utils::Level::INFO, "P", party_,
               ": OT TIME[s]: ", Utils::to_sec(Utils::time_diff(start)) / samples_);
    Utils::log(Utils::Level::INFO, "P", party_,
               ": OT data[MB]: ", (Utils::to_MB(counter_())) / samples_);

    reset_counter_();
}

template <class Channel>
template <class T>
void HE<Channel>::run_he(std::vector<class T::Meta>& layers, const T& conv) {
    double total      = 0;
    double total_data = 0;

    std::vector<Result> results(samples_);          // all samples
    std::vector<Result> all_results(layers.size()); // averaged samples

    for (size_t i = 0; i < layers.size(); ++i) {
        ios_vec_[0][0].sync();
        Utils::log(Utils::Level::DEBUG, "Current layer: ", i);
        double tmp_total = 0;

        for (size_t round = 0; round < samples_; ++round) {
            gemini::ThreadPool tpool(ios_vec_.size());
            std::vector<Result> batches_results(ios_vec_.size());
            auto batch = [&](long wid, size_t start, size_t end) -> Code {
                auto& ios = ios_vec_[wid];
                for (size_t cur = start; cur < end; ++cur) {
                    Result result;
                    if ((PROTO == 2 && party_ == emp::ALICE)
                        || (PROTO == 1 && (cur + party_ - 1) % 2 == 0)) {
                        result = Server::perform_proto(layers[i], ios, context_, conv, ios.size());
                    } else {
                        result = Client::perform_proto(layers[i], ios, context_, conv, ios.size());
                    }

                    if (result.ret != Code::OK)
                        return result.ret;

                    Utils::add_result(batches_results[wid], result);
                }
                return Code::OK;
            };

            auto start = measure::now();
            auto code  = gemini::LaunchWorks(tpool, batchSize_, batch);
            total += Utils::to_sec(Utils::time_diff(start));
            if (code != Code::OK)
                Utils::log(Utils::Level::ERROR, CodeMessage(code));

            results[round] = Utils::average(batches_results, false);
        }

        total += tmp_total / samples_;

        all_results[i] = Utils::average(results, true);
        total_data += Utils::to_MB(all_results[i].bytes);
    }

    switch (party_) {
    case emp::ALICE:
        Utils::make_csv(all_results, batchSize_, threads_, "server.csv");
        break;
    case emp::BOB:
        Utils::make_csv(all_results, batchSize_, threads_, "client.csv");
        break;
    }
    std::cout << "Party " << party_ << ": total time [s]: " << total << "\n";
    std::cout << "Party " << party_ << ": total data [MB]: " << total_data << "\n";

    reset_counter_();
}

template <class Channel>
void HE<Channel>::test_bn() {
    gemini::Tensor<uint64_t> image({2048, 49, 1});
    gemini::Tensor<uint64_t> scales({2048});
    for (long c = 0; c < image.channels(); ++c) {
        for (long r = 0; r < image.height(); ++r) {
            for (long i = 0; i < image.width(); ++i) {
                image(c, r, i) = r + 1;
            }
        }
    }

    for (long i = 0; i < scales.NumElements(); ++i) scales(i) = i + 1;

    gemini::HomBNSS::Meta meta;
    meta.ishape          = image.shape();
    meta.vec_shape       = scales.shape();
    meta.target_base_mod = PLAIN_MOD;
    meta.is_shared_input = false;

    std::vector<seal::Serializable<seal::Ciphertext>> enc_image;
    bn_.encryptTensor(image, meta, enc_image, threads_);

    std::vector<seal::Ciphertext> enc_images(enc_image.size());
    double total = 0;
    for (size_t i = 0; i < enc_image.size(); ++i) {
        std::stringstream ss;
        enc_image[i].save(ss);
        total += ss.tellp();
        enc_images[i].load(context_, ss);
    }
    std::cerr << "Total: " << Utils::to_MB(total) << "\n";

    gemini::Tensor<uint64_t> R;
    std::vector<seal::Ciphertext> out;
    Code code
        = bn_.bn_direct(enc_images, std::vector<seal::Plaintext>(), scales, meta, out, R, threads_);
    if (code != Code::OK) {
        std::cerr << CodeMessage(code) << "\n";
        return;
    }

    gemini::Tensor<uint64_t> out_tensor;
    bn_.decryptToTensor(out, meta, out_tensor, threads_);

    Utils::op_inplace<uint64_t>(
        out_tensor, R, [](uint64_t a, uint64_t b) -> uint64_t { return (a + b) & moduloMask; });

    gemini::Tensor<uint64_t> ideal;
    bn_.idealFunctionality(image, scales, meta, ideal);

    bool same = ideal.shape() == out_tensor.shape();
    for (long c = 0; c < image.channels(); ++c) {
        for (long h = 0; h < image.height(); ++h) {
            for (long w = 0; w < image.width(); ++w) {
                if (!same || out_tensor(c, h, w) != ideal(c, h, w)) {
                    same = false;
                    goto ret;
                }
            }
        }
    }
ret:
    if (same)
        Utils::log(Utils::Level::PASSED, "BN: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "BN: FAILED");
}

template <class Channel>
void HE<Channel>::alt_bn() {
    long rows            = 49;
    long channel         = 2048;
    auto meta            = Utils::init_meta_fc(1, rows);
    meta.is_shared_input = false;

    Tensor<uint64_t> scales({channel});
    Tensor<uint64_t> image({channel, rows});
    Tensor<uint64_t> res({channel, rows});

    std::atomic<uint64_t> total = 0;

    auto fc_prog = [&](long wid, size_t start, size_t end) -> Code {
        uint64_t local_tot = 0;
        for (size_t i = start; i < end; ++i) {
            Tensor<uint64_t> weight({meta.weight_shape});
            for (long j = 0; j < image.cols(); ++j) {
                image(i, j)  = j + 1;
                weight(j, 0) = image(i, j);
            }
            scales(i) = i + 1;

            Tensor scale({1});
            scale(0) = scales(i);

            std::vector<seal::Serializable<seal::Ciphertext>> enc_scale;
            fc_.encryptInputVector(scale, meta, enc_scale, 1);

            std::vector<seal::Ciphertext> enc_images(enc_scale.size());
            for (size_t j = 0; j < enc_scale.size(); ++j) {
                std::stringstream ss;
                enc_scale[j].save(ss);
                local_tot += ss.tellp();
                enc_images[j].load(context_, ss);
            }

            std::vector<std::vector<seal::Plaintext>> enc_weight;
            fc_.encodeWeightMatrix(weight, meta, enc_weight, 1);

            gemini::Tensor<uint64_t> R;
            std::vector<seal::Ciphertext> enc_out;
            fc_.MatVecMul(enc_images, std::vector<seal::Plaintext>(), enc_weight, meta, enc_out, R,
                          1);

            Tensor<uint64_t> out;
            fc_.decryptToVector(enc_out, meta, out, 1);

            Utils::op_inplace<uint64_t>(
                out, R, [](uint64_t a, uint64_t b) -> uint64_t { return (a + b) & moduloMask; });

            for (long j = 0; j < out.NumElements(); ++j) {
                res(i, j) = out(j);
            }
        }
        total.fetch_add(local_tot);
        return Code::OK;
    };
    gemini::ThreadPool tpool(threads_);
    gemini::LaunchWorks(tpool, channel, fc_prog);

    std::cout << "Total: " << Utils::to_MB(total) << "\n";

    Tensor<uint64_t> ideal({channel, rows});
    for (long i = 0; i < image.rows(); ++i) {
        for (long j = 0; j < image.cols(); ++j) {
            ideal(i, j)
                = seal::util::multiply_uint_mod(image(i, j), scales(i), fc_.plain_modulus());
        }
    }

    bool same = ideal.shape() == res.shape();
    for (long c = 0; c < image.rows(); ++c) {
        for (long h = 0; h < image.cols(); ++h) {
            if (!same || res(c, h) != ideal(c, h)) {
                same = false;
                goto ret;
            }
        }
    }
ret:
    if (same)
        Utils::log(Utils::Level::PASSED, "BN: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "BN: FAILED");
}

} // namespace HE_OT

#endif