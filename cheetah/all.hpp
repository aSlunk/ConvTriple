#ifndef ALL_HPP
#define ALL_HPP

#include <memory>

#include <seal/seal.h>

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
    gemini::HomConv2DSS conv;
    gemini::HomFCSS fc;

    std::unique_ptr<sci::OTPack<Channel>> ot_pack;
    std::unique_ptr<TripleGenerator<Channel>> triple_gen;

    seal::SEALContext context = Utils::init_he_context();

    size_t threads;
    size_t batchSize;
    size_t samples;

    Channel** ios_c;
    std::vector<std::vector<Channel>> ios_vec;

    int party;

    void exchange_keys(const seal::PublicKey& pkey, seal::PublicKey& o_pkey);

    inline size_t counter() {
        size_t counter = 0;
        for (size_t i = 0; i < threads; ++i) counter += ios_c[i]->counter;
        return counter;
    }

    inline void reset_counter() {
        for (size_t i = 0; i < threads; ++i) ios_c[i]->counter = 0;
    }

  public:
    explicit HE(const int& party, const char* addr, const int& port, const size_t& threads,
                const size_t& batchSize, size_t& samples);

    HE(const HE& other) = delete;
    HE(HE&& other)      = delete;

    HE& operator=(const HE& other) = delete;
    HE& operator=(HE&& other)      = delete;

    ~HE() { delete[] ios_c; }

    const gemini::HomConv2DSS& get_conv() const { return conv; }
    const gemini::HomFCSS& get_fc() const { return fc; }

    template <class T>
    void run_he(std::vector<class T::Meta>& layers, const T& conv);

    void run_ot(const size_t& batchSize, bool packed = false);
};

template <class Channel>
HE<Channel>::HE(const int& party, const char* addr, const int& port, const size_t& threads,
                const size_t& batchSize, size_t& samples)
    : threads(threads), batchSize(batchSize), samples(samples), party(party) {
    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads / batch_threads;

    ios_vec = Utils::init_ios<Channel>(addr, port, batch_threads, threads_per_thread);
    ios_c   = new Channel*[threads];

    for (size_t i = 0; i < batch_threads; ++i)
        for (size_t j = 0; j < threads_per_thread; ++j)
            ios_c[i * threads_per_thread + j] = &ios_vec[i][j];

    auto start = measure::now();
    ot_pack    = std::make_unique<sci::OTPack<Channel>>(ios_c, threads, party);
    triple_gen = std::make_unique<TripleGenerator<Channel>>(party, ios_c[0], ot_pack.get());
    Utils::log(Utils::Level::INFO, "P", party, ": OT startup time: ", Utils::to_sec(Utils::time_diff(start)));

    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    auto o_pkey          = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);
    exchange_keys(*pkey, *o_pkey);

    conv.setUp(context, skey, o_pkey);
    fc.setUp(context, skey, o_pkey);

    reset_counter();
}

template <class Channel>
void HE<Channel>::exchange_keys(const seal::PublicKey& pkey, seal::PublicKey& o_pkey) {
    switch (party) {
    case emp::ALICE:
        IO::send_pkey(ios_vec[0][0], pkey);
        IO::recv_pkey(ios_vec[0][0], context, o_pkey);
        break;
    case emp::BOB:
        IO::recv_pkey(ios_vec[0][0], context, o_pkey);
        IO::send_pkey(ios_vec[0][0], pkey);
        break;
    }
}

template <class Channel>
void HE<Channel>::run_ot(const size_t& batchSize, bool packed) {
    auto start = measure::now();
    for (size_t i = 0; i < samples; ++i) {
        switch (party) {
        case emp::ALICE:
            Server::RunGen(*triple_gen, batchSize, packed);
            break;
        case emp::BOB:
            Client::RunGen(*triple_gen, batchSize, packed);
            break;
        }
    }
    Utils::log(Utils::Level::INFO, "P", party,
                ": OT TIME[s]: ", Utils::to_sec(Utils::time_diff(start)) / samples);
    Utils::log(Utils::Level::INFO, "P", party,
                ": OT data[MB]: ", (Utils::to_MB(counter())) / samples);

    reset_counter();
}

template <class Channel>
template <class T>
void HE<Channel>::run_he(std::vector<class T::Meta>& layers, const T& conv) {
    double total      = 0;
    double total_data = 0;

    std::vector<Result> results(samples);           // all samples
    std::vector<Result> all_results(layers.size()); // averaged samples

    for (size_t i = 0; i < layers.size(); ++i) {
        ios_vec[0][0].sync();
        Utils::log(Utils::Level::DEBUG, "Current layer: ", i);
        double tmp_total = 0;

        for (size_t round = 0; round < samples; ++round) {
            gemini::ThreadPool tpool(ios_vec.size());
            std::vector<Result> batches_results(ios_vec.size());
            auto batch = [&](long wid, size_t start, size_t end) -> Code {
                auto& ios = ios_vec[wid];
                for (size_t cur = start; cur < end; ++cur) {
                    Result result;
                    if ((PROTO == 2 && party == emp::ALICE)
                        || (PROTO == 1 && (cur + party - 1) % 2 == 0)) {
                        result = Server::perform_proto(layers[i], ios, context, conv, ios.size());
                    } else {
                        result = Client::perform_proto(layers[i], ios, context, conv, ios.size());
                    }

                    if (result.ret != Code::OK)
                        return result.ret;

                    Utils::add_result(batches_results[wid], result);
                }
                return Code::OK;
            };

            auto start = measure::now();
            auto code  = gemini::LaunchWorks(tpool, batchSize, batch);
            total += Utils::to_sec(Utils::time_diff(start));
            if (code != Code::OK)
                Utils::log(Utils::Level::ERROR, CodeMessage(code));

            results[round] = Utils::average(batches_results, false);
        }

        total += tmp_total / samples;

        all_results[i] = Utils::average(results, true);
        total_data += Utils::to_MB(all_results[i].bytes);
    }

    switch (party) {
    case emp::ALICE:
        Utils::make_csv(all_results, batchSize, threads, "server.csv");
        break;
    case emp::BOB:
        Utils::make_csv(all_results, batchSize, threads, "client.csv");
        break;
    }
    std::cout << "Party " << party << ": total time [s]: " << total << "\n";
    std::cout << "Party " << party << ": total data [GB]: " << total_data << "\n";

    reset_counter();
}

} // namespace HE_OT

#endif