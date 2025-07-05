#ifndef ALL_HPP
#define ALL_HPP

#include <seal/seal.h>

#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/hom_fc_ss.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"
#include "protocols/ot_proto.hpp"

#include "defs.hpp"

using Utils::Result;

namespace HE_OT {

class HE {
  private:
    gemini::HomConv2DSS conv;
    gemini::HomFCSS fc;

    seal::SEALContext context = Utils::init_he_context();

    size_t threads;
    size_t batchSize;
    size_t samples;

    IO::NetIO** ios_c;
    std::vector<std::vector<IO::NetIO>> ios_vec;

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

    ~HE() { delete[] ios_c; }

    gemini::HomConv2DSS& get_conv() { return conv; }
    gemini::HomFCSS& get_fc() { return fc; }

    template <class T>
    void run_he(std::vector<class T::Meta>& layers, const T& conv);

    void run_ot(const size_t& batchSize);
};

HE::HE(const int& party, const char* addr, const int& port, const size_t& threads,
       const size_t& batchSize, size_t& samples)
    : threads(threads), batchSize(batchSize), samples(samples), party(party) {
    size_t batch_threads      = batchSize > 1 ? batchSize : 1;
    size_t threads_per_thread = threads / batch_threads;

    ios_vec = Utils::init_ios<IO::NetIO>(addr, port, batch_threads, threads_per_thread);
    ios_c   = new IO::NetIO*[threads];

    for (size_t i = 0; i < batch_threads; ++i)
        for (size_t j = 0; j < threads_per_thread; ++j)
            ios_c[i * threads_per_thread + j] = &ios_vec[i][j];

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

void HE::exchange_keys(const seal::PublicKey& pkey, seal::PublicKey& o_pkey) {
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

void HE::run_ot(const size_t& batchSize) {
    auto start = measure::now();
    for (size_t i = 0; i < samples; ++i) {
        switch (party) {
        case emp::ALICE:
            Server::RunGen(ios_c, threads, batchSize);
            break;
        case emp::BOB:
            Client::RunGen(ios_c, threads, batchSize);
            break;
        }
    }
    Utils::log(Utils::Level::INFO,"P", party, ": OT TIME[s]: ", Utils::to_sec(Utils::time_diff(start) / samples));
    Utils::log(Utils::Level::INFO, "P", party, ": OT data[MB]: ", Utils::to_MB(static_cast<double>(counter()) / samples));

    reset_counter();
}

template <class T>
void HE::run_he(std::vector<class T::Meta>& layers, const T& conv) {
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
                    if (PROTO == 2 || cur % 2 == 0) {
                        switch (party) {
                        case emp::ALICE:
                            result = (Server::perform_proto(layers[i], ios, context, conv,
                                                            ios.size()));
                            break;
                        case emp::BOB:
                            result = (Client::perform_proto(layers[i], ios, context, conv,
                                                            ios.size()));
                            break;
                        }
                    } else {
                        switch (party) {
                        case emp::ALICE:
                            result = (Client::perform_proto(layers[i], ios, context, conv,
                                                            ios.size()));
                            break;
                        case emp::BOB:
                            result = (Server::perform_proto(layers[i], ios, context, conv,
                                                            ios.size()));
                            break;
                        }
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

    // total_data /= 1'000.0;

    switch (party) {
    case emp::ALICE:
        Utils::make_csv(all_results, batchSize, threads, "server.csv");
        break;
    case emp::BOB:
        Utils::make_csv(all_results, batchSize, threads, "client.csv");
        break;
    }
    std::cout << "Party 1: total time [s]: " << total << "\n";
    std::cout << "Party 1: total data [GB]: " << total_data << "\n";

    reset_counter();
}

} // namespace HE_OT

#endif