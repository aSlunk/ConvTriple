#include <iostream>
#include <memory>
#include <seal/seal.h>
#include <seal/util/uintcore.h>
#include <vector>

#include <gemini/cheetah/hom_conv2d_ss.h>
#include <gemini/cheetah/shape_inference.h>
#include <gemini/cheetah/tensor_encoder.h>
#include <gemini/cheetah/tensor_shape.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"
#include "proto.hpp"

using namespace gemini;

namespace {

// Result conv2d
//     [[maybe_unused]] (const IO::NetIO& io, const HomConv2DSS& conv, const HomConv2DSS::Meta&
//     META,
//                       const Tensor<uint64_t>& image, const std::vector<Tensor<uint64_t>>&
//                       filters) {
//     Result measures;
//
//     auto start = measure::now();
//     std::vector<seal::Serializable<seal::Ciphertext>> ctxts;
//     measures.ret        = conv.encryptImage(image, META, ctxts);
//     measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();
//     if (measures.ret != Code::OK)
//         return measures;
//
//     std::stringstream stream;
//     for (auto& cipher : ctxts) {
//         cipher.save(stream);
//     }
//
//     measures.bytes = stream.str().length();
//     std::cerr << ctxts.size() << "\n";
//     std::cerr << "send bytes: " << stream.str().length() << "\n";
//
//     std::vector<std::vector<seal::Plaintext>> p_filter;
//     measures.ret = conv.encodeFilters(filters, META, p_filter);
//     if (measures.ret != Code::OK)
//         return measures;
//
//     start = measure::now();
//     std::vector<seal::Ciphertext> result;
//     Tensor<uint64_t> out;
//     measures.ret
//         = conv.conv2DSS(ctxts, std::vector<seal::Plaintext>(), p_filter, META, result, out);
//     measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();
//     if (measures.ret != Code::OK)
//         return measures;
//
//     stream.clear();
//     for (auto& cipher : result) {
//         cipher.save(stream);
//     }
//
//     measures.bytes += stream.str().length();
//     std::cerr << "sending result: " << stream.str().length() << "\n";
//
//     start = measure::now();
//     Tensor<uint64_t> out_tensor;
//     measures.ret        = conv.decryptToTensor(result, META, out_tensor);
//     measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();
//
//     return measures;
// }

Result conv2D_online(const HomConv2DSS::Meta& meta, IO::NetIO& server,
                     const seal::SEALContext& context, const HomConv2DSS& conv,
                     const Tensor<uint64_t>& A1, const std::vector<Tensor<uint64_t>>& B1,
                     const size_t& threads = 1) {
    Result measures;

    auto start = measure::now();
    std::vector<seal::Serializable<seal::Ciphertext>> enc_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Ciphertext>> enc_B1;
    measures.ret = conv.encryptFilters(B1, meta, enc_B1, threads);
    if (measures.ret != Code::OK)
        return measures;
    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();
    IO::send_encrypted_vector(server, enc_A1);
    IO::send_encrypted_filters(server, enc_B1);

    ////////////////////////////////////////////////////////////////////////////
    // A ⊙ B
    ////////////////////////////////////////////////////////////////////////////
    std::vector<seal::Ciphertext> result;
    IO::recv_encrypted_vector(server, context, result);
    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // Dec(M)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();
    Tensor<uint64_t> out_tensor;
    measures.ret        = conv.decryptToTensor(result, meta, out_tensor, threads);
    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    if (measures.ret != Code::OK)
        return measures;

    std::cerr << out_tensor.channels() << " x " << out_tensor.height() << " x "
              << out_tensor.width() << "\n";

    ////////////////////////////////////////////////////////////////////////////
    // A ⊙ B + Dec(M)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();
    Tensor<uint64_t> AB;
    conv.idealFunctionality(A1, B1, meta, AB);

    Utils::op_inplace<uint64_t>(AB, out_tensor, [](uint64_t a, uint64_t b) { return a + b; });
    measures.plain_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    measures.bytes = server.counter;
    measures.ret   = Code::OK;
    return measures;
}

Result perform_proto(HomConv2DSS::Meta& meta, IO::NetIO& server, const seal::SEALContext& context,
                     const HomConv2DSS& hom_conv, const size_t& threads = 1) {
    auto A1 = Utils::init_image(meta, 5);
    auto B1 = Utils::init_filter(meta, 2.0);

    Tensor<uint64_t> R(HomConv2DSS::GetConv2DOutShape(meta));
    R.Randomize(1000 << filter_prec);
    // for (int i = 0; i < R.channels(); ++i)
    //     for (int j = 0; j < R.height(); ++j)
    //         for (int k = 0; k < R.width(); ++k) R(i, j, k) = 1ULL << filter_prec;

    server.sync();
#if PROTO == 2
    auto measures = Server::Protocol2(meta, server, context, hom_conv, A1, B1, threads);
#elif PROTO == 3
    auto measures = Server::Protocol3(meta, server, context, hom_conv, A1, threads);
#endif
    server.counter = 0;
    return measures;
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc != 5 && argc != 4) {
        std::cout << argv[0] << " <port> <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
    }

    seal::SEALContext context = Utils::init_he_context();

    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);

    HomConv2DSS conv;
    conv.setUp(context, skey, pkey);

    int port   = strtol(argv[1], NULL, 10);
    // IO::NetIO server(nullptr, port, true);

    double total_time = 0;
    double total_data = 0;

    int threads;
    if (argc == 3)
        threads = N_THREADS;
    else
        threads = strtol(argv[4], NULL, 10);

    int samples   = strtol(argv[2], NULL, 10);
    int batchSize = strtol(argv[3], NULL, 10);
    std::cerr << "Samples: " << samples << "\n";
    std::cerr << "batchSize: " << batchSize << "\n";
    std::cerr << "threads: " << threads << "\n";
    std::vector<Result> results(samples);

    auto layers = Utils::init_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cerr << "Current layer: " << i << std::endl;

        for (int round = 0; round < samples; ++round) {
            size_t batch_threads = 1;
            ThreadPool tpool(batch_threads);
            std::vector<Result> batches_results(batch_threads);
            auto batch = [&] (long wid, size_t start, size_t end) -> Code {
                IO::NetIO server(nullptr, port + wid, true);
                for (size_t cur = start; cur < end; ++cur) {
                    auto result = (perform_proto(layers[i], server, context, conv, threads / batch_threads));
                    if (result.ret != Code::OK)
                        return result.ret;

                    Utils::add_result(batches_results[wid], result);
                }
                return Code::OK;
            };

            auto code = gemini::LaunchWorks(tpool, batchSize, batch);
            if (code != Code::OK) {
                std::cerr << CodeMessage(code) << "\n";
                return EXEC_FAILED;
            }
            results[round] = average(batches_results);
        }

        auto res = average(results);
        total_time += print_results(res, i, batchSize, threads);
        total_data += res.bytes / 1'000'000.0;
    }

    total_data /= 1'000.0;

    std::cout << "Party 1: total time [s]: " << total_time << "\n";
    std::cout << "Party 1: total data [GB]: " << total_data << "\n";
}
