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

using namespace gemini;
namespace {

Result conv2d
    [[maybe_unused]] (const IO::NetIO& io, const HomConv2DSS& conv, const HomConv2DSS::Meta& META,
                      const Tensor<uint64_t>& image, const std::vector<Tensor<uint64_t>>& filters) {
    Result measures;

    auto start = measure::now();
    std::vector<seal::Ciphertext> ctxts;
    measures.ret        = conv.encryptImage(image, META, ctxts);
    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    if (measures.ret != Code::OK)
        return measures;

    std::stringstream stream;
    for (auto& cipher : ctxts) {
        cipher.save(stream);
    }

    measures.bytes = stream.str().length();
    std::cerr << ctxts.size() << "\n";
    std::cerr << "send bytes: " << stream.str().length() << "\n";

    std::vector<std::vector<seal::Plaintext>> p_filter;
    measures.ret = conv.encodeFilters(filters, META, p_filter);
    if (measures.ret != Code::OK)
        return measures;

    start = measure::now();
    std::vector<seal::Ciphertext> result;
    Tensor<uint64_t> out;
    measures.ret
        = conv.conv2DSS(ctxts, std::vector<seal::Plaintext>(), p_filter, META, result, out);
    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    if (measures.ret != Code::OK)
        return measures;

    stream.clear();
    for (auto& cipher : result) {
        cipher.save(stream);
    }

    measures.bytes += stream.str().length();
    std::cerr << "sending result: " << stream.str().length() << "\n";

    start = measure::now();
    Tensor<uint64_t> out_tensor;
    measures.ret        = conv.decryptToTensor(result, META, out_tensor);
    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    return measures;
}

Result conv2D_online3(HomConv2DSS::Meta& meta, IO::NetIO& server, const seal::SEALContext& context,
                      const HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                      const size_t& threads = 1) {

    meta.is_shared_input = true;
    Result measures;
    measures.plain_op  = 0;
    measures.cipher_op = 0;

    auto start = measure::now();

    std::vector<seal::Ciphertext> enc_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();
    IO::send_encrypted_vector(server, enc_A1);
    measures.send_recv = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    enc_A1.clear();

    start = measure::now();
    IO::recv_encrypted_vector(server, context, enc_A1);
    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();
    Tensor<uint64_t> C1;
    measures.ret        = conv.decryptToTensor(enc_A1, meta, C1, threads);
    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    std::cerr << C1.channels() << " x " << C1.height() << " x " << C1.width() << "\n";

    measures.bytes = server.counter;
    return measures;
}

Result conv2D_online2(HomConv2DSS::Meta& meta, IO::NetIO& server, const seal::SEALContext& context,
                      const HomConv2DSS& conv, const Tensor<uint64_t>& A1,
                      const std::vector<Tensor<uint64_t>>& B1, const size_t& threads = 1) {
    meta.is_shared_input = true;

    Result measures;
    measures.send_recv = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Enc(A1), enc(B1), send(A1), recv(A2)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();
    std::vector<seal::Ciphertext> enc_A1;
    std::vector<seal::Plaintext> encoded_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, encoded_A1, threads);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Plaintext>> enc_B1;
    measures.ret = conv.encodeFilters(B1, meta, enc_B1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    IO::send_encrypted_vector(server, enc_A1);
    std::vector<seal::Ciphertext> enc_A2;
    IO::recv_encrypted_vector(server, context, enc_A2);

    measures.send_recv = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // M1 = (A1 + A2') ⊙ B1 - R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    // conv.add_plain_inplace(enc_A2, encoded_A1);
    std::vector<seal::Ciphertext> M1;
    Tensor<uint64_t> R1;
    measures.ret = conv.conv2DSS(enc_A2, encoded_A1, enc_B1, meta, M1, R1, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // Send(M1), Recv(M2), Dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    IO::send_encrypted_vector(server, M1);
    std::vector<seal::Ciphertext> enc_M2;
    IO::recv_encrypted_vector(server, context, enc_M2);

    measures.send_recv += std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // Dec(M2) + R1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    Tensor<uint64_t> M2;
    measures.ret = conv.decryptToTensor(enc_M2, meta, M2, threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    start = measure::now();

    Utils::op_inplace<uint64_t>(M2, R1, [](uint64_t a, uint64_t b) { return a - b; });

    measures.plain_op = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    std::cerr << M2.channels() << " x " << M2.height() << " x " << M2.width() << "\n";

    measures.bytes = server.counter;
    measures.ret   = Code::OK;
    return measures;
}

Result conv2D_online(const HomConv2DSS::Meta& meta, IO::NetIO& server,
                     const seal::SEALContext& context, const HomConv2DSS& conv,
                     const Tensor<uint64_t>& A1, const std::vector<Tensor<uint64_t>>& B1,
                     const size_t& threads = 1) {
    Result measures;

    auto start = measure::now();
    std::vector<seal::Ciphertext> enc_A1;
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
    auto measures = conv2D_online2(meta, server, context, hom_conv, A1, B1, threads);
#elif PROTO == 3
    auto measures = conv2D_online3(meta, server, context, hom_conv, A1, threads);
#endif
    server.counter = 0;
    return measures;
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 3) {
        std::cout << argv[0] << " <samples> <batchSize> (<threads>)\n";
        return EXEC_FAILED;
    }

    seal::SEALContext context = Utils::init_he_context();

    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);

    HomConv2DSS conv;
    conv.setUp(context, skey, pkey);

    IO::NetIO server(nullptr, PORT, true);

    double total_time = 0;
    double total_data = 0;

    int threads;
    if (argc == 3)
        threads = N_THREADS;
    else
        threads = strtol(argv[3], NULL, 10);

    int samples   = strtol(argv[1], NULL, 10);
    int batchSize = strtol(argv[2], NULL, 10);
    std::cerr << "Samples: " << samples << "\n";
    std::cerr << "batchSize: " << batchSize << "\n";
    std::cerr << "threads: " << threads << "\n";
    std::vector<Result> results(samples);

    auto layers = Utils::init_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cerr << "Current layer: " << i << std::endl;

        for (int round = 0; round < samples; ++round) {
            results[round].encryption = 0;
            results[round].cipher_op  = 0;
            results[round].plain_op   = 0;
            results[round].decryption = 0;
            results[round].send_recv  = 0;
            results[round].bytes      = 0;
            results[round].ret        = Code::OK;
            for (int batch = 0; batch < batchSize; ++batch) {
                auto cur = (perform_proto(layers[i], server, context, conv, threads));
                if (cur.ret != Code::OK) {
                    std::cerr << CodeMessage(cur.ret) << "\n";
                    return EXEC_FAILED;
                }

                results[round].encryption += cur.encryption;
                results[round].cipher_op += cur.cipher_op;
                results[round].plain_op += cur.plain_op;
                results[round].decryption += cur.decryption;
                results[round].send_recv += cur.send_recv;
                results[round].bytes += cur.bytes;
            }
        }

        auto res = average(results);
        total_time += print_results(res, i, batchSize, threads);
        total_data += res.bytes / 1'000'000.0;
    }

    total_data /= 1'000.0;

    std::cout << "Party 1: total time [s]: " << total_time << "\n";
    std::cout << "Party 1: total data [GB]: " << total_data << "\n";
}
