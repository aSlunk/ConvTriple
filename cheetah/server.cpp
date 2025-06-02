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

struct Result {
    size_t encryption;
    size_t step2;
    size_t step3;
    size_t decryption;
    size_t bytes;
    Code ret;
};

void print_results(const Result& res, const bool& header = false) {
    if (header)
        std::cout
            << "Encryption [ms],Step 2 [s],Decryption [ms],Step 3 [ms],Total [s],Bytes Send [MB]\n";

    double total = res.encryption / 1'000.0 + res.step2 / 1'000.0 + res.decryption / 1'000.0
                   + res.step3 / 1'000.0;
    total /= 1000.0;
    std::cout << res.encryption / 1'000.0 << ", " << res.step2 / 1'000'000. << ", "
              << res.decryption / 1'000. << ", " << res.step3 / 1'000.0 << ", " << total << ", "
              << res.bytes / 1'000'000.0 << "\n";
}

Result conv2d
    [[maybe_unused]] (const IO::NetIO& io, const HomConv2DSS& conv, const Tensor<uint64_t>& image,
                      const std::vector<Tensor<uint64_t>>& filters) {
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
    measures.step2 = std::chrono::duration_cast<Unit>(measure::now() - start).count();
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

Result conv2D_online(const HomConv2DSS::Meta& meta, IO::NetIO& server,
                     const seal::SEALContext& context, const HomConv2DSS& conv,
                     const Tensor<uint64_t>& A1, const std::vector<Tensor<uint64_t>>& B1) {
    Result measures;

    auto start = measure::now();
    std::vector<seal::Ciphertext> enc_A1;
    measures.ret = conv.encryptImage(A1, meta, enc_A1, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<std::vector<seal::Ciphertext>> enc_B1;
    measures.ret = conv.encryptFilters(B1, meta, enc_B1, N_THREADS);
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
    measures.step2 = std::chrono::duration_cast<Unit>(measure::now() - start).count();

    ////////////////////////////////////////////////////////////////////////////
    // Dec(M)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();
    Tensor<uint64_t> out_tensor;
    measures.ret        = conv.decryptToTensor(result, meta, out_tensor, N_THREADS);
    measures.decryption = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    if (measures.ret != Code::OK)
        return measures;

    ////////////////////////////////////////////////////////////////////////////
    // A ⊙ B + Dec(M)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();
    Tensor<uint64_t> AB;
    conv.idealFunctionality(A1, B1, meta, AB);

    Utils::op_inplace<uint64_t>(AB, out_tensor, [](uint64_t a, uint64_t b) { return a + b; });
    measures.step3 = std::chrono::duration_cast<Unit>(measure::now() - start).count();
    // auto f64_out = Utils::convert_double(out_tensor);
    // std::cout << "result:\n";
    // Utils::print_tensor(f64_out);

    measures.bytes = server.counter;
    measures.ret   = Code::OK;
    return measures;
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc < 2)
        return EXEC_FAILED;

    seal::SEALContext context = Utils::init_he_context();

    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);

    HomConv2DSS conv;
    conv.setUp(context, skey, pkey);

    Utils::print_info();

    auto A1 = Utils::init_image(META, 5);
    auto B1 = Utils::init_filter(META, 3.0);

    size_t samples = std::strtoul(argv[1], NULL, 10);

    // std::cout << "Image:\n";
    // Utils::print_tensor(Utils::convert_double(input_img));

    IO::NetIO server(nullptr, PORT, true);

    for (size_t i = 0; i < samples; ++i) {
        server.sync();
        auto res = conv2D_online(META, server, context, conv, A1, B1);
        if (res.ret != Code::OK) {
            std::cerr << CodeMessage(res.ret) << "\n";
            return EXEC_FAILED;
        }
        print_results(res, i == 0);
        server.counter = 0;
    }
}
