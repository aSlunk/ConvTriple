#include <vector>

#include <gemini/cheetah/tensor.h>
#include <gemini/cheetah/tensor_encoder.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"

using namespace gemini;

namespace {

// constexpr char ADDRESS[11] = "127.0.0.1\0";

struct Result {
    size_t recv;
    size_t conv;
    size_t send;
    size_t bytes;
    Code ret;
};

void print_results(const Result& res, const bool& header = false, std::ostream& out = std::cout) {
    if (header)
        out << "Recv [ms],Conv2d [s],Send [ms],Total [s], Bytes Send [MB]\n";

    double total = res.recv / 1'000.0 + res.conv / 1'000.0 + res.send / 1'000.0;
    total /= 1'000.0;

    out << res.recv / 1'000.0 << ", " << res.conv / 1'000'000. << ", " << res.send / 1'000. << ", "
        << total << ", " << res.bytes / 1'000'000.0 << "\n";
}

Code add_inplace(const HomConv2DSS& hom, std::vector<RLWECt>& ciphers,
                 const Tensor<uint64_t>& tensor) {
    std::vector<RLWEPt> plains;
    Code res = hom.encodeImage(tensor, META, plains);
    if (res != Code::OK)
        return res;

    return hom.add_inplace(ciphers, plains);
}

Result Protocol(IO::NetIO& client, const seal::SEALContext& context, const HomConv2DSS& hom_conv,
                const Tensor<uint64_t>& A2, const std::vector<Tensor<uint64_t>>& B2,
                const Tensor<uint64_t>& R) {
    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // Receive enc(A1), enc(B1)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now(); // MEASURE_START

    std::vector<seal::Ciphertext> enc_A1;
    IO::recv_encrypted_vector(client, context, enc_A1);

    std::vector<std::vector<seal::Ciphertext>> enc_B1;
    IO::recv_encrypted_filters(client, context, enc_B1);

    measures.recv = std::chrono::duration_cast<Unit>(measure::now() - start).count(); // MEASURE_END

    ////////////////////////////////////////////////////////////////////////////
    // Encode A2, B2
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::vector<seal::Plaintext>> enc_B2;
    measures.ret = hom_conv.encodeFilters(B2, META, enc_B2, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<seal::Plaintext> enc_A2;
    measures.ret = hom_conv.encodeImage(A2, META, enc_A2, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    ////////////////////////////////////////////////////////////////////////////
    // A1 ⊙ B2 - R + A2 ⊙ B1
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now(); // MEASURE_START
    std::vector<seal::Ciphertext> result;
    Tensor<uint64_t> out;

    measures.ret = hom_conv.conv2DSS(enc_A1, enc_B2, META, R, result, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    std::vector<seal::Ciphertext> result2;
    measures.ret = hom_conv.conv2DSS(enc_A2, enc_B1, META, result2, N_THREADS);
    if (measures.ret != Code::OK)
        return measures;

    hom_conv.add_inplace(result, result2, N_THREADS);

    measures.conv = std::chrono::duration_cast<Unit>(measure::now() - start).count(); // MEASURE_END

    ////////////////////////////////////////////////////////////////////////////
    // Send result
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now(); // MEASURE_START
    IO::send_encrypted_vector(client, result);
    measures.send = std::chrono::duration_cast<Unit>(measure::now() - start).count(); // MEASURE END

    ////////////////////////////////////////////////////////////////////////////
    // A2 ⊙ B2 - R
    ////////////////////////////////////////////////////////////////////////////
    Tensor<uint64_t> final;
    hom_conv.idealFunctionality(A2, B2, META, final);
    Utils::op_inplace<uint64_t>(final, R, [](uint64_t a, uint64_t b) -> uint64_t { return a - b; });

    measures.bytes = client.counter;
    measures.ret   = Code::OK;
    return measures;
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        return EXEC_FAILED;
    }

    auto context = Utils::init_he_context();
    HomConv2DSS hom_conv;
    hom_conv.setUp(context);

    auto A2 = Utils::init_image(META, 5);
    auto B2 = Utils::init_filter(META, 2.0);

    // std::cout << "filter:\n";
    // Utils::print_tensor(Utils::convert_double(filters[0]));

    ////////////////////////////////////////////////////////////////////////////
    // Sample R
    ////////////////////////////////////////////////////////////////////////////
    Tensor<uint64_t> R(HomConv2DSS::GetConv2DOutShape(META));
    // R.Randomize(5 << filter_prec);
    for (int i = 0; i < R.channels(); ++i)
        for (int j = 0; j < R.height(); ++j)
            for (int k = 0; k < R.width(); ++k) R(i, j, k) = 1ULL << filter_prec;

    IO::NetIO client(argv[1], PORT, true);

    size_t samples = std::strtoul(argv[2], NULL, 10);
    for (size_t i = 0; i < samples; ++i) {
        client.sync();
        auto measures = Protocol(client, context, hom_conv, A2, B2, R);
        if (measures.ret != Code::OK) {
            std::cerr << CodeMessage(measures.ret) << "\n";
            return EXEC_FAILED;
        }
        print_results(measures, i == 0);
        client.counter = 0;
    }
}