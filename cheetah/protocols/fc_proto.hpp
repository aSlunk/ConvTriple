#ifndef FC_PROTO_HPP
#define FC_PROTO_HPP

#include <gemini/cheetah/hom_fc_ss.h>
#include <gemini/cheetah/shape_inference.h>
#include <gemini/cheetah/tensor_encoder.h>
#include <gemini/cheetah/tensor_shape.h>

#include <io/net_io_channel.hpp>
#include <io/send.hpp>

#include "defs.hpp"

using gemini::HomFCSS;
using gemini::Tensor;
using Utils::Result;
using std::vector;

static uint64_t add(const HomFCSS& conv, const uint64_t& a, const uint64_t& b) {
    uint64_t sum;
    seal::util::add_uint(&a, 1, b, &sum);
    return seal::util::barrett_reduce_64(sum, conv.plain_modulus());
}

namespace Server {

template <class Channel>
Result Protocol2(const HomFCSS::Meta& meta, Channel** server, const seal::SEALContext& context,
                 const HomFCSS& conv, const vector<Tensor<uint64_t>>& A1, vector<Tensor<uint64_t>>& C1,
                 const size_t& threads = 1, const size_t& batch = 1);

template <class Channel>
Result Protocol1(const HomFCSS::Meta& meta, Channel** server, const seal::SEALContext& context,
                 const HomFCSS& conv, const vector<Tensor<uint64_t>>& A1, const vector<Tensor<uint64_t>>& B1,
                 vector<Tensor<uint64_t>>& C1, const size_t& threads = 1, const size_t& batch = 1);

template <class Channel>
Result perform_proto(HomFCSS::Meta& meta, Channel** server, const seal::SEALContext& context,
                     const HomFCSS& hom_conv, const size_t& threads = 1, const size_t& batch = 1);

#ifdef VERIFY
template <class T>
void Verify_Conv(IO::NetIO& io, const HomFCSS::Meta& meta, const HomFCSS& conv, const Tensor<T>& A1,
                 const Tensor<T>& B1, const Tensor<T>& C1);
#endif

} // namespace Server

namespace Client {

template <class Channel>
Result Protocol1(Channel** client, const seal::SEALContext& context, const HomFCSS& hom_conv,
                 const HomFCSS::Meta& meta, const vector<Tensor<uint64_t>>& A2, const vector<Tensor<uint64_t>>& B2,
                 vector<Tensor<uint64_t>>& C2, const size_t& threads = 1, const size_t& batch = 1);

template <class Channel>
Result Protocol2(Channel** client, const seal::SEALContext& context, const HomFCSS& hom_conv,
                 const HomFCSS::Meta& meta, const vector<Tensor<uint64_t>>& A2, const vector<Tensor<uint64_t>>& B2,
                 vector<Tensor<uint64_t>>& C2, const size_t& threads = 1, const size_t& batch = 1);

template <class Channel>
Result perform_proto(HomFCSS::Meta& meta, Channel** client, const seal::SEALContext& context,
                     const HomFCSS& hom_conv, const size_t& threads, const size_t& batch = 1);

#ifdef VERIFY
template <class T>
void Verify_Conv(IO::NetIO& io, const Tensor<T>& A1, const Tensor<T>& B1, const Tensor<T>& C1);
#endif

} // namespace Client

template <class Channel>
Result Client::Protocol2(Channel** client, const seal::SEALContext& context, const HomFCSS& hom_fc,
                         const HomFCSS::Meta& meta, const vector<Tensor<uint64_t>>& A2,
                         const vector<Tensor<uint64_t>>& B2, vector<Tensor<uint64_t>>& C2, const size_t& threads, const size_t& batch) {
    Result measures;

    auto start = measure::now();

    vector<vector<seal::Plaintext>> enc_A2(batch);
    for (size_t i = 0; i < batch; ++i) {
        measures.ret = hom_fc.encodeInputVector(A2[i], meta, enc_A2[i], threads);
    }
    if (measures.ret != Code::OK)
        return measures;

    vector<vector<vector<seal::Plaintext>>> enc_B2(batch);
    for (size_t i = 0; i < batch; ++i)
        measures.ret = hom_fc.encodeWeightMatrix(B2[i], meta, enc_B2[i], threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // recv A' + deserialize
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    vector<vector<seal::Ciphertext>> enc_A1(batch);
    for (size_t i = 0; i < batch; ++i)
        IO::recv_encrypted_vector(client, context, enc_A1[i]);

    measures.send_recv += Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // M2' = (A1' + A2) ⊙ B2 - R
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    vector<vector<seal::Ciphertext>> M2(batch);
    for (size_t i = 0; i < batch; ++i)
        measures.ret = hom_fc.MatVecMul(enc_A1[i], enc_A2[i], enc_B2[i], meta, M2[i], C2[i], threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.cipher_op = Utils::time_diff(start);

    ////////////////////////////////////////////////////////////////////////////
    // serialize + send M2'
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    for (size_t i = 0; i < batch; ++i)
        IO::send_encrypted_vector(client, M2[i]);

    measures.send_recv += Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += client[i]->counter;
    return measures;
}

template <class Channel>
Result Client::Protocol1(Channel** client, const seal::SEALContext& context,
                         const HomFCSS& hom_conv, const HomFCSS::Meta& meta,
                         const vector<Tensor<uint64_t>>& A2, const vector<Tensor<uint64_t>>& B2,
                         vector<Tensor<uint64_t>>& C2, const size_t& threads, const size_t& batch) {
    gemini::ThreadPool tpool(threads);
    Result fin_measures;
    auto prog = [&](long wid, size_t start, size_t end) {
        if (start >= end)
            return Code::OK;
        size_t ele = end - start;
        Result measures;

        ////////////////////////////////////////////////////////////////////////////
        // Receive A1' and enc/send A2'
        ////////////////////////////////////////////////////////////////////////////
        auto mess = measure::now();

        vector<vector<seal::Plaintext>> encoded_A2(ele);
        vector<vector<seal::Serializable<seal::Ciphertext>>> enc_A2(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret = hom_conv.encryptInputVector(A2[start + i], meta, enc_A2[i], encoded_A2[i], 1);

        if (measures.ret != Code::OK)
            return measures.ret;

        vector<vector<vector<seal::Plaintext>>> enc_B2(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret = hom_conv.encodeWeightMatrix(B2[start + i], meta, enc_B2[i], 1);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.encryption = Utils::time_diff(mess);
        mess               = measure::now();

        vector<vector<seal::Ciphertext>> enc_A1;
        IO::recv_send(context, client + wid, enc_A2, enc_A1);

        measures.send_recv += Utils::time_diff(mess);
        ////////////////////////////////////////////////////////////////////////////
        // M2' = (A1' + A2) ⊙ B2 - R2
        ////////////////////////////////////////////////////////////////////////////
        mess = measure::now();

        vector<vector<seal::Ciphertext>> enc_M2(ele);
        vector<Tensor<uint64_t>> R2(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret = hom_conv.MatVecMul(enc_A1[i], encoded_A2[i], enc_B2[i], meta, enc_M2[i], R2[i], 1);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.cipher_op = Utils::time_diff(mess);
        ////////////////////////////////////////////////////////////////////////////
        // send M2' + recv M1'
        ////////////////////////////////////////////////////////////////////////////
        mess = measure::now();

        vector<vector<seal::Ciphertext>> enc_M1(ele);
        measures.ret = IO::recv_send(context, client + wid, enc_M2, enc_M1, 1);

        measures.send_recv += Utils::time_diff(mess);
        ////////////////////////////////////////////////////////////////////////////
        // dec(M1') + R2
        ////////////////////////////////////////////////////////////////////////////
        mess = measure::now();

        for (size_t i = 0; i < ele; ++i)
            hom_conv.decryptToVector(enc_M1[i], meta, C2[start + i], 1);

        measures.decryption = Utils::time_diff(mess);

        mess = measure::now();

        for (size_t i = 0; i < ele; ++i)
            Utils::op_inplace<uint64_t>(
                C2[start + i], R2[i], [&hom_conv](uint64_t a, uint64_t b) -> uint64_t { return add(hom_conv, a, b); });

        measures.plain_op = Utils::time_diff(mess);
        return Code::OK;
    };

    fin_measures.ret = gemini::LaunchWorks(tpool, batch, prog);
    for (size_t i = 0; i < threads; ++i) fin_measures.bytes += client[i]->counter;

    return fin_measures;
}

template <class Channel>
Result Server::Protocol2(const HomFCSS::Meta& meta, Channel** server,
                         const seal::SEALContext& context, const HomFCSS& conv,
                         const vector<Tensor<uint64_t>>& A1, vector<Tensor<uint64_t>>& C1, const size_t& threads, const size_t& batch) {

    Result measures;

    ////////////////////////////////////////////////////////////////////////////
    // send Enc(A1)
    ////////////////////////////////////////////////////////////////////////////
    auto start = measure::now();

    vector<vector<seal::Serializable<seal::Ciphertext>>> enc_A1(batch);
    for (size_t i = 0; i < batch; ++i)
        measures.ret = conv.encryptInputVector(A1[i], meta, enc_A1[i], threads);
    if (measures.ret != Code::OK)
        return measures;

    measures.encryption = Utils::time_diff(start);

    start = measure::now();

    for (size_t i = 0; i < batch; ++i)
        IO::send_encrypted_vector(server, enc_A1[i]);

    measures.send_recv = Utils::time_diff(start);
    ////////////////////////////////////////////////////////////////////////////
    // recv C1 = dec(M2)
    ////////////////////////////////////////////////////////////////////////////
    start = measure::now();

    vector<vector<seal::Ciphertext>> enc_C1(batch);
    for (size_t i = 0; i < batch; ++i)
        IO::recv_encrypted_vector(server, context, enc_C1[i]);

    measures.send_recv += Utils::time_diff(start);
    start = measure::now();

    for (size_t i = 0; i < batch; ++i)
        measures.ret        = conv.decryptToVector(enc_C1[i], meta, C1[i], threads);
    measures.decryption = Utils::time_diff(start);

    for (size_t i = 0; i < threads; ++i) measures.bytes += server[i]->counter;
    return measures;
}

template <class Channel>
Result Server::Protocol1(const HomFCSS::Meta& meta, Channel** server,
                         const seal::SEALContext& context, const HomFCSS& conv,
                         const vector<Tensor<uint64_t>>& A1, const vector<Tensor<uint64_t>>& B1,
                         vector<Tensor<uint64_t>>& C1, const size_t& threads, const size_t& batch) {
    gemini::ThreadPool tpool(threads);
    auto prog = [&] (long wid, size_t first, size_t end) {
        if (first >= end)
            return Code::OK;

        Result measures;
        measures.send_recv = 0;
        size_t ele = end - first;

        ////////////////////////////////////////////////////////////////////////////
        // Enc(A1), enc(B1), send(A1), recv(A2)
        ////////////////////////////////////////////////////////////////////////////
        auto start = measure::now();
        vector<vector<seal::Serializable<seal::Ciphertext>>> enc_A1(ele);
        vector<vector<seal::Plaintext>> encoded_A1(ele);

        for (size_t i = 0; i < ele; ++i)
            measures.ret = conv.encryptInputVector(A1[first + i], meta, enc_A1[i], encoded_A1[i], 1);
        if (measures.ret != Code::OK)
            return measures.ret;

        vector<vector<vector<seal::Plaintext>>> enc_B1(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret = conv.encodeWeightMatrix(B1[first + i], meta, enc_B1[i], 1);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.encryption = Utils::time_diff(start);

        start = measure::now();

        vector<vector<seal::Ciphertext>> enc_A2(ele);
        IO::send_recv(context, server + wid, enc_A1, enc_A2, 1);

        measures.send_recv = Utils::time_diff(start);
        ////////////////////////////////////////////////////////////////////////////
        // M1 = (A1 + A2') ⊙ B1 - R1
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        vector<vector<seal::Ciphertext>> M1(ele);
        vector<Tensor<uint64_t>> R1(ele);
        for (size_t i = 0; i < ele; ++i)
            measures.ret = conv.MatVecMul(enc_A2[i], encoded_A1[i], enc_B1[i], meta, M1[i], R1[i], 1);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.cipher_op = Utils::time_diff(start);

        ////////////////////////////////////////////////////////////////////////////
        // Send(M1), Recv(M2), Dec(M2)
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        vector<vector<seal::Ciphertext>> enc_M2(ele);
        IO::send_recv(context, server + wid, M1, enc_M2, 1);

        measures.send_recv += Utils::time_diff(start);
        ////////////////////////////////////////////////////////////////////////////
        // Dec(M2) + R1
        ////////////////////////////////////////////////////////////////////////////
        start = measure::now();

        for (size_t i = 0; i < ele; ++i)
            measures.ret = conv.decryptToVector(enc_M2[i], meta, C1[first + i], 1);
        if (measures.ret != Code::OK)
            return measures.ret;

        measures.decryption = Utils::time_diff(start);
        start               = measure::now();

        for (size_t i = 0; i < ele; ++i)
            Utils::op_inplace<uint64_t>(C1[first + i], R1[i],
                                        [&conv](uint64_t a, uint64_t b) { return add(conv, a, b); });

        measures.plain_op = Utils::time_diff(start);
        return Code::OK;
    };

    Result final;
    final.ret = gemini::LaunchWorks(tpool, batch, prog);

    for (size_t i = 0; i < threads; ++i) final.bytes += server[i]->counter;
    return final;
}

template <class Channel>
Result Server::perform_proto(HomFCSS::Meta& meta, Channel** server,
                             const seal::SEALContext& context, const HomFCSS& hom_conv,
                             const size_t& threads, const size_t& batch) {
    vector<Tensor<uint64_t>> vecs(batch, Tensor<uint64_t>(meta.input_shape));
    for (auto& vec : vecs)
        for (long i = 0; i < vec.length(); i++) vec(i) = 2;
    vector<Tensor<uint64_t>> weights(batch, Tensor<uint64_t>(meta.weight_shape));
    for (auto& weight : weights)
        for (long i = 0; i < weight.rows(); i++)
            for (long j = 0; j < weight.cols(); j++) weight(i, j) = 2;

    vector<Tensor<uint64_t>> C1(batch);

    server[0]->sync();

#if PROTO == 1
    auto measures = Server::Protocol1(meta, server, context, hom_conv, vecs, weights, C1, threads, batch);
#else
    auto measures = Server::Protocol2(meta, server, context, hom_conv, vecs, C1, threads, batch);
#endif
    for (size_t i = 0; i < threads; ++i) server[i]->counter = 0;

#ifdef VERIFY
    server[0]->sync();
    Verify_Conv(*(server[0]), meta, hom_conv, vecs[0], weights[0], C1[0]);
#endif
    return measures;
}

template <class Channel>
Result Client::perform_proto(HomFCSS::Meta& meta, Channel** client,
                             const seal::SEALContext& context, const HomFCSS& hom_conv,
                             const size_t& threads, const size_t& batch) {
    vector<Tensor<uint64_t>> vecs(batch, Tensor<uint64_t>(meta.input_shape));
    for (auto& vec : vecs)
        for (long i = 0; i < vec.length(); i++) vec(i) = 2;
    vector<Tensor<uint64_t>> weights(batch, Tensor<uint64_t>(meta.weight_shape));
    for (auto& weight : weights)
        for (long i = 0; i < weight.rows(); i++)
            for (long j = 0; j < weight.cols(); j++) weight(i, j) = 2;

    client[0]->sync();

    std::vector<Tensor<uint64_t>> C2(batch);

#if PROTO == 1
    auto measures = Client::Protocol1(client, context, hom_conv, meta, vecs, weights, C2, threads, batch);
#else
    auto measures = Client::Protocol2(client, context, hom_conv, meta, vecs, weights, C2, threads, batch);
#endif

    for (size_t i = 0; i < threads; ++i) client[i]->counter = 0;

#ifdef VERIFY
    client[0]->sync();
    Verify_Conv(*(client[0]), vecs[0], weights[0], C2[0]);
#endif
    return measures;
}

#ifdef VERIFY
template <class T>
void Server::Verify_Conv(IO::NetIO& io, const HomFCSS::Meta& meta, const HomFCSS& conv,
                         const Tensor<T>& A1, const Tensor<T>& B1, const Tensor<T>& C1) {
    Utils::log(Utils::Level::INFO, "VERIFYING FC");
    Tensor<T> A2(A1.shape());
    Tensor<T> B2(meta.weight_shape);
    Tensor<T> C2(C1.shape());

    io.recv_data(A2.data(), A2.NumElements() * sizeof(T));
    io.recv_data(B2.data(), B2.NumElements() * sizeof(T));
    io.recv_data(C2.data(), C2.NumElements() * sizeof(T));

    Utils::op_inplace<T>(C2, C1, [&conv](T a, T b) -> T { return add(conv, a, b); }); // C
    Utils::op_inplace<T>(A2, A1, [&conv](T a, T b) -> T { return add(conv, a, b); }); // A1 + A2

#if PROTO == 1
    Utils::op_inplace<T>(B2, B1, [&conv](T a, T b) -> T { return add(conv, a, b); });
#endif

    Tensor<T> test;                              // (A1 + A2) (B1 + B2)
    conv.idealFunctionality(A2, B2, meta, test); // (A1 + A2) (B1 + B2)

    bool same = C2.shape() == test.shape();

    for (long i = 0; i < C2.length(); ++i) {
        if (!same || test(i) != C2(i)) {
            Utils::log(Utils::Level::FAILED, i, ", ", test(i), ", ", C2(i));
            same = false;
            break;
        }
    }

    if (same)
        Utils::log(Utils::Level::PASSED, "FC: PASSED");
    else
        Utils::log(Utils::Level::FAILED, "FC: FAILED");
}

template <class T>
void Client::Verify_Conv(IO::NetIO& io, const Tensor<T>& A2, const Tensor<T>& B2,
                         const Tensor<T>& C2) {
    log(Utils::Level::INFO, "SENDING");
    io.send_data(A2.data(), A2.NumElements() * sizeof(T), false);
    io.send_data(B2.data(), B2.NumElements() * sizeof(T), false);
    io.send_data(C2.data(), C2.NumElements() * sizeof(T), false);
    io.flush();
}
#endif

#endif