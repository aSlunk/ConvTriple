#include "hpmpc_interface.hpp"

#include <algorithm>

#include "protocols/bn_direct_proto.hpp"
#include "protocols/conv_proto.hpp"
#include "protocols/fc_proto.hpp"
#include "protocols/ot_proto.hpp"

#include "ot/bit-triple-generator.h"
#include "ot/cheetah-ot_pack.h"

#include "core/keys.hpp"

#if USE_CONV_CUDA
#include "troy/conv2d_gpu.cuh"
#endif

#include "elem.hpp"

constexpr uint64_t MAX_BOOL  = 1ULL << 24;
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
                                TripleGenMethod method, unsigned io_offset) {
    Utils::log(Utils::Level::INFO, "P", party - 1, ": num_triples (BOOL): ", num_triples);
    const char* addr = ip.c_str();
    if (party == emp::ALICE)
        addr = nullptr;

    // std::atomic<int> setup = 0;
    auto start = measure::now();

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads, io_offset);

    auto func = [&](int wid, int start, int end) -> Code {
        if (start >= end)
            return Code::OK;

        int cur_party = wid & 1 ? OTHER_PARTY(party) : party;
        // auto start_setup = measure::now();

        sci::OTPack<IO::NetIO> ot_pack(ios + wid, 1, cur_party, true, false);
        TripleGenerator<IO::NetIO> triple_gen(cur_party, ios[wid], &ot_pack, false);

        // setup += Utils::time_diff(start_setup);

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

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": Bool triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": Bool triple data[", unit, "]: ", data);

    // Utils::log(Utils::Level::INFO, "P", party - 1, ": Setup time [s]: ",
    //            Utils::to_sec(setup.load())
    //                / (num_triples > static_cast<size_t>(threads) ? threads : num_triples));

    for (int i = 0; i < threads; ++i) {
        delete ios[i];
    }
    delete[] ios;
}

void generateArithTriplesCheetah(const uint32_t a[], const uint32_t b[], uint32_t c[],
                                 int bitlength, uint64_t num_triples, std::string ip, int port,
                                 int party, int threads, Utils::PROTO proto, unsigned io_offset) {
    assert(bitlength == 32 && "[arith. triples] Unsupported bitlength");
    Utils::log(Utils::Level::INFO, "P", party - 1, ": num_triples (ARITH): ", num_triples,
               " " + Utils::proto_str(proto));
    const char* addr = ip.c_str();
    if (party == emp::ALICE)
        addr = nullptr;

    auto start = measure::now();

    IO::NetIO** ios = Utils::init_ios<IO::NetIO>(addr, port, threads, io_offset);

    auto& bn = Keys::instance(ios, party).get_bn();

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
            m.vec_shape             = gemini::TensorShape({static_cast<long>(current)});

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
                Utils::log(Utils::Level::ERROR, "Unknown party: P", party - 1);
            }
            }

            for (uint64_t i = 0; i < current; ++i) c[i + total] = static_cast<uint32_t>(tmp_C(i));
            total += current;
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_triples, func);

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": Arith triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": Arith triple data[", unit, "]: ", data);

    for (int i = 0; i < threads; ++i) delete ios[i];
    delete[] ios;
}

void generateFCTriplesCheetah(IO::NetIO** ios, const uint32_t* a, const uint32_t* b, uint32_t* c,
                              int batch, uint64_t com_dim, uint64_t dim2, int party, int threads,
                              Utils::PROTO proto, int factor) {
    auto meta = Utils::init_meta_fc(com_dim, dim2);
    Utils::log(Utils::Level::INFO, "P", party - 1, " FC: ", meta.input_shape, " x ",
               meta.weight_shape, " ", Utils::proto_str(proto));

    auto start = measure::now();

    // meta.is_shared_input = proto == Utils::PROTO::AB;
    auto& fc = Keys::instance(ios, party).get_fc();

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

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": FC triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": FC triple data[", unit, "]: ", data);

    delete[] ai;
    delete[] bi;
}

void generateConvTriplesCheetahWrapper(IO::NetIO** ios, const uint32_t* a, const uint32_t* b,
                                       uint32_t* c, Utils::ConvParm parm, int party, int threads,
                                       Utils::PROTO proto, int factor) {
#if USE_CONV_CUDA
    if (proto == Utils::PROTO::AB2) {
        TROY::conv2d(ios, OTHER_PARTY(party), a, b, c, parm.batchsize, parm.ic, parm.ih, parm.iw,
                     parm.fh, parm.fw, parm.n_filters, parm.stride, parm.padding, false, factor);
        return;
    }
#endif
    auto meta = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                      parm.n_filters, parm.stride, parm.padding);

    Utils::log(Utils::Level::INFO, "P", party - 1, " CONV: ", meta.ishape, " x ", meta.fshape,
               " x ", parm.n_filters, ", ", parm.stride, ", ", parm.padding, ", ",
               Utils::proto_str(proto));

    meta.is_shared_input = proto == Utils::PROTO::AB;
    if (Utils::getOutDim(parm) == gemini::GetConv2DOutShape(meta)) {
        generateConvTriplesCheetah(ios, a, b, c, meta, parm.batchsize, party, threads, proto,
                                   factor);
    } else {
        Utils::log(Utils::Level::INFO, "Adding padding manually");

        std::vector<uint32_t> ai;
        std::tuple<int, int> dim;

        dim = Utils::pad_zero(a, ai, parm.ic, parm.ih, parm.iw, parm.padding, parm.batchsize);

        parm.ih      = std::get<0>(dim);
        parm.iw      = std::get<1>(dim);
        parm.padding = 0;

        meta = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                     parm.n_filters, parm.stride, parm.padding);
        generateConvTriplesCheetah(ios, ai.data(), b, c, meta, parm.batchsize, party, threads,
                                   proto, factor);
    }
}

void generateConvTriplesCheetah(IO::NetIO** ios, size_t total_batches,
                                std::vector<Utils::ConvParm>& parms, uint32_t** a, uint32_t** b,
                                uint32_t* c, Utils::PROTO proto, int party, int threads,
                                int factor) {
    auto start = measure::now();

    vector<vector<seal::Plaintext>> enc_a(total_batches);
    vector<vector<vector<seal::Plaintext>>> enc_b(total_batches);
    vector<vector<seal::Ciphertext>> enc_a2(total_batches);
    vector<vector<seal::Serializable<seal::Ciphertext>>> enc_a1(total_batches);

    auto& hom_conv = Keys::instance(ios, party).get_conv();

    auto pool = seal::MemoryPoolHandle::New();
    auto pg   = seal::MMProfGuard(std::make_unique<seal::MMProfFixed>(std::move(pool)));

    size_t offset = 0;

    Result result;
    for (size_t n = 0; n < parms.size(); ++n) {
        auto& parm = parms[n];
        auto meta  = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                           parm.n_filters, parm.stride, parm.padding);

        meta.is_shared_input = false;
        uint64_t* ai         = new uint64_t[meta.ishape.num_elements() * parm.batchsize];
        for (long i = 0; i < meta.ishape.num_elements() * parm.batchsize; ++i)
            ai[i] = a != nullptr ? a[n][i] : 0;

        uint64_t* bi = new uint64_t[meta.fshape.num_elements() * meta.n_filters * factor];
        if (b)
            for (size_t i = 0; i < meta.fshape.num_elements() * meta.n_filters * factor; ++i)
                bi[i] = b[n][i];

        int ac_batch_size = parm.batchsize / factor;
        for (int cur_batch = 0; cur_batch < parm.batchsize; ++cur_batch) {
            Tensor<uint64_t> A
                = Tensor<uint64_t>::Wrap(ai + meta.ishape.num_elements() * cur_batch, meta.ishape);

            std::vector<Tensor<uint64_t>> B(meta.n_filters);
            for (size_t i = 0; i < meta.n_filters; ++i)
                B[i] = Tensor<uint64_t>::Wrap(
                    bi + meta.fshape.num_elements() * meta.n_filters * (cur_batch / ac_batch_size)
                        + meta.fshape.num_elements() * i,
                    meta.fshape);

            switch (party) {
            case emp::ALICE: {
                hom_conv.encodeImage(A, meta, enc_a[cur_batch + offset], threads);
                hom_conv.encodeFilters(B, meta, enc_b[cur_batch + offset], threads);
                break;
            }
            case emp::BOB: {
                hom_conv.encryptImage(A, meta, enc_a1[cur_batch + offset], threads);
                break;
            }
            }
            offset += parm.batchsize;
        }
        delete[] ai;
        delete[] bi;
    }

    offset = 0;
    for (size_t n = 0; n < parms.size(); ++n) {
        for (int batch = 0; batch < parms[n].batchsize; ++batch) {
            switch (party) {
            case emp::BOB: {
                IO::send_encrypted_vector(ios, enc_a1[batch + offset], threads, false);
                break;
            }
            case emp::ALICE: {
                IO::recv_encrypted_vector(ios, hom_conv.getContext(), enc_a2[batch + offset],
                                          threads);
            }
            }
        }
        offset += parms[n].batchsize;
    }
    for (int i = 0; i < threads; ++i) ios[i]->flush();

    vector<vector<seal::Ciphertext>> M(total_batches);
    vector<Tensor<uint64_t>> C(total_batches);
    offset = 0;
    for (size_t n = 0; n < parms.size(); ++n) {
        auto& parm = parms[n];
        auto meta  = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                           parm.n_filters, parm.stride, parm.padding);
        for (int cur_batch = 0; cur_batch < parm.batchsize; ++cur_batch) {
            switch (party) {
            case emp::ALICE: {
                result.ret
                    = hom_conv.conv2DSS(enc_a2[cur_batch + offset], enc_a[cur_batch + offset],
                                        enc_b[cur_batch + offset], meta, M[cur_batch + offset],
                                        C[cur_batch + offset], threads);
                enc_a[n].clear();
                enc_b[n].clear();
                enc_a2[n].clear();
                break;
            }
            }
        }
        offset += parm.batchsize;
    }
    enc_a.clear();
    enc_b.clear();
    enc_a2.clear();

    offset = 0;
    for (size_t n = 0; n < parms.size(); ++n) {
        for (int cur_batch = 0; cur_batch < parms[n].batchsize; ++cur_batch) {
            switch (party) {
                case emp::ALICE: { // send
                    IO::send_encrypted_vector(ios, M[cur_batch + offset], threads, false);
                    M[cur_batch + offset].clear();
                    break;
                }
                case emp::BOB: { // recv
                    IO::recv_encrypted_vector(ios, hom_conv.getContext(), M[cur_batch + offset], threads);
                    break;
                }
            }
        }
        offset += parms[n].batchsize;
    }
    for (int i = 0; i < threads; ++i) ios[i]->flush();

    offset          = 0;
    size_t c_offset = 0;
    for (size_t n = 0; n < parms.size(); ++n) {
        auto& parm = parms[n];
        auto meta  = Utils::init_meta_conv(parm.ic, parm.ih, parm.iw, parm.fc, parm.fh, parm.fw,
                                           parm.n_filters, parm.stride, parm.padding);

        for (int cur_batch = 0; cur_batch < parm.batchsize; ++cur_batch) {
            switch (party) {
            case emp::BOB: {
                result.ret = hom_conv.decryptToTensor(M[cur_batch + offset], meta, C[cur_batch + offset], threads);
                break;
            }
            }

            for (long i = 0; i < C[cur_batch + offset].NumElements(); ++i)
                c[c_offset + i] = C[cur_batch + offset].data()[i];
            c_offset += C[cur_batch].NumElements();
        }
        offset += parm.batchsize;
    }

    auto time = Utils::to_sec(Utils::time_diff(start));
    Utils::log(Utils::Level::INFO, "P", party - 1, ": Conv triple time [s]: ", time);
}

void generateConvTriplesCheetah(IO::NetIO** ios, const uint32_t* a, const uint32_t* b, uint32_t* c,
                                const gemini::HomConv2DSS::Meta& meta, int batch, int party,
                                int threads, Utils::PROTO proto, int factor) {
    auto start = measure::now();

    auto& conv = Keys::instance(ios, party).get_conv();

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

        Result result;
        switch (party) {
        case emp::ALICE: {
            result = Client::perform_proto(meta, ios, conv, A, B, C, threads, proto);
            break;
        }
        case emp::BOB: {
            result = Server::perform_proto(meta, ios, conv, A, B, C, threads, proto);
            break;
        }
        }

        // Utils::print_results(result, 0, batch, threads);
        for (long i = 0; i < C.NumElements(); ++i) c[i + C.NumElements() * cur_batch] = C.data()[i];
    }

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": CONV triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": CONV triple data[", unit, "]: ", data);

    delete[] ai;
    delete[] bi;
}

void generateBNTriplesCheetah(IO::NetIO** ios, const uint32_t* a, const uint32_t* b, uint32_t* c,
                              int batch, size_t num_ele, size_t h, size_t w, int party, int threads,
                              Utils::PROTO proto, int factor) {
    auto meta = Utils::init_meta_bn(num_ele, h, w);
    Utils::log(Utils::Level::INFO, "P", party - 1, " BN: ", meta.ishape, " x ", meta.vec_shape,
               ", ", Utils::proto_str(proto));

    auto start = measure::now();

    meta.is_shared_input = proto == Utils::PROTO::AB;
    auto& bn             = Keys::instance(ios, party).get_bn();

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

    Utils::log(Utils::Level::INFO, "P", party - 1,
               ": BN triple time[s]: ", Utils::to_sec(Utils::time_diff(start)));
    std::string unit;
    double data = 0;
    for (int i = 0; i < threads; ++i) data += Utils::to_MB(ios[i]->counter, unit);
    Utils::log(Utils::Level::INFO, "P", party - 1, ": BN triple data[", unit, "]: ", data);
}

void tmp(int party, int threads) {
    // auto context = Utils::init_he_context();
    auto start = measure::now();
    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_MOD);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(POLY_MOD, {60, 49}));
    parms.set_n_special_primes(0);
    size_t prime_mod = seal::PlainModulus::Batching(POLY_MOD, 32).value();
    // std::cout << prime_mod << "\n";
    parms.set_plain_modulus(PLAIN_MOD);
    seal::SEALContext context(parms, true, seal::sec_level_type::tc128);

    auto io
        = Utils::init_ios<IO::NetIO>(party == emp::ALICE ? nullptr : "127.0.0.1", 6969, threads);

    seal::KeyGenerator keygen(context);
    seal::SecretKey skey = keygen.secret_key();
    auto pkey            = std::make_shared<seal::PublicKey>();
    auto o_pkey          = std::make_shared<seal::PublicKey>();
    keygen.create_public_key(*pkey);
    exchange_keys(io, *pkey, *o_pkey, context, party);

    seal::Encryptor enc(context, *o_pkey);
    enc.set_secret_key(skey);
    seal::Decryptor dec(context, skey);

    uint64_t num_triples = 9'006'592;
    std::vector<uint64_t> A(num_triples);
    std::vector<uint64_t> B(num_triples);
    std::vector<uint64_t> C(num_triples);

    auto func = [&](int wid, size_t start, size_t end) {
        if (start >= end)
            return Code::OK;
        size_t triple = end - start;

        for (size_t i = start; i < end; ++i) {
            A[i] = 2;
            B[i] = 3;
        }

        elemwise_product_ab(&context, io[wid], &enc, &dec, triple, A.data() + start,
                            B.data() + start, C.data() + start, prime_mod, party, *o_pkey);
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, num_triples, func);

    size_t data = 0;
    for (int i = 0; i < threads; ++i) data += io[i]->counter;
    string st;
    std::cout << "P" << party - 1 << ": time[s]: " << Utils::to_sec(Utils::time_diff(start))
              << "\n";
    std::cout << "P" << party - 1 << ": data: " << Utils::to_MB(data, st) << st << "\n";

    for (int i = 0; i < threads; ++i) {
        delete io[i];
    }
    delete[] io;
}

} // namespace Iface