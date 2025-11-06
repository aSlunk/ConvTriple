#ifndef HPMPC_INTERFACE_HPP_
#define HPMPC_INTERFACE_HPP_

#include "protocols/bn_direct_proto.hpp"
#include "utils.hpp"

#include "ot/bit-triple-generator.h"

#include <string>

#include <io/net_io_channel.hpp>

#include <gemini/cheetah/hom_bn_ss.h>

namespace Iface {

template <class Channel, class SerKey>
void exchange_keys(Channel** ios, const SerKey& pkey, seal::PublicKey& o_pkey,
                   const seal::SEALContext& ctx, int party) {
    switch (party) {
    case emp::ALICE:
        IO::send_pkey(*(ios[0]), pkey);
        IO::recv_pkey(*(ios[0]), ctx, o_pkey);
        break;
    case emp::BOB:
        IO::recv_pkey(*(ios[0]), ctx, o_pkey);
        IO::send_pkey(*(ios[0]), pkey);
        break;
    }
}

void generateBoolTriplesCheetah(uint8_t a[], uint8_t b[], uint8_t c[], int bitlength,
                                uint64_t num_triples, std::string ip, int port, int party,
                                int threads = 1, TripleGenMethod method = _16KKOT_to_4OT,
                                unsigned io_offset = 1);

void generateArithTriplesCheetah(const uint32_t a[], const uint32_t b[], uint32_t c[],
                                 int bitlength, uint64_t num_triples, std::string ip, int port,
                                 int party, int threads = 1, Utils::PROTO proto = Utils::PROTO::AB,
                                 unsigned io_offset = 1);

void generateFCTriplesCheetah(IO::NetIO** ios, const uint32_t* a, const uint32_t* b, uint32_t* c,
                              int batch, uint64_t com_dim, uint64_t dim2, int party, int threads,
                              Utils::PROTO proto, int factor = 1);

void generateConvTriplesCheetahWrapper(IO::NetIO** ios, const uint32_t* a, const uint32_t* b,
                                       uint32_t* c, Utils::ConvParm parm, int party, int threads,
                                       Utils::PROTO proto, int factor = 1,
                                       bool is_shared_input = false);

void generateConvTriplesCheetah(IO::NetIO** ios, size_t total_batches,
                                std::vector<Utils::ConvParm>& parms, uint32_t** a, uint32_t** b,
                                uint32_t* c, Utils::PROTO proto, int party, int threads, int factor,
                                bool is_shared_input = false);

void generateConvTriplesCheetah(IO::NetIO** ios, const uint32_t* a, const uint32_t* b, uint32_t* c,
                                const gemini::HomConv2DSS::Meta& meta, int batch, int party,
                                int threads, Utils::PROTO proto, int factor);

void generateBNTriplesCheetah(IO::NetIO** ios, const uint32_t* a, const uint32_t* b, uint32_t* c,
                              int batch, size_t num_ele, size_t h, size_t w, int party, int threads,
                              Utils::PROTO proto, int factor = 1);

void tmp(int party, int threads);

template <class Channel, class Serial>
void send_vec(Channel** ios, std::vector<std::vector<Serial>>& vec, int threads) {
    uint64_t n = vec.size();
    ios[0]->send_data(&n, sizeof(uint64_t));

    auto func = [&](int wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;

        auto* io = ios[wid];
        std::vector<uint64_t> idxs(end - start + 1);

        std::stringstream stream;
        for (size_t i = start; i < end; ++i) {
            idxs[i - start] = vec[i].size();
            for (auto& ele : vec[i]) {
                ele.save(stream);
            }
        }

        auto data   = stream.str();
        idxs.back() = data.size();
        io->send_data(idxs.data(), idxs.size() * sizeof(uint64_t));
        if (idxs.back() == 0) {
            io->flush();
            return Code::OK;
        }
        io->send_data(data.c_str(), idxs.back());

        io->flush();

        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, vec.size(), func);
}

template <class Channel, class Serial>
void recv_vec(Channel** ios, const seal::SEALContext& ctx, std::vector<std::vector<Serial>>& vec,
              int threads) {
    uint64_t n = 0;
    ios[0]->recv_data(&n, sizeof(uint64_t));
    vec.resize(n);

    auto func = [&](int wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;
        auto* io = ios[wid];

        std::vector<uint64_t> idxs(end - start + 1);
        io->recv_data(idxs.data(), idxs.size() * sizeof(uint64_t));

        if (idxs.back() == 0)
            return Code::OK;

        char* data = new char[idxs.back()];
        io->recv_data((void*)data, idxs.back());

        std::stringstream stream(std::string(data, idxs.back()));
        for (size_t i = 0; i < idxs.size() - 1; ++i) {
            vec[start + i].resize(idxs[i]);

            for (auto& ct : vec[start + i]) {
                ct.load(ctx, stream);
            }
        }

        delete[] data;
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, vec.size(), func);
}

} // namespace Iface

#endif