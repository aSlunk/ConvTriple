#ifndef SEND_HPP_
#define SEND_HPP_

#include <cassert>
#include <chrono>
#include <string>
#include <vector>

#include <seal/seal.h>
#include <sys/socket.h>

#include <gemini/cheetah/hom_conv2d_ss.h>

#include "net_io_channel.hpp"

#include "core/defs.hpp"

using std::string;
using std::vector;

namespace IO {

using Unit = std::chrono::milliseconds;
using CLK  = std::chrono::high_resolution_clock;

template <class EncVec>
Code send_recv(const seal::SEALContext& ctx, IO::NetIO** ios, const std::vector<EncVec>& send,
               vector<vector<seal::Ciphertext>>& recv, const size_t& threads = 1);

template <class EncVec>
Code send_recv(const seal::SEALContext& ctx, vector<IO::NetIO>& ios, const EncVec& send,
               vector<seal::Ciphertext>& recv);

/**
 * Serialize/Deseralize multithreaded but using only @param threads ports for sending/receiving
 */
template <class EncVec>
Code send_recv(const seal::SEALContext& ctx, IO::NetIO** ios, const EncVec& send,
               vector<seal::Ciphertext>& recv, const size_t& threads = 1);

/**
 * Serialize/Deseralize multithreaded but using only ONE port for sending/receiving
 */
template <class EncVec>
Code send_recv2(const seal::SEALContext& ctx, IO::NetIO** ios, const EncVec& send,
                vector<seal::Ciphertext>& recv, const size_t& threads = 1);

template <class Vec>
Code recv_send(const seal::SEALContext& ctx, IO::NetIO** ios, const vector<Vec>& send,
               vector<vector<seal::Ciphertext>>& recv, const size_t& threads = 1);

template <class Vec>
Code recv_send(const seal::SEALContext& ctx, vector<IO::NetIO>& ios, const Vec& send,
               vector<seal::Ciphertext>& recv);

template <class Vec>
Code recv_send(const seal::SEALContext& ctx, IO::NetIO** ios, const Vec& send,
               vector<seal::Ciphertext>& recv, const size_t& threads = 1);

template <class Vec>
Code recv_send2(const seal::SEALContext& ctx, IO::NetIO** ios, const Vec& send,
                vector<seal::Ciphertext>& recv, const size_t& threads = 1);

template <class CtType>
void send_ciphertext(IO::NetIO& io, const CtType& ct);

template <class EncVecCtType>
void send_encrypted_vector(IO::NetIO& io, const EncVecCtType& ct_vec);

template <class EncVecCtType>
void send_encrypted_vector(IO::NetIO** ios, const EncVecCtType& ct_vec, const size_t& threads = 1);

template <class EncVecCtType>
void send_encrypted_vector(vector<IO::NetIO>& io, const EncVecCtType& ct_vec);

void send_encrypted_vector(IO::NetIO& io, std::stringstream& ct, const uint32_t& size);

template <class EncVecCtType>
void send_encrypted_filters(IO::NetIO& io, const EncVecCtType& ct_vec); // ADDED

void recv_encrypted_filters(IO::NetIO& io, const seal::SEALContext& context,
                            vector<vector<seal::Ciphertext>>& ct_vec,
                            bool is_truncated = false); // ADDED

void recv_encrypted_vector(vector<IO::NetIO>& io, const seal::SEALContext& context,
                           vector<seal::Ciphertext>& ct_vec, bool is_truncated = false);

void recv_encrypted_vector(IO::NetIO** io, const seal::SEALContext& context,
                           vector<seal::Ciphertext>& ct_vec, const size_t& threads = 1,
                           bool is_truncated = false);

void recv_encrypted_vector(IO::NetIO& io, const seal::SEALContext& context,
                           vector<seal::Ciphertext>& ct_vec, bool is_truncated = false);

uint32_t recv_encrypted_vector(IO::NetIO& io, std::stringstream& is);

void recv_ciphertext(IO::NetIO& io, const seal::SEALContext& context, seal::Ciphertext& ct,
                     bool is_truncated);

template <class PKey>
void send_pkey(IO::NetIO& io, const PKey& pkey);

template <class PKey>
void recv_pkey(IO::NetIO& io, const seal::SEALContext& context, PKey& pkey);

} // namespace IO

template <class CtType>
void IO::send_ciphertext(IO::NetIO& io, const CtType& ct) {
    std::stringstream os;
    uint64_t ct_size;
    ct.save(os);
    ct_size       = os.tellp();
    string ct_ser = os.str();
    io.send_data(&ct_size, sizeof(uint64_t));
    io.send_data(ct_ser.c_str(), ct_ser.size());
}

template <class EncVecCtType>
void IO::send_encrypted_vector(IO::NetIO& io, const EncVecCtType& ct_vec) {
    uint32_t ncts = ct_vec.size();
    io.send_data(&ncts, sizeof(uint32_t));

    std::stringstream os;
    uint64_t ct_size;
    for (size_t i = 0; i < ncts; ++i) {
        ct_vec.at(i).save(os);
    }
    ct_size       = os.tellp();
    string ct_ser = os.str();
    io.send_data(&ct_size, sizeof(uint64_t));
    io.send_data(ct_ser.c_str(), ct_ser.size());
    io.flush();
}

template <class EncVecCtType>
void IO::send_encrypted_filters(IO::NetIO& io, const EncVecCtType& ct_vec) {
    uint32_t ncts = ct_vec.size();
    io.send_data(&ncts, sizeof(uint32_t));
    for (size_t i = 0; i < ncts; ++i) {
        send_encrypted_vector(io, ct_vec[i]);
    }
}

template <class EncVecCtType>
void IO::send_encrypted_vector(IO::NetIO** ios, const EncVecCtType& ct_vec, const size_t& threads) {
    uint32_t ncts = ct_vec.size();
    ios[0]->send_data(&ncts, sizeof(uint32_t));

    auto program = [&](long wid, size_t start, size_t end) -> Code {
        auto& io = *(ios[wid]);
        for (size_t i = start; i < end; ++i) {
            send_ciphertext(io, ct_vec.at(i));
            io.flush();
        }
        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    gemini::LaunchWorks(tpool, ncts, program);
}

template <class EncVecCtType>
void IO::send_encrypted_vector(vector<IO::NetIO>& ios, const EncVecCtType& ct_vec) {
    NetIO** ios_c = new NetIO*[ios.size()];
    for (size_t i = 0; i < ios.size(); ++i) {
        ios_c[i] = &ios[i];
    }

    send_encrypted_vector(ios_c, ct_vec, ios.size());

    delete[] ios_c;
}

template <class PKey>
void IO::recv_pkey(IO::NetIO& io, const seal::SEALContext& context, PKey& pkey) {
    size_t len{0};
    io.recv_data(&len, sizeof(len));
    char* recv = new char[len];
    io.recv_data(recv, len);

    std::stringstream is;
    is.write(recv, len);
    pkey.load(context, is);
    delete[] recv;
}

template <class PKey>
void IO::send_pkey(IO::NetIO& io, const PKey& pkey) {
    std::stringstream is;
    pkey.save(is);
    size_t len = is.str().size();
    io.send_data(&len, sizeof(len));
    io.send_data(is.str().data(), len);
    io.flush();
}

template <class EncVec>
Code IO::send_recv(const seal::SEALContext& ctx, IO::NetIO** ios, const std::vector<EncVec>& send,
                   vector<vector<seal::Ciphertext>>& recv, const size_t& threads) {
    EncVec to_send;
    vector<seal::Ciphertext> to_recv;

    for (size_t i = 0; i < send.size(); i++) {
        for (size_t j = 0; j < send[i].size(); j++) {
            to_send.push_back(send[i][j]);
        }
    }

    send_recv(ctx, ios, to_send, to_recv, threads);
    to_send.clear();

    recv.resize(send.size());
    for (size_t i = 0; i < send.size(); ++i) {
        recv[i].reserve(send[i].size());
        for (size_t j = 0; j < send[i].size(); ++j) {
            recv[i].push_back(to_recv[i * send[i].size() + j]);
        }
    }

    return Code::OK;
}

template <class EncVec>
Code IO::send_recv(const seal::SEALContext& ctx, vector<IO::NetIO>& ios, const EncVec& send,
                   vector<seal::Ciphertext>& recv) {
    NetIO** ios_c = new NetIO*[ios.size()];
    for (size_t i = 0; i < ios.size(); ++i) {
        ios_c[i] = &ios[i];
    }
    Code code = send_recv(ctx, ios_c, send, recv, ios.size());

    delete[] ios_c;
    return code;
}

template <class EncVec>
Code IO::send_recv(const seal::SEALContext& ctx, IO::NetIO** ios, const EncVec& send,
                   vector<seal::Ciphertext>& recv, const size_t& threads) {
    vector<vector<seal::Ciphertext>> result(threads, vector<seal::Ciphertext>());

    auto program = [&](long wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;

        auto& server = *(ios[wid]);
        std::stringstream is;
        for (size_t cur = start; cur < end; ++cur) {
            send.at(cur).save(is);
        }

        IO::send_encrypted_vector(server, is, end - start);
        is.clear();

        uint32_t ncts = IO::recv_encrypted_vector(server, is);
        result[wid].resize(ncts);
        for (auto& res : result[wid]) res.load(ctx, is);

        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    CHECK_ERR(gemini::LaunchWorks(tpool, send.size(), program), "send_recv");

    for (auto& vec : result) {
        if (!vec.size())
            continue;

        recv.reserve(recv.size() + vec.size());
        for (auto& ele : vec) recv.push_back(ele);
    }
    return Code::OK;
}

template <class EncVec>
Code IO::send_recv2(const seal::SEALContext& ctx, IO::NetIO** ios, const EncVec& send,
                    vector<seal::Ciphertext>& recv, const size_t& threads) {
    std::cerr << "EXPERIMENTAL: send_recv2\n";
    vector<std::tuple<std::stringstream, size_t>> is_th(threads);

    auto serialize = [&is_th, &send](long wid, size_t start, size_t end) -> Code {
        auto& [ct, num] = is_th[wid];

        if (start >= end) {
            num = 0;
            return Code::OK;
        }

        num = end - start;

        for (size_t cur = start; cur < end; ++cur) {
            send.at(cur).save(ct);
        }

        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    CHECK_ERR(gemini::LaunchWorks(tpool, send.size(), serialize), "Serialize");

    for (auto& [stream, len] : is_th) {
        IO::send_encrypted_vector(*ios[0], stream, len);
        stream.clear();
    }

    size_t total = 0;
    for (auto& [stream, len] : is_th) {
        len = IO::recv_encrypted_vector(*ios[0], stream);
        total += len;
    }

    if (total != send.size()) {
        Utils::log(Utils::Level::ERROR, "send_recv: Input lengths mismatch");
    }

    recv.resize(total);

    auto deserialize = [&ctx, &is_th, &recv](long wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;

        auto& [stream, num] = is_th[wid];

        if (end - start != num)
            return Code::ERR_DIM_MISMATCH;

        for (size_t i = start; i < end; ++i) {
            recv[i].load(ctx, stream);
        }
        return Code::OK;
    };
    CHECK_ERR(gemini::LaunchWorks(tpool, total, deserialize), "deserialize");
    return Code::OK;
}

template <class Vec>
Code IO::recv_send(const seal::SEALContext& ctx, IO::NetIO** ios, const std::vector<Vec>& send,
                   vector<vector<seal::Ciphertext>>& recv, const size_t& threads) {
    Vec to_send;
    vector<seal::Ciphertext> to_recv;

    for (size_t i = 0; i < send.size(); i++) {
        for (size_t j = 0; j < send[i].size(); j++) {
            to_send.push_back(send[i][j]);
        }
    }

    recv_send(ctx, ios, to_send, to_recv, threads);

    recv.resize(send.size());
    for (size_t i = 0; i < send.size(); ++i) {
        recv[i].reserve(send[i].size());
        for (size_t j = 0; j < send[i].size(); ++j) {
            recv[i].push_back(to_recv[i * send[i].size() + j]);
        }
    }

    return Code::OK;
}

template <class Vec>
Code IO::recv_send(const seal::SEALContext& ctx, vector<IO::NetIO>& ios, const Vec& send,
                   vector<seal::Ciphertext>& recv) {
    NetIO** ios_c = new NetIO*[ios.size()];
    for (size_t i = 0; i < ios.size(); ++i) {
        ios_c[i] = &ios[i];
    }
    Code code = recv_send(ctx, ios_c, send, recv, ios.size());
    delete[] ios_c;
    return code;
}

template <class Vec>
Code IO::recv_send(const seal::SEALContext& ctx, IO::NetIO** ios, const Vec& send,
                   vector<seal::Ciphertext>& recv, const size_t& threads) {
    vector<vector<seal::Ciphertext>> result(threads, vector<seal::Ciphertext>(0));

    auto program = [&](long wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;

        auto& client = *(ios[wid]);
        std::stringstream is;
        for (size_t cur = start; cur < end; ++cur) {
            send.at(cur).save(is);
        }

        std::stringstream os;
        uint32_t ncts = IO::recv_encrypted_vector(client, os);

        IO::send_encrypted_vector(client, is, end - start);
        is.clear();
        result[wid].resize(ncts);

        for (auto& res : result[wid]) {
            res.load(ctx, os);
        }

        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    CHECK_ERR(gemini::LaunchWorks(tpool, send.size(), program), "recv_send");

    for (auto& vec : result) {
        if (!vec.size())
            continue;

        recv.reserve(recv.size() + vec.size());
        for (auto& ele : vec) recv.push_back(ele);
    }
    return Code::OK;
}

template <class EncVec>
Code IO::recv_send2(const seal::SEALContext& ctx, IO::NetIO** ios, const EncVec& send,
                    vector<seal::Ciphertext>& recv, const size_t& threads) {
    std::cerr << "EXPERIMENTAL: recv_send2\n";
    vector<std::tuple<std::stringstream, size_t>> is_th_r(threads);
    vector<std::tuple<std::stringstream, size_t>> is_th_s(threads);

    auto serialize = [&is_th_s, &send](long wid, size_t start, size_t end) -> Code {
        auto& [ct, num] = is_th_s[wid];

        if (start >= end) {
            num = 0;
            return Code::OK;
        }

        num = end - start;

        for (size_t cur = start; cur < end; ++cur) {
            send.at(cur).save(ct);
        }

        return Code::OK;
    };

    gemini::ThreadPool tpool(threads);
    CHECK_ERR(gemini::LaunchWorks(tpool, send.size(), serialize), "Serialize");

    size_t total = 0;
    for (auto& [stream, len] : is_th_r) {
        len = IO::recv_encrypted_vector(*ios[0], stream);
        total += len;
    }

    for (auto& [stream, len] : is_th_s) {
        IO::send_encrypted_vector(*ios[0], stream, len);
    }

    is_th_s.clear();

    if (total != send.size()) {
        Utils::log(Utils::Level::ERROR, "send_recv: Input lengths mismatch");
    }

    recv.resize(total);

    auto deserialize = [&ctx, &is_th_r, &recv](long wid, size_t start, size_t end) -> Code {
        if (start >= end)
            return Code::OK;

        auto& [stream, num] = is_th_r[wid];

        if (end - start != num)
            return Code::ERR_DIM_MISMATCH;

        for (size_t i = start; i < end; ++i) {
            recv[i].load(ctx, stream);
        }
        return Code::OK;
    };
    CHECK_ERR(gemini::LaunchWorks(tpool, total, deserialize), "deserialize");
    return Code::OK;
}

#endif