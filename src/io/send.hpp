#ifndef SEND_HPP_
#define SEND_HPP_

#include <cassert>
#include <chrono>
#include <string>
#include <vector>

#include <seal/seal.h>
#include <sys/socket.h>

#include "net_io_channel.hpp"
#include <gemini/cheetah/hom_conv2d_ss.h>

using std::string;
using std::vector;

namespace IO {

using Unit = std::chrono::milliseconds;
using CLK  = std::chrono::high_resolution_clock;

template <class EncVec>
Code send_recv(const seal::SEALContext& ctx, IO::NetIO** ios, std::vector<EncVec>& send,
               vector<vector<seal::Ciphertext>>& recv, const size_t& threads = 1);

template <class EncVec>
Code send_recv(const seal::SEALContext& ctx, IO::NetIO** ios, EncVec& send,
               vector<seal::Ciphertext>& recv, const size_t& threads = 1);

template <class EncVec>
Code send_recv(const seal::SEALContext& ctx, vector<IO::NetIO>& ios, EncVec& send,
               vector<seal::Ciphertext>& recv);

template <class Vec>
Code recv_send(const seal::SEALContext& ctx, IO::NetIO** ios, const vector<Vec>& send,
               vector<vector<seal::Ciphertext>>& recv, const size_t& threads = 1);

template <class Vec>
Code recv_send(const seal::SEALContext& ctx, IO::NetIO** ios, const Vec& send,
               vector<seal::Ciphertext>& recv, const size_t& threads = 1);

template <class Vec>
Code recv_send(const seal::SEALContext& ctx, vector<IO::NetIO>& ios, const Vec& send,
               vector<seal::Ciphertext>& recv);

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

void IO::send_encrypted_vector(IO::NetIO& io, std::stringstream& ct, const uint32_t& ncts) {
    io.send_data(&ncts, sizeof(uint32_t));
    uint64_t ct_size = ct.tellp();
    string ct_ser    = ct.str();
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

void IO::recv_encrypted_filters(IO::NetIO& io, const seal::SEALContext& context,
                                vector<vector<seal::Ciphertext>>& ct_vec, bool is_truncated) {
    uint32_t ncts{0};
    io.recv_data(&ncts, sizeof(uint32_t));
    if (ncts > 0) {
        ct_vec.resize(ncts);
        for (size_t i = 0; i < ncts; ++i) {
            recv_encrypted_vector(io, context, ct_vec[i]);
        }
    }
}

uint32_t IO::recv_encrypted_vector(IO::NetIO& io, std::stringstream& is) {
    uint32_t ncts{0};
    io.recv_data(&ncts, sizeof(uint32_t));
    if (ncts > 0) {
        uint64_t ct_size;
        io.recv_data(&ct_size, sizeof(uint64_t));
        char* c_enc_result = new char[ct_size];
        io.recv_data(c_enc_result, ct_size);
        is = std::stringstream(std::string(c_enc_result, ct_size));
        delete[] c_enc_result;
    }
    return ncts;
}

void IO::recv_encrypted_vector(IO::NetIO& io, const seal::SEALContext& context,
                               vector<seal::Ciphertext>& ct_vec, bool is_truncated) {
    uint32_t ncts{0};
    io.recv_data(&ncts, sizeof(uint32_t));
    if (ncts > 0) {
        ct_vec.resize(ncts);
        uint64_t ct_size;
        io.recv_data(&ct_size, sizeof(uint64_t));
        char* c_enc_result = new char[ct_size];
        io.recv_data(c_enc_result, ct_size);
        std::istringstream is(std::string(c_enc_result, ct_size));
        for (size_t i = 0; i < ncts; ++i) {
            ct_vec[i].load(context, is);
        }
        delete[] c_enc_result;
    }
}

void IO::recv_ciphertext(IO::NetIO& io, const seal::SEALContext& context, seal::Ciphertext& ct,
                         bool is_truncated) {
    std::stringstream is;
    uint64_t ct_size;
    io.recv_data(&ct_size, sizeof(uint64_t));
    char* c_enc_result = new char[ct_size];
    io.recv_data(c_enc_result, ct_size);
    is.write(c_enc_result, ct_size);

    if (is_truncated) {
        ct.unsafe_load(context, is);
    } else {
        ct.load(context, is);
    }
    delete[] c_enc_result;
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

void IO::recv_encrypted_vector(vector<IO::NetIO>& ios, const seal::SEALContext& context,
                               vector<seal::Ciphertext>& ct_vec, bool is_truncated) {
    NetIO** ios_c = new NetIO*[ios.size()];
    for (size_t i = 0; i < ios.size(); ++i) {
        ios_c[i] = &ios[i];
    }
    recv_encrypted_vector(ios_c, context, ct_vec, ios.size(), is_truncated);
    delete[] ios_c;
}

void IO::recv_encrypted_vector(IO::NetIO** ios, const seal::SEALContext& context,
                               vector<seal::Ciphertext>& ct_vec, const size_t& threads,
                               bool is_truncated) {
    uint32_t ncts{0};
    ios[0]->recv_data(&ncts, sizeof(uint32_t));
    ct_vec.resize(ncts);

    auto program = [&](long wid, size_t start, size_t end) -> Code {
        auto& io = *(ios[wid]);
        for (size_t i = start; i < end; ++i) {
            recv_ciphertext(io, context, ct_vec.at(i), is_truncated);
        }
        return Code::OK;
    };

    if (ncts > 0) {
        gemini::ThreadPool tpool(threads);
        gemini::LaunchWorks(tpool, ncts, program);
    }
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
Code IO::send_recv(const seal::SEALContext& ctx, IO::NetIO** ios, std::vector<EncVec>& send,
                   vector<vector<seal::Ciphertext>>& recv, const size_t& threads) {
    EncVec to_send;
    vector<seal::Ciphertext> to_recv;

    for (size_t i = 0; i < send.size(); i++) {
        for (size_t j = 0; j < send[i].size(); j++) {
            to_send.push_back(send[i][j]);
        }
    }

    send_recv(ctx, ios, to_send, to_recv, threads);

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
Code IO::send_recv(const seal::SEALContext& ctx, vector<IO::NetIO>& ios, EncVec& send,
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
Code IO::send_recv(const seal::SEALContext& ctx, IO::NetIO** ios, EncVec& send,
                   vector<seal::Ciphertext>& recv, const size_t& threads) {
    vector<vector<seal::Ciphertext>> result(threads, vector<seal::Ciphertext>(0));

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

#endif