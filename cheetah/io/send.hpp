#pragma once

#include <cassert>
#include <chrono>
#include <string>
#include <vector>

#include <seal/seal.h>
#include <sys/socket.h>

#include "io/net_io_channel.hpp"
#include <gemini/cheetah/hom_conv2d_ss.h>

using std::string;

namespace IO {

using Unit = std::chrono::milliseconds;
using CLK  = std::chrono::high_resolution_clock;

template <class CtType>
void send_ciphertext(IO::NetIO& io, const CtType& ct);

template <class EncVecCtType>
void send_encrypted_vector(IO::NetIO& io, const EncVecCtType& ct_vec);

template <class EncVecCtType>
void send_encrypted_vector(std::vector<IO::NetIO>& io, const EncVecCtType& ct_vec);

void send_encrypted_vector(IO::NetIO& io, const std::stringstream& ct, const uint32_t& size);

template <class EncVecCtType>
void send_encrypted_filters(IO::NetIO& io, const EncVecCtType& ct_vec); // ADDED

void recv_encrypted_filters(IO::NetIO& io, const seal::SEALContext& context,
                            std::vector<std::vector<seal::Ciphertext>>& ct_vec,
                            bool is_truncated = false); // ADDED

void recv_encrypted_vector(std::vector<IO::NetIO>& io, const seal::SEALContext& context,
                           std::vector<seal::Ciphertext>& ct_vec, bool is_truncated = false);

void recv_encrypted_vector(IO::NetIO& io, const seal::SEALContext& context,
                           std::vector<seal::Ciphertext>& ct_vec, bool is_truncated = false);

uint32_t recv_encrypted_vector(IO::NetIO& io, std::stringstream& is);

void recv_ciphertext(IO::NetIO& io, const seal::SEALContext& context, seal::Ciphertext& ct,
                     bool is_truncated);

template <class CtType>
void send_ciphertext(IO::NetIO& io, const CtType& ct) {
    std::stringstream os;
    uint64_t ct_size;
    ct.save(os);
    ct_size       = os.tellp();
    string ct_ser = os.str();
    io.send_data(&ct_size, sizeof(uint64_t));
    io.send_data(ct_ser.c_str(), ct_ser.size());
}

template <class EncVecCtType>
void send_encrypted_vector(IO::NetIO& io, const EncVecCtType& ct_vec) {
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

void send_encrypted_vector(IO::NetIO& io, std::stringstream& ct, const uint32_t& ncts) {
    io.send_data(&ncts, sizeof(uint32_t));
    io.flush();
    uint64_t ct_size = ct.tellp();
    string ct_ser    = ct.str();
    io.send_data(&ct_size, sizeof(uint64_t));
    io.flush();
    io.send_data(ct_ser.c_str(), ct_ser.size());
    io.flush();
    ct.clear();
}

template <class EncVecCtType>
void send_encrypted_filters(IO::NetIO& io, const EncVecCtType& ct_vec) {
    uint32_t ncts = ct_vec.size();
    io.send_data(&ncts, sizeof(uint32_t));
    for (size_t i = 0; i < ncts; ++i) {
        send_encrypted_vector(io, ct_vec[i]);
    }
}

void recv_encrypted_filters(IO::NetIO& io, const seal::SEALContext& context,
                            std::vector<std::vector<seal::Ciphertext>>& ct_vec, bool is_truncated) {
    uint32_t ncts{0};
    io.recv_data(&ncts, sizeof(uint32_t));
    if (ncts > 0) {
        ct_vec.resize(ncts);
        for (size_t i = 0; i < ncts; ++i) {
            recv_encrypted_vector(io, context, ct_vec[i]);
        }
    }
}

uint32_t recv_encrypted_vector(IO::NetIO& io, std::stringstream& is) {
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

void recv_encrypted_vector(IO::NetIO& io, const seal::SEALContext& context,
                           std::vector<seal::Ciphertext>& ct_vec, bool is_truncated) {
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

void recv_ciphertext(IO::NetIO& io, const seal::SEALContext& context, seal::Ciphertext& ct,
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
void send_encrypted_vector(std::vector<IO::NetIO>& ios, const EncVecCtType& ct_vec) {
    uint32_t ncts = ct_vec.size();
    ios[0].send_data(&ncts, sizeof(uint32_t));

    auto program = [&](long wid, size_t start, size_t end) -> Code {
        auto& io = ios[wid];
        for (size_t i = start; i < end; ++i) {
            send_ciphertext(io, ct_vec.at(i));
            io.flush();
        }
        io.flush();
        return Code::OK;
    };

    gemini::ThreadPool tpool(ios.size());
    gemini::LaunchWorks(tpool, ncts, program);
}

void recv_encrypted_vector(std::vector<IO::NetIO>& ios, const seal::SEALContext& context,
                           std::vector<seal::Ciphertext>& ct_vec, bool is_truncated) {
    uint32_t ncts{0};
    ios[0].recv_data(&ncts, sizeof(uint32_t));
    ct_vec.resize(ncts);

    auto program = [&](long wid, size_t start, size_t end) -> Code {
        auto& io = ios[wid];
        for (size_t i = start; i < end; ++i) {
            recv_ciphertext(io, context, ct_vec.at(i), is_truncated);
        }
        return Code::OK;
    };

    if (ncts > 0) {
        gemini::ThreadPool tpool(ios.size());
        gemini::LaunchWorks(tpool, ncts, program);
    }
}

template <class PKey>
void send_pkey(IO::NetIO& io, const PKey& pkey) {
    std::stringstream is;
    pkey.save(is);
    size_t len = is.str().size();
    io.send_data(&len, sizeof(len));
    io.send_data(is.str().data(), len);
    io.flush();
}

template <class PKey>
void recv_pkey(IO::NetIO& io, const seal::SEALContext& context, PKey& pkey) {
    size_t len{0};
    io.recv_data(&len, sizeof(len));
    char* recv = new char[len];
    io.recv_data(recv, len);

    std::stringstream is;
    is.write(recv, len);
    pkey.load(context, is);
    delete[] recv;
}

} // namespace IO