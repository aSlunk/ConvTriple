#pragma once

#include <chrono>
#include <string>

#include <seal/seal.h>
#include <sys/socket.h>

#include "io/net_io_channel.hpp"

using std::string;

namespace IO {

using Unit = std::chrono::milliseconds;
using CLK  = std::chrono::high_resolution_clock;

template <class CtType>
void send_ciphertext(IO::NetIO& io, const CtType& ct);

template <class EncVecCtType>
void send_encrypted_vector(IO::NetIO& io, const EncVecCtType& ct_vec);

void send_encrypted_vector(IO::NetIO& io, const std::stringstream& ct, const uint32_t& size);

template <class EncVecCtType>
void send_encrypted_filters(IO::NetIO& io, const EncVecCtType& ct_vec); // ADDED

void recv_encrypted_filters(IO::NetIO& io, const seal::SEALContext& context,
                            std::vector<std::vector<seal::Ciphertext>>& ct_vec,
                            bool is_truncated = false); // ADDED

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

    auto start = CLK::now();
    std::stringstream os;
    uint64_t ct_size;
    for (size_t i = 0; i < ncts; ++i) {
        ct_vec.at(i).save(os);
        // send_ciphertext(io, ct_vec.at(i));
    }
    auto cur = std::chrono::duration_cast<Unit>(CLK::now() - start).count();
    if (!io.is_server)
        std::cerr << "save: " << cur << "\n";
    start         = CLK::now();
    ct_size       = os.tellp();
    string ct_ser = os.str();
    io.send_data(&ct_size, sizeof(uint64_t));
    io.send_data(ct_ser.c_str(), ct_ser.size());
    io.flush();
    cur = std::chrono::duration_cast<Unit>(CLK::now() - start).count();
    if (!io.is_server)
        std::cerr << "send: " << cur << "\n";
}

void send_encrypted_vector(IO::NetIO& io, std::stringstream& ct, const uint32_t& ncts) {
    io.send_data(&ncts, sizeof(uint32_t));
    uint64_t ct_size = ct.str().size();
    io.send_data(&ct_size, sizeof(uint64_t));
    io.send_data(ct.str().c_str(), ct_size);
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
        is.write(c_enc_result, ct_size);
        delete[] c_enc_result;
    }
    return ncts;
}

void recv_encrypted_vector(IO::NetIO& io, const seal::SEALContext& context,
                           std::vector<seal::Ciphertext>& ct_vec, bool is_truncated) {
    uint32_t ncts{0};
    auto start = CLK::now();
    io.recv_data(&ncts, sizeof(uint32_t));
    auto cur = std::chrono::duration_cast<Unit>(CLK::now() - start).count();
    if (!io.is_server)
        std::cerr << "recv: " << cur << "\n";
    if (ncts > 0) {
        ct_vec.resize(ncts);
        uint64_t ct_size;
        io.recv_data(&ct_size, sizeof(uint64_t));
        char* c_enc_result = new char[ct_size];
        io.recv_data(c_enc_result, ct_size);
        start = CLK::now();
        std::istringstream is(std::string(c_enc_result, ct_size));
        // is.write(c_enc_result, ct_size);
        for (size_t i = 0; i < ncts; ++i) {
            // recv_ciphertext(io, context, ct_vec[i], is_truncated);
            ct_vec[i].load(context, is);
        }
        delete[] c_enc_result;
        cur = std::chrono::duration_cast<Unit>(CLK::now() - start).count();
        if (!io.is_server)
            std::cerr << "load: " << cur << "\n";
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

} // namespace IO
