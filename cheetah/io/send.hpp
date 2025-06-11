#pragma once

#include <string>

#include <seal/seal.h>
#include <sys/socket.h>

#include "io/net_io_channel.hpp"

using std::string;

namespace IO {

template <class CtType>
void send_ciphertext(IO::NetIO& io, const CtType& ct);

template <class EncVecCtType>
void send_encrypted_vector(IO::NetIO& io, const EncVecCtType& ct_vec);

template <class EncVecCtType>
void send_encrypted_filters(IO::NetIO& io, const EncVecCtType& ct_vec); // ADDED

void recv_encrypted_filters(IO::NetIO& io, const seal::SEALContext& context,
                            std::vector<std::vector<seal::Ciphertext>>& ct_vec,
                            bool is_truncated = false); // ADDED

void recv_encrypted_vector(IO::NetIO& io, const seal::SEALContext& context,
                           std::vector<seal::Ciphertext>& ct_vec, bool is_truncated = false);

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
    for (size_t i = 0; i < ncts; ++i) {
        send_ciphertext(io, ct_vec.at(i));
    }
    io.flush();
}

template <class EncVecCtType>
void send_encrypted_filters(IO::NetIO& io, const EncVecCtType& ct_vec) {
    uint32_t ncts = ct_vec.size();
    io.send_data(&ncts, sizeof(uint32_t));
    for (size_t i = 0; i < ncts; ++i) {
        send_encrypted_vector(io, ct_vec[i]);
    }
    io.flush();
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

void recv_encrypted_vector(IO::NetIO& io, const seal::SEALContext& context,
                           std::vector<seal::Ciphertext>& ct_vec, bool is_truncated) {
    uint32_t ncts{0};
    io.recv_data(&ncts, sizeof(uint32_t));
    if (ncts > 0) {
        ct_vec.resize(ncts);
        for (size_t i = 0; i < ncts; ++i) {
            recv_ciphertext(io, context, ct_vec[i], is_truncated);
        }
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