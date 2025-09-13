#include "send.hpp"

void IO::send_encrypted_vector(IO::NetIO& io, std::stringstream& ct, const uint32_t& ncts) {
    io.send_data(&ncts, sizeof(uint32_t));
    if (ncts > 0) {
        uint64_t ct_size = ct.tellp();
        string ct_ser    = ct.str();
        io.send_data(&ct_size, sizeof(uint64_t));
        io.send_data(ct_ser.c_str(), ct_ser.size());
    }
    io.flush();
}

void IO::recv_encrypted_filters(IO::NetIO& io, const seal::SEALContext& context,
                                vector<vector<seal::Ciphertext>>& ct_vec, bool is_truncated) {
    uint32_t ncts{0};
    io.recv_data(&ncts, sizeof(uint32_t));
    if (ncts > 0) {
        ct_vec.resize(ncts);
        for (size_t i = 0; i < ncts; ++i) {
            recv_encrypted_vector(io, context, ct_vec[i], is_truncated);
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
            if (is_truncated)
                ct_vec[i].unsafe_load(context, is);
            else
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