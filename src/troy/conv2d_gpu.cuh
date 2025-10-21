#ifndef CONV_GPU_CUH_
#define CONV_GPU_CUH_

#include <cstdint>
#include <vector>

#include "io/net_io_channel.hpp"

using std::vector;

namespace troy {
class HeContext;
}

namespace TROY {

std::shared_ptr<troy::HeContext> setup();

void conv2d(IO::NetIO** ios, int party, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
            size_t kw, size_t oc, size_t stride, size_t padding, bool mod_switch = false);

void conv2d_ab2(IO::NetIO** ios, int party, uint32_t* x, uint32_t* w, uint32_t* c, size_t bs,
                size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, size_t stride,
                bool mod_switch);

void conv2d_ab2_reverse(IO::NetIO** ios, int party, uint32_t* x, uint32_t* w, uint32_t* c,
                        size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
                        size_t stride, bool mod_switch);

void conv2d_ab(IO::NetIO** ios, int party, uint32_t* x, uint32_t* w, uint32_t* c, size_t bs,
               size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, size_t stride,
               bool mod_switch);

template <class T>
bool vector_equal(const vector<T>& a, const vector<T>& b);
vector<uint32_t> ideal_conv(uint32_t* x, uint32_t* w, size_t t, size_t bs, size_t ic, size_t ih,
                            size_t iw, size_t kh, size_t kw, size_t oc, size_t stride = 1);
vector<uint32_t> random_polynomial(size_t size, uint64_t max_value = (1UL << 32));

void add_inplace(std::vector<uint32_t>& a, const uint32_t* b, size_t t);
size_t apply_stride(uint32_t* dest, std::vector<uint32_t>& x, const size_t& stride,
                    const size_t& bs, const size_t& ic, const size_t& ih, const size_t& iw,
                    const size_t& kh, const size_t& kw, const size_t& oc);

template <class T>
void send(T** ios, const std::stringstream& ss);

template <class T>
std::stringstream recv(T** ios);

inline uint64_t multiply_mod(uint64_t a, uint64_t b, uint64_t t) {
    __uint128_t c = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return static_cast<uint64_t>(c % static_cast<__uint128_t>(t));
}

inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t t) { return (a + b) % t; }

template <class T>
inline void add_mod_inplace(T& a, uint64_t b, uint64_t t) {
    a = add_mod(a, b, t);
}

} // namespace TROY

template <class T>
void TROY::send(T** ios, const std::stringstream& ss) {
    auto tmp = ss.str();
    size_t n = tmp.size();
    ios[0]->send_data(&n, sizeof(size_t));
    if (n > 0) {
        ios[0]->send_data(tmp.c_str(), n);
    }
    ios[0]->flush();
}

template <class T>
std::stringstream TROY::recv(T** ios) {
    std::stringstream res;
    size_t n{0};
    ios[0]->recv_data(&n, sizeof(size_t));
    if (n > 0) {
        char* s = new char[n];
        ios[0]->recv_data(s, n);
        res = std::stringstream(std::string(s, n));
        delete[] s;
    }
    return res;
}

template <class T>
bool TROY::vector_equal(const vector<T>& a, const vector<T>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

#endif