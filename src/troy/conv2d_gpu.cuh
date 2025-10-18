#ifndef CONV_GPU_CUH_
#define CONV_GPU_CUH_

#include <cstdint>
#include <vector>

using std::vector;

namespace TROY {

void conv2d(size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
            size_t stride);
bool vector_equal(const vector<uint64_t>& a, const vector<uint64_t>& b);
vector<uint64_t> ideal_conv(uint64_t* x, uint64_t* w, uint64_t* R, size_t t, size_t bs, size_t ic,
                            size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
                            size_t stride = 1);
vector<uint64_t> random_polynomial(size_t size, uint64_t max_value = 10);
vector<uint64_t> apply_stride(std::vector<uint64_t>& x, const size_t& stride, const size_t& bs,
                              const size_t& ic, const size_t& ih, const size_t& iw,
                              const size_t& kh, const size_t& kw, const size_t& oc);

inline uint64_t multiply_mod(uint64_t a, uint64_t b, uint64_t t) {
    __uint128_t c = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return static_cast<uint64_t>(c % static_cast<__uint128_t>(t));
}

inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t t) {
    if (a + b >= t) {
        return a + b - t;
    } else {
        return a + b;
    }
}

inline void add_mod_inplace(uint64_t& a, uint64_t b, uint64_t t) { a = add_mod(a, b, t); }

} // namespace TROY

#endif