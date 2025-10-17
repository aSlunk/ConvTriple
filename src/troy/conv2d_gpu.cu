#include "conv2d_gpu.cuh"

#include "troy/troy.h"

namespace TROY {

void conv2d() {
    using namespace troy;
    size_t poly_mod = 4096;
    size_t plain_mod = 1lu << 32;
    SchemeType scheme = SchemeType::BFV;
        
    EncryptionParameters parms(scheme);
    parms.set_coeff_modulus(CoeffModulus::create(poly_mod, { 60, 40 }).to_vector());
    parms.set_plain_modulus(plain_mod);
    parms.set_poly_modulus_degree(poly_mod);
    HeContextPointer he = HeContext::create(parms, true, SecurityLevel::Classical128);

    BatchEncoder encoder(he);
    std::cout << utils::device_count() << "\n";
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    }
}

} // namespace TROY