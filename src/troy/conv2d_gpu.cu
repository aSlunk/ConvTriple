#include <sstream>

#include "conv2d_gpu.cuh"

#include "troy/troy.h"

namespace TROY {

void conv2d(size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
            size_t stride) {
    std::cout << "conv\n";
    using namespace troy;
    size_t poly_mod   = 4096;
    size_t plain_mod  = 1lu << 32;
    SchemeType scheme = SchemeType::BFV;

    EncryptionParameters parms(scheme);
    parms.set_coeff_modulus(CoeffModulus::create(poly_mod, {60, 40}).to_vector());
    parms.set_plain_modulus(plain_mod);
    parms.set_poly_modulus_degree(poly_mod);
    HeContextPointer he = HeContext::create(parms, true, SecurityLevel::Classical128);

    BatchEncoder encoder(he);
    std::cout << utils::device_count() << "\n";
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    }

    uint64_t mod = parms.plain_modulus_host().value();
    size_t oh    = ih - kh + 1;
    size_t ow    = iw - kw + 1;

    vector<uint64_t> x = random_polynomial(bs * ic * ih * iw);
    vector<uint64_t> w = random_polynomial(oc * ic * kh * kw);
    vector<uint64_t> R = random_polynomial(bs * oc * oh * ow);

    linear::Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw, parms.poly_modulus_degree(),
                                linear::MatmulObjective::EncryptLeft);

    KeyGenerator keygen(he);
    Encryptor encryptor(he);
    encryptor.set_secret_key(keygen.secret_key());
    Evaluator evaluator(he);
    Decryptor decryptor(he, keygen.secret_key());

    linear::Plain2d x_encoded = helper.encode_inputs_uint64s(encoder, x.data());
    linear::Plain2d w_encoded = helper.encode_weights_uint64s(encoder, w.data());
    linear::Plain2d R_encoded = helper.encode_outputs_uint64s(encoder, R.data());

    linear::Cipher2d x_encrypted = x_encoded.encrypt_symmetric(encryptor);

    std::stringstream x_serialized;
    x_encrypted.save(x_serialized, he);
    x_encrypted = linear::Cipher2d::load_new(x_serialized, he);

    linear::Cipher2d y_encrypted = helper.conv2d(evaluator, x_encrypted, w_encoded);
    y_encrypted.sub_plain_inplace(evaluator, R_encoded);

    std::stringstream y_serialized;
    helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
    y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

    vector<uint64_t> y_decrypted = helper.decrypt_outputs_uint64s(encoder, decryptor, y_encrypted);
    vector<uint64_t> y_stride    = apply_stride(y_decrypted, stride, bs, ic, ih, iw, kh, kw, oc);

    vector<uint64_t> idea
        = ideal_conv(x.data(), w.data(), R.data(), mod, bs, ic, ih, iw, kh, kw, oc, stride);

    if (vector_equal(y_stride, idea)) {
        std::cout << "SUCCESS\n";
    } else {
        std::cout << "FAILED\n";
    }
}

std::vector<uint64_t> random_polynomial(size_t size, uint64_t max_value) {
    std::vector<uint64_t> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = rand() % max_value;
    }
    return result;
}

bool vector_equal(const vector<uint64_t>& a, const vector<uint64_t>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

vector<uint64_t> ideal_conv(uint64_t* x, uint64_t* w, uint64_t* R, size_t t, size_t bs, size_t ic,
                            size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, size_t stride) {
    size_t oh = (ih - kh) / stride + 1;
    size_t ow = (iw - kw) / stride + 1;

    vector<uint64_t> y_truth(bs * oc * oh * ow, 0);

    for (size_t b = 0; b < bs; b++) {
        for (size_t o = 0; o < oc; o++) {
            for (size_t i = 0; i < oh; i++) {
                for (size_t j = 0; j < ow; j++) {
                    for (size_t c = 0; c < ic; c++) {
                        for (size_t p = 0; p < kh; p++) {
                            for (size_t q = 0; q < kw; q++) {
                                add_mod_inplace(
                                    y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j],
                                    multiply_mod(x[b * ic * ih * iw + c * ih * iw
                                                   + (i * stride + p) * iw + (j * stride + q)],
                                                 w[o * ic * kh * kw + c * kh * kw + p * kw + q], t),
                                    t);
                            }
                        }
                    }
                    auto old_h = (ih - kh) + 1;
                    auto old_w = (iw - kw) + 1;
                    add_mod_inplace(y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j],
                                    -R[b * oc * old_h * old_w + o * old_h * old_w
                                       + i * stride * old_w + j * stride],
                                    t);
                }
            }
        }
    }
    return y_truth;
}

vector<uint64_t> apply_stride(std::vector<uint64_t>& x, const size_t& stride, const size_t& bs,
                              const size_t& ic, const size_t& ih, const size_t& iw,
                              const size_t& kh, const size_t& kw, const size_t& oc) {
    size_t oh  = (ih - kh) + 1;
    size_t ow  = (iw - kw) + 1;
    size_t nh  = (ih - kh) / stride + 1;
    size_t nw  = (iw - kw) / stride + 1;
    auto nsize = oc * nh * nw;

    vector<uint64_t> res(bs * nsize);

    for (size_t b = 0; b < bs; ++b) {
        for (size_t c = 0; c < oc; ++c) {
            for (size_t h = 0; h < oh; h += stride) {
                for (size_t w = 0; w < ow; w += stride) {
                    size_t out_h = h / stride;
                    size_t out_w = w / stride;
                    res[b * nsize + c * nh * nw + out_h * nw + out_w]
                        = x[b * oc * oh * ow + c * oh * ow + h * ow + w];
                }
            }
        }
    }
    return res;
}

} // namespace TROY