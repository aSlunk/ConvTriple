#include <chrono>
#include <sstream>

#include <troy/troy.h>

#include "constants.hpp"
#include "conv2d_gpu.cuh"

namespace TROY {

void conv2d(IO::NetIO** ios, int party, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
            size_t kw, size_t oc, size_t stride, size_t padding) {
    vector<uint32_t> x = random_polynomial(bs * ic * ih * iw);
    vector<uint32_t> w = random_polynomial(oc * ic * kh * kw);

    size_t ow = ((iw + 2 * padding - kw) / stride) + 1;
    size_t oh = ((ih + 2 * padding - kh) / stride) + 1;
    vector<uint32_t> c(bs * oc * oh * ow);

    if (!padding) {
        conv2d_ab2(ios, party, x.data(), w.data(), c.data(), bs, ic, ih, iw, kh, kw, oc, stride);
    } else {
        vector<uint32_t> dest;

        auto dim = Utils::pad_zero(x.data(), dest, ic, ih, iw, padding, bs);
        ih       = std::get<0>(dim);
        iw       = std::get<1>(dim);

        conv2d_ab2(ios, party, dest.data(), w.data(), c.data(), bs, ic, ih, iw, kh, kw, oc, stride);
    }
}

void conv2d_ab2(IO::NetIO** ios, int party, uint32_t* x, uint32_t* w, uint32_t* c, size_t bs,
                size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, size_t stride) {
    using namespace troy;
    auto start        = measure::now();
    size_t bitlen     = BIT_LEN;
    size_t poly_mod   = POLY_MOD << 1;
    size_t plain_mod  = PLAIN_MOD;
    SchemeType scheme = SchemeType::BFV;

    EncryptionParameters parms(scheme);
    parms.set_coeff_modulus(CoeffModulus::create(poly_mod, {60, 40, 40, 60}));
    parms.set_plain_modulus(plain_mod);
    parms.set_poly_modulus_degree(poly_mod);
    HeContextPointer he = HeContext::create(parms, true, SecurityLevel::Classical128, 0x123);

    // BatchEncoder encoder(he);
    linear::PolynomialEncoderRing2k<uint32_t> encoder(he, bitlen);
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    } else {
        std::cout << RED << "Couldn't find GPU" << NC << "\n";
    }

    size_t oh = ih - kh + 1;
    size_t ow = iw - kw + 1;

    linear::Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw, parms.poly_modulus_degree(),
                                linear::MatmulObjective::EncryptLeft);

    KeyGenerator keygen(he);
    Encryptor encryptor(he);
    encryptor.set_secret_key(keygen.secret_key());
    Evaluator evaluator(he);
    Decryptor decryptor(he, keygen.secret_key());

    if (party == 1) {
        linear::Cipher2d x_encrypted
            = helper.encrypt_inputs_ring2k(encryptor, encoder, x, std::nullopt);
        std::stringstream x_serialized;
        x_encrypted.save(x_serialized, he);
        send(ios, x_serialized);

        auto y_serialized = recv(ios);
        auto y_encrypted  = helper.deserialize_outputs(evaluator, y_serialized);
        vector<uint32_t> y_decrypted
            = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);
        auto res = apply_stride(y_decrypted, stride, bs, ic, ih, iw, kh, kw, oc);

#ifdef VERIFY
        std::cout << PURPLE << "Verifying CONV" << NC << "\n";
        size_t nh = (ih - kh) / stride + 1;
        size_t nw = (iw - kw) / stride + 1;
        std::cout << PURPLE << "[" << ic << ", " << ih << ", " << iw << "] x [" << ic << ", " << kh
                  << ", " << kw << "] = [" << oc << ", " << nh << ", " << nw << "]" << NC << "\n";

        std::vector<uint32_t> w(oc * ic * kh * kw);
        std::vector<uint32_t> R(bs * oc * nh * nw);

        ios[0]->recv_data(w.data(), w.size() * sizeof(uint32_t));
        ios[0]->recv_data(R.data(), R.size() * sizeof(uint32_t));

        add_inplace(R, res, plain_mod);
        std::cout << "okay\n";
        vector<uint32_t> ideal
            = ideal_conv(x, w.data(), R.data(), plain_mod, bs, ic, ih, iw, kh, kw, oc, stride);
        std::cout << "okay2\n";
        if (vector_equal(R, ideal)) {
            std::cout << GREEN << "GPU-CONV: PASSED" << NC << "\n";
        } else {
            std::cout << RED << "GPU-CONV: FAILED" << NC << "\n";
            // for (size_t h = 0; h < oh; ++h) {
            //     for (size_t w = 0; w < ow; ++w) {
            //         std::cout << R[h * ow + w] << ", ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "\n";
            // for (size_t h = 0; h < oh; ++h) {
            //     for (size_t w = 0; w < ow; ++w) {
            //         std::cout << ideal[h * ow + w] << ", ";
            //     }
            //     std::cout << "\n";
            // }
        }
#endif
    } else {
        vector<uint32_t> R = random_polynomial(bs * oc * oh * ow);

        linear::Plain2d w_encoded = helper.encode_weights_ring2k(encoder, w, std::nullopt, false);
        linear::Plain2d R_encoded = helper.encode_outputs_ring2k(encoder, R.data(), std::nullopt);

        auto stream      = recv(ios);
        auto x_encrypted = linear::Cipher2d::load_new(stream, he);

        linear::Cipher2d y_encrypted = helper.conv2d(evaluator, x_encrypted, w_encoded);
        // y_encrypted.mod_switch_to_next_inplace(evaluator);
        y_encrypted.sub_plain_inplace(evaluator, R_encoded);

        std::stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        send(ios, y_serialized);

        auto res = apply_stride(R, stride, bs, ic, ih, iw, kh, kw, oc);
#ifdef VERIFY
        ios[0]->send_data(w, oc * ic * kw * kh * sizeof(uint32_t));
        ios[0]->send_data(res.data(), res.size() * sizeof(uint32_t));
        ios[0]->flush();
#endif
        std::cout << "P" << party - 1 << ": " << (1.0 * ios[0]->counter) / (1 << 20) << "MiB\n";
    }

    double time = std::chrono::duration<double, std::milli>(measure::now() - start).count();

    std::cout << "P" << party - 1 << " conv time[s]: " << time / 1000.0 << "\n";
    std::cout << "P" << party - 1 << " conv data[MiB]: " << (1.0 * ios[0]->counter) / (1 << 20)
              << "\n";
}

std::vector<uint32_t> random_polynomial(size_t size, uint64_t max_value) {
    std::vector<uint32_t> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = rand() % max_value;
    }
    return result;
}

vector<uint32_t> ideal_conv(uint32_t* x, uint32_t* w, uint32_t* R, size_t t, size_t bs, size_t ic,
                            size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, size_t stride) {
    size_t oh = (ih - kh) / stride + 1;
    size_t ow = (iw - kw) / stride + 1;

    vector<uint32_t> y_truth(bs * oc * oh * ow, 0);

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
                    // auto old_h = (ih - kh) + 1;
                    // auto old_w = (iw - kw) + 1;
                    // add_mod_inplace(y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j],
                    //                 -R[b * oc * old_h * old_w + o * old_h * old_w
                    //                    + i * stride * old_w + j * stride],
                    //                 t);
                }
            }
        }
    }
    return y_truth;
}

vector<uint32_t> apply_stride(std::vector<uint32_t>& x, const size_t& stride, const size_t& bs,
                              const size_t& ic, const size_t& ih, const size_t& iw,
                              const size_t& kh, const size_t& kw, const size_t& oc) {
    size_t oh  = (ih - kh) + 1;
    size_t ow  = (iw - kw) + 1;
    size_t nh  = (ih - kh) / stride + 1;
    size_t nw  = (iw - kw) / stride + 1;
    auto nsize = oc * nh * nw;

    vector<uint32_t> res(bs * nsize);

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

void add_inplace(std::vector<uint32_t>& a, const std::vector<uint32_t>& b, size_t t) {
    for (size_t i = 0; i < a.size(); ++i) add_mod_inplace(a[i], b[i], t);
}

} // namespace TROY