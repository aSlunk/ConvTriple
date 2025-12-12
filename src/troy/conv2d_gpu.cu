#include <chrono>
#include <sstream>

#include "constants.hpp"
#include "conv2d_gpu.cuh"

#include <troy/troy.h>

namespace TROY {

troy::HeContextPointer setup() {
    using namespace troy;
    size_t poly_mod   = POLY_MOD;
    size_t plain_mod  = PLAIN_MOD;
    SchemeType scheme = SchemeType::BFV;

    EncryptionParameters parms(scheme);
    parms.set_coeff_modulus(CoeffModulus::create(poly_mod, {43, 33, 33}));
    parms.set_plain_modulus(plain_mod);
    parms.set_poly_modulus_degree(poly_mod);
    return HeContext::create(parms, true, SecurityLevel::Classical128, 0x42);
}

void conv2d(IO::NetIO** ios, int party, const INT_TYPE* a, const INT_TYPE* b, INT_TYPE* c,
            size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
            size_t stride, size_t padding, bool mod_switch, int factor) {
    auto start = measure::now();

    vector<INT_TYPE> dest;
    const INT_TYPE* ai;

    if (!padding) {
        ai = a;
    } else {
        auto dim = Utils::pad_zero(a, dest, ic, ih, iw, padding, bs);
        ih       = std::get<0>(dim);
        iw       = std::get<1>(dim);
        ai       = dest.data();
    }

    size_t ac_batch = bs / factor;

    auto oh = dim(ih, kh, stride, padding);
    auto ow = dim(iw, kw, stride, padding);

    size_t i_size = ac_batch * ih * iw * ic;
    size_t w_size = ic * kh * kw * oc;
    size_t c_size = ac_batch * oc * oh * ow;

    for (int i = 0; i < factor; ++i) {
#if REVERSE_GPU == 0
        conv2d_ab2(ios, party, ai + i_size * i, b + w_size * i, c + c_size * i, ac_batch, ic, ih,
                   iw, kh, kw, oc, stride, mod_switch);
#else
        conv2d_ab2_reverse(ios, party, ai + i_size * i, b + w_size * i, c + c_size * i, ac_batch,
                           ic, ih, iw, kh, kw, oc, stride, mod_switch);
#endif
    }

    double time = std::chrono::duration<double, std::milli>(measure::now() - start).count();

    std::cout << "P" << party - 1 << " conv time[s]: " << time / 1000.0 << "\n";
    std::cout << "P" << party - 1 << " conv data[MiB]: " << (1.0 * ios[0]->counter) / (1 << 20)
              << "\n";
}

void conv2d_dummy(IO::NetIO** ios, int party, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
                  size_t kw, size_t oc, size_t stride, size_t padding, bool mod_switch) {
    vector<INT_TYPE> x = random_polynomial(bs * ic * ih * iw, PLAIN_MOD);
    vector<INT_TYPE> w = random_polynomial(oc * ic * kh * kw, PLAIN_MOD);

    size_t oh = dim(ih, kh, stride, padding);
    size_t ow = dim(iw, kw, stride, padding);
    vector<INT_TYPE> c(bs * oc * oh * ow);

    conv2d(ios, party, x.data(), w.data(), c.data(), bs, ic, ih, iw, kh, kw, oc, stride, padding,
           mod_switch);
}

void conv2d_ab2(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
                size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
                size_t stride, bool mod_switch) {
    using namespace troy;
    auto he = setup();
    linear::PolynomialEncoderRing2k<INT_TYPE> encoder(he, BIT_LEN);
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    } else {
        std::cout << RED << "Couldn't find a GPU" << NC << "\n";
    }

    size_t oh = ih - kh + 1;
    size_t ow = iw - kw + 1;

    linear::Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw, POLY_MOD,
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
        vector<INT_TYPE> y_decrypted
            = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);
        [[maybe_unused]] auto size
            = apply_stride(c, y_decrypted, stride, bs, ic, ih, iw, kh, kw, oc);

#ifdef VERIFY
        std::cout << PURPLE << "Verifying CONV" << NC << "\n";
        size_t nh = dim(ih, kh, stride, 0);
        size_t nw = dim(iw, kw, stride, 0);
        std::cout << PURPLE << "[" << ic << ", " << ih << ", " << iw << "] x [" << ic << ", " << kh
                  << ", " << kw << "] = [" << oc << ", " << nh << ", " << nw << "]" << NC << "\n";

        std::vector<INT_TYPE> x2(bs * ic * ih * iw);
        std::vector<INT_TYPE> w2(oc * ic * kh * kw);
        std::vector<INT_TYPE> R(bs * oc * nh * nw);

        ios[0]->recv_data(x2.data(), bs * ic * ih * iw * sizeof(INT_TYPE));
        ios[0]->recv_data(w2.data(), w2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(R.data(), R.size() * sizeof(INT_TYPE));

        add_inplace(R, c, PLAIN_MOD);
        add_inplace(x2, x, PLAIN_MOD);
        vector<INT_TYPE> ideal
            = ideal_conv(x2.data(), w2.data(), PLAIN_MOD, bs, ic, ih, iw, kh, kw, oc, stride);
        if (vector_equal(R, ideal)) {
            std::cout << GREEN << "GPU-CONV: PASSED" << NC << "\n";
        } else {
            std::cout << RED << "GPU-CONV: FAILED" << NC << "\n";
            // for (size_t h = 0; h < nh; ++h) {
            //     for (size_t w = 0; w < nw; ++w) {
            //         std::cout << R[h * nw + w] << ", ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "\n";
            // for (size_t h = 0; h < nh; ++h) {
            //     for (size_t w = 0; w < nw; ++w) {
            //         std::cout << ideal[h * nw + w] << ", ";
            //     }
            //     std::cout << "\n";
            // }
        }
#endif
    } else {
        vector<INT_TYPE> R = random_polynomial(bs * oc * oh * ow);

        linear::Plain2d x_encoded;
        if (x)
            x_encoded = helper.encode_inputs_ring2k(encoder, x, std::nullopt, true);
        linear::Plain2d w_encoded
            = helper.encode_weights_ring2k(encoder, w, std::nullopt, false, evaluator, true);
        linear::Plain2d R_encoded = helper.encode_outputs_ring2k(encoder, R.data(), std::nullopt);

        auto stream      = recv(ios);
        auto x_encrypted = linear::Cipher2d::load_new(stream, he);

        if (x)
            x_encrypted.add_plain_inplace(evaluator, x_encoded);

        linear::Cipher2d y_encrypted = helper.conv2d(evaluator, x_encrypted, w_encoded, true);
        if (mod_switch)
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        y_encrypted.sub_plain_inplace(evaluator, R_encoded);

        std::stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        send(ios, y_serialized);

        [[maybe_unused]] auto size = apply_stride(c, R, stride, bs, ic, ih, iw, kh, kw, oc);
#ifdef VERIFY
        if (x)
            ios[0]->send_data(x, bs * ic * ih * iw * sizeof(INT_TYPE));
        else {
            std::vector<INT_TYPE> zeros(bs * ic * ih * iw, 0);
            ios[0]->send_data(zeros.data(), bs * ic * ih * iw * sizeof(INT_TYPE));
        }
        ios[0]->send_data(w, oc * ic * kw * kh * sizeof(INT_TYPE));
        ios[0]->send_data(c, size * sizeof(INT_TYPE));
        ios[0]->flush();
#endif
    }
}

void conv2d_ab(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
               size_t bs, size_t ic, size_t ih, size_t iw, size_t kh, size_t kw, size_t oc,
               size_t stride, bool mod_switch) {
    using namespace troy;
    auto he = setup();
    linear::PolynomialEncoderRing2k<INT_TYPE> encoder(he, BIT_LEN);
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    } else {
        std::cout << RED << "Couldn't find a GPU" << NC << "\n";
    }

    size_t oh = ih - kh + 1;
    size_t ow = iw - kw + 1;

    linear::Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw, POLY_MOD,
                                linear::MatmulObjective::EncryptLeft);

    KeyGenerator keygen(he);
    Encryptor encryptor(he);
    encryptor.set_secret_key(keygen.secret_key());
    Evaluator evaluator(he);
    Decryptor decryptor(he, keygen.secret_key());

    if (party == 1) {
        linear::Cipher2d x_encrypted
            = helper.encrypt_inputs_ring2k(encryptor, encoder, x, std::nullopt);

        std::stringstream a1_serialized;
        x_encrypted.save(a1_serialized, he);
        vector<INT_TYPE> R1 = random_polynomial(bs * oc * oh * ow);

        linear::Plain2d w_encoded  = helper.encode_weights_ring2k(encoder, w, std::nullopt, false);
        linear::Plain2d R1_encoded = helper.encode_outputs_ring2k(encoder, R1.data(), std::nullopt);

        send(ios, a1_serialized);
        auto a2_serialized = recv(ios);
        auto a2_encrypted  = linear::Cipher2d::load_new(a2_serialized, he);

        auto m1_encrypted = helper.conv2d(evaluator, a2_encrypted, w_encoded);
        m1_encrypted.sub_plain_inplace(evaluator, R1_encoded);

        std::stringstream m1_serialized;
        m1_encrypted.save(m1_serialized, he);

        send(ios, m1_serialized);
        auto y_serialized = recv(ios);
        auto y_encrypted  = helper.deserialize_outputs(evaluator, y_serialized);

        vector<INT_TYPE> y_decrypted
            = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);
        [[maybe_unused]] auto size
            = apply_stride(c, y_decrypted, stride, bs, ic, ih, iw, kh, kw, oc);

#ifdef VERIFY
        std::cout << PURPLE << "Verifying CONV" << NC << "\n";
        size_t nh = (ih - kh) / stride + 1;
        size_t nw = (iw - kw) / stride + 1;
        std::cout << PURPLE << "[" << ic << ", " << ih << ", " << iw << "] x [" << ic << ", " << kh
                  << ", " << kw << "] = [" << oc << ", " << nh << ", " << nw << "]" << NC << "\n";

        std::vector<INT_TYPE> w(oc * ic * kh * kw);
        std::vector<INT_TYPE> R(bs * oc * nh * nw);

        ios[0]->recv_data(w.data(), w.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(R.data(), R.size() * sizeof(INT_TYPE));

        add_inplace(R, c, PLAIN_MOD);
        vector<INT_TYPE> ideal
            = ideal_conv(x, w.data(), PLAIN_MOD, bs, ic, ih, iw, kh, kw, oc, stride);
        if (vector_equal(R, ideal)) {
            std::cout << GREEN << "GPU-CONV: PASSED" << NC << "\n";
        } else {
            std::cout << RED << "GPU-CONV: FAILED" << NC << "\n";
        }
#endif
    } else {
        linear::Cipher2d a2_encrypted
            = helper.encrypt_inputs_ring2k(encryptor, encoder, x, std::nullopt);

        std::stringstream a2_serialized;
        a2_encrypted.save(a2_serialized, he);

        vector<INT_TYPE> R2 = random_polynomial(bs * oc * oh * ow);

        linear::Plain2d w_encoded  = helper.encode_weights_ring2k(encoder, w, std::nullopt, false);
        linear::Plain2d R2_encoded = helper.encode_outputs_ring2k(encoder, R2.data(), std::nullopt);

        auto a1_serialized = recv(ios);
        send(ios, a2_serialized);

        auto a1_encrypted = linear::Cipher2d::load_new(a1_serialized, he);

        linear::Cipher2d m2_encrypted = helper.conv2d(evaluator, a1_encrypted, w_encoded);
        if (mod_switch)
            m2_encrypted.mod_switch_to_next_inplace(evaluator);
        m2_encrypted.sub_plain_inplace(evaluator, R2_encoded);

        std::stringstream m2_serialized;
        helper.serialize_outputs(evaluator, m2_encrypted, m2_serialized);

        auto m1_serialized = recv(ios);
        send(ios, m2_serialized);

        [[maybe_unused]] auto size = apply_stride(c, R2, stride, bs, ic, ih, iw, kh, kw, oc);
#ifdef VERIFY
        ios[0]->send_data(w, oc * ic * kw * kh * sizeof(INT_TYPE));
        ios[0]->send_data(c, size * sizeof(INT_TYPE));
        ios[0]->flush();
#endif
    }
}

std::vector<INT_TYPE> random_polynomial(size_t size, uint64_t max_value) {
    std::vector<INT_TYPE> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = rand() % max_value;
    }
    return result;
}

vector<INT_TYPE> ideal_conv(const INT_TYPE* x, const INT_TYPE* w, size_t t, size_t bs, size_t ic,
                            size_t ih, size_t iw, size_t kh, size_t kw, size_t oc, size_t stride) {
    size_t oh = (ih - kh) / stride + 1;
    size_t ow = (iw - kw) / stride + 1;

    vector<INT_TYPE> y_truth(bs * oc * oh * ow, 0);

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
                }
            }
        }
    }
    return y_truth;
}

size_t apply_stride(INT_TYPE* dest, const std::vector<INT_TYPE>& x, const size_t& stride,
                    const size_t& bs, const size_t& ic, const size_t& ih, const size_t& iw,
                    const size_t& kh, const size_t& kw, const size_t& oc) {
    size_t oh  = (ih - kh) + 1;
    size_t ow  = (iw - kw) + 1;
    size_t nh  = (ih - kh) / stride + 1;
    size_t nw  = (iw - kw) / stride + 1;
    auto nsize = oc * nh * nw;

    for (size_t b = 0; b < bs; ++b) {
        for (size_t c = 0; c < oc; ++c) {
            for (size_t h = 0; h < oh; h += stride) {
                for (size_t w = 0; w < ow; w += stride) {
                    size_t out_h = h / stride;
                    size_t out_w = w / stride;
                    dest[b * nsize + c * nh * nw + out_h * nw + out_w]
                        = x[b * oc * oh * ow + c * oh * ow + h * ow + w];
                }
            }
        }
    }
    return bs * nsize;
}

void add_inplace(std::vector<INT_TYPE>& a, const INT_TYPE* b, size_t t) {
    for (size_t i = 0; i < a.size(); ++i) add_mod_inplace(a[i], b[i], t);
}

void conv2d_ab2_reverse(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w,
                        INT_TYPE* c, size_t bs, size_t ic, size_t ih, size_t iw, size_t kh,
                        size_t kw, size_t oc, size_t stride, bool mod_switch) {
    using namespace troy;
    auto he = setup();
    linear::PolynomialEncoderRing2k<INT_TYPE> encoder(he, BIT_LEN);
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    } else {
        std::cout << RED << "Couldn't find a GPU" << NC << "\n";
    }

    size_t oh = ih - kh + 1;
    size_t ow = iw - kw + 1;

    linear::Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw, POLY_MOD,
                                linear::MatmulObjective::EncryptRight);

    KeyGenerator keygen(he);
    Encryptor encryptor(he);
    encryptor.set_secret_key(keygen.secret_key());
    Evaluator evaluator(he);
    Decryptor decryptor(he, keygen.secret_key());

    if (party == 2) {
        linear::Cipher2d w_encrypted
            = helper.encrypt_weights_ring2k(encryptor, encoder, w, std::nullopt);
        std::stringstream w_serialized;
        w_encrypted.save(w_serialized, he);
        send(ios, w_serialized);

        auto y_serialized = recv(ios);
        auto y_encrypted  = helper.deserialize_outputs(evaluator, y_serialized);
        vector<INT_TYPE> y_decrypted
            = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);
        [[maybe_unused]] auto size
            = apply_stride(c, y_decrypted, stride, bs, ic, ih, iw, kh, kw, oc);

#ifdef VERIFY
        std::cout << PURPLE << "Verifying CONV REVERSED" << NC << "\n";
        size_t nh = (ih - kh) / stride + 1;
        size_t nw = (iw - kw) / stride + 1;
        std::cout << PURPLE << "[" << ic << ", " << ih << ", " << iw << "] x [" << ic << ", " << kh
                  << ", " << kw << "] = [" << oc << ", " << nh << ", " << nw << "]" << NC << "\n";

        std::vector<INT_TYPE> x2(bs * ic * ih * iw);
        std::vector<INT_TYPE> R(size);

        ios[0]->recv_data(x2.data(), x2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(R.data(), R.size() * sizeof(INT_TYPE));

        add_inplace(R, c, PLAIN_MOD);
        vector<INT_TYPE> ideal
            = ideal_conv(x2.data(), w, PLAIN_MOD, bs, ic, ih, iw, kh, kw, oc, stride);
        if (vector_equal(R, ideal)) {
            std::cout << GREEN << "GPU-CONV: PASSED" << NC << "\n";
        } else {
            std::cout << RED << "GPU-CONV: FAILED" << NC << "\n";
            // for (size_t h = 0; h < nh; ++h) {
            //     for (size_t w = 0; w < nw; ++w) {
            //         std::cout << R[h * nw + w] << ", ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "\n";
            // for (size_t h = 0; h < nh; ++h) {
            //     for (size_t w = 0; w < nw; ++w) {
            //         std::cout << ideal[h * nw + w] << ", ";
            //     }
            //     std::cout << "\n";
            // }
        }
#endif
    } else {
        vector<INT_TYPE> R = random_polynomial(bs * oc * oh * ow);

        linear::Plain2d x_encoded = helper.encode_inputs_ring2k(encoder, x, std::nullopt, false);
        linear::Plain2d R_encoded = helper.encode_outputs_ring2k(encoder, R.data(), std::nullopt);

        auto stream      = recv(ios);
        auto w_encrypted = linear::Cipher2d::load_new(stream, he);

        linear::Cipher2d y_encrypted = helper.conv2d_reverse(evaluator, x_encoded, w_encrypted);
        if (mod_switch)
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        y_encrypted.sub_plain_inplace(evaluator, R_encoded);

        std::stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        send(ios, y_serialized);

        [[maybe_unused]] auto size = apply_stride(c, R, stride, bs, ic, ih, iw, kh, kw, oc);
#ifdef VERIFY
        ios[0]->send_data(x, bs * ic * ih * iw * sizeof(INT_TYPE));
        ios[0]->send_data(c, size * sizeof(INT_TYPE));
        ios[0]->flush();
#endif
    }
}

} // namespace TROY