#ifndef ELEM_HPP_
#define ELEM_HPP_

#include <vector>

#include <seal/seal.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rlwe.h>

#include <core/utils.hpp>

#include <io/send.hpp>

#include <ot/prg.h>

using std::vector;

constexpr int64_t slot_count       = POLY_MOD;
constexpr uint64_t SMUDGING_BITLEN = 100 - BIT_LEN;

int64_t neg_mod(int64_t val, int64_t mod) { return ((val % mod) + mod) % mod; }

vector<uint64_t> ideal_functionality(vector<uint64_t>& inArr, vector<uint64_t>& multArr) {
    vector<uint64_t> result(inArr.size(), 0ULL);

    for (size_t i = 0; i < inArr.size(); i++) {
        result[i] = multArr[i] * inArr[i];
    }
    return result;
}

void set_poly_coeffs_uniform(uint64_t* poly, uint32_t bitlen,
                             std::shared_ptr<seal::UniformRandomGenerator> random,
                             std::shared_ptr<const seal::SEALContext::ContextData>& context_data) {
    assert(bitlen < 128 && bitlen > 0);
    auto& parms            = context_data->parms();
    auto& coeff_modulus    = parms.coeff_modulus();
    size_t coeff_count     = parms.poly_modulus_degree();
    size_t coeff_mod_count = coeff_modulus.size();
    uint64_t bitlen_mask   = (1ULL << (bitlen % 64)) - 1;

    seal::RandomToStandardAdapter engine(random);
    for (size_t i = 0; i < coeff_count; i++) {
        if (bitlen < 64) {
            uint64_t noise = (uint64_t(engine()) << 32) | engine();
            noise &= bitlen_mask;
            for (size_t j = 0; j < coeff_mod_count; j++) {
                poly[i + (j * coeff_count)]
                    = seal::util::barrett_reduce_64(noise, coeff_modulus[j]);
            }
        } else {
            uint64_t noise[2]; // LSB || MSB
            for (int j = 0; j < 2; j++) {
                noise[0] = (uint64_t(engine()) << 32) | engine();
                noise[1] = (uint64_t(engine()) << 32) | engine();
            }
            noise[1] &= bitlen_mask;
            for (size_t j = 0; j < coeff_mod_count; j++) {
                poly[i + (j * coeff_count)]
                    = seal::util::barrett_reduce_128(noise, coeff_modulus[j]);
            }
        }
    }
}

void flood_ciphertext(seal::Ciphertext& ct,
                      std::shared_ptr<const seal::SEALContext::ContextData>& context_data,
                      uint32_t noise_len,
                      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    auto& parms            = context_data->parms();
    auto& coeff_modulus    = parms.coeff_modulus();
    size_t coeff_count     = parms.poly_modulus_degree();
    size_t coeff_mod_count = coeff_modulus.size();

    auto noise(seal::util::allocate_poly(coeff_count, coeff_mod_count, pool));
    std::shared_ptr<seal::UniformRandomGenerator> random(parms.random_generator()->create());

    set_poly_coeffs_uniform(noise.get(), noise_len, random, context_data);
    for (size_t i = 0; i < coeff_mod_count; i++) {
        seal::util::add_poly_coeffmod(noise.get() + (i * coeff_count),
                                      ct.data() + (i * coeff_count), coeff_count, coeff_modulus[i],
                                      ct.data() + (i * coeff_count));
    }

    set_poly_coeffs_uniform(noise.get(), noise_len, random, context_data);
    for (size_t i = 0; i < coeff_mod_count; i++) {
        seal::util::add_poly_coeffmod(noise.get() + (i * coeff_count),
                                      ct.data(1) + (i * coeff_count), coeff_count, coeff_modulus[i],
                                      ct.data(1) + (i * coeff_count));
    }
}

void elemwise_product(seal::SEALContext* context, IO::NetIO* io, seal::Encryptor* encryptor,
                      seal::Decryptor* decryptor, int32_t size, uint64_t* inArr, uint64_t* multArr,
                      uint64_t* outputArr, uint64_t prime_mod, int party) {
    using namespace seal;
    auto encoder   = new BatchEncoder(*context);
    auto evaluator = new Evaluator(*context);

    int num_ct = ceil(float(size) / slot_count);

    if (party == emp::BOB) {
        vector<Ciphertext> ct(num_ct);
        for (int i = 0; i < num_ct; i++) {
            int offset = i * slot_count;
            vector<uint64_t> tmp_vec(slot_count, 0);
            Plaintext tmp_pt;
            for (int j = 0; j < slot_count && j + offset < size; j++) {
                tmp_vec[j] = inArr[j + offset] % prime_mod;
            }
            encoder->encode(tmp_vec, tmp_pt);
            encryptor->encrypt_symmetric(tmp_pt, ct[i]);
        }
        IO::send_encrypted_vector(*io, ct);
        io->flush();

        vector<Ciphertext> enc_result(num_ct);
        IO::recv_encrypted_vector(*io, *context, enc_result);
        for (int i = 0; i < num_ct; i++) {
            int offset = i * slot_count;
            vector<uint64_t> tmp_vec(slot_count, 0);
            Plaintext tmp_pt;
            decryptor->decrypt(enc_result[i], tmp_pt);
            encoder->decode(tmp_pt, tmp_vec);
            for (int j = 0; j < slot_count && j + offset < size; j++) {
                outputArr[j + offset] = tmp_vec[j];
            }
        }

#ifdef VERIFY
        Utils::log(Utils::Level::DEBUG, "VERIFIYING ELEM. MULT");
        std::vector<uint64_t> a(size);
        std::vector<uint64_t> b(size);
        std::vector<uint64_t> R(size);
        io->recv_data(a.data(), size * sizeof(uint64_t));
        io->recv_data(b.data(), size * sizeof(uint64_t));
        io->recv_data(R.data(), size * sizeof(uint64_t));

        for (int i = 0; i < size; ++i) {
            a[i] += inArr[i];
        }
        auto result = ideal_functionality(a, b);
        bool passed = true;
        for (int i = 0; i < size; i++) {
            if (((outputArr[i] + R[i]) % prime_mod) != result[i] % prime_mod) {
                passed = false;
            }
        }
        if (passed)
            Utils::log(Utils::Level::PASSED, "ELEM. MULT.: PASSED");
        else
            Utils::log(Utils::Level::PASSED, "ELEM. MULT.: FAILED");
#endif
    } else // party == ALICE
    {
        vector<Plaintext> multArr_pt(num_ct);
        vector<Plaintext> inArr_pt(num_ct);

        for (int i = 0; i < num_ct; i++) {
            int offset = i * slot_count;
            vector<uint64_t> tmp_vec(slot_count, 0);
            vector<uint64_t> tmp_vec2(slot_count, 0);
            for (int j = 0; j < slot_count && j + offset < size; j++) {
                tmp_vec[j]  = multArr[j + offset] % prime_mod;
                tmp_vec2[j] = inArr[j + offset] % prime_mod;
            }
            encoder->encode(tmp_vec, multArr_pt[i]);
            encoder->encode(tmp_vec2, inArr_pt[i]);
        }

        sci::PRG128 prg;
        vector<Plaintext> enc_noise(num_ct);
        // vector<vector<uint64_t>> secret_share(num_ct, vector<uint64_t>(slot_count, 0));
        vector<uint64_t> secret_share(num_ct * slot_count, 0);
        for (int i = 0; i < num_ct; i++) {
            prg.random_mod_p<uint64_t>(secret_share.data() + i * slot_count, slot_count, prime_mod);
            std::vector<uint64_t> tmp(secret_share.data() + i * slot_count,
                                      secret_share.data() + (i + 1) * slot_count);
            encoder->encode(tmp, enc_noise[i]);
        }
        std::memcpy(outputArr, secret_share.data(), sizeof(uint64_t) * size);

        vector<Ciphertext> ct(num_ct);
        IO::recv_encrypted_vector(*io, *context, ct);

        vector<Ciphertext> enc_result(num_ct);
        for (int i = 0; i < num_ct; i++) {
            evaluator->add_plain(ct[i], inArr_pt[i], enc_result[i]);
            evaluator->multiply_plain_inplace(enc_result[i], multArr_pt[i]);
            evaluator->sub_plain_inplace(enc_result[i], enc_noise[i]);

            // evaluator->mod_switch_to_next_inplace(enc_result[i]);

            parms_id_type parms_id = enc_result[i].parms_id();
            std::shared_ptr<const SEALContext::ContextData> context_data
                = context->get_context_data(parms_id);

            // flood_ciphertext(enc_result[i], context_data, SMUDGING_BITLEN);

            // evaluator->mod_switch_to_next_inplace(enc_result[i]);
        }
        IO::send_encrypted_vector(*io, enc_result);
        io->flush();

#ifdef VERIFY
        io->send_data(inArr, size * sizeof(uint64_t));
        io->send_data(multArr, size * sizeof(uint64_t));
        io->send_data(outputArr, size * sizeof(uint64_t));
        io->flush();
#endif
    }
    delete encoder;
    delete evaluator;
}

#endif
