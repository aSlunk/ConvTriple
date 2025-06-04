//  Authors: Wen-jie Lu on 2021/9/11.
#ifndef GEMINI_CHEETAH_HOMCONVSS_H_
#define GEMINI_CHEETAH_HOMCONVSS_H_
#include <seal/secretkey.h>
#include <seal/serializable.h>

#include <optional>
#include <vector>

#include "gemini/cheetah/tensor.h"
#include "gemini/cheetah/tensor_shape.h"

// Forward
namespace seal {
class SEALContext;
class PublicKey;

class Plaintext;
class Ciphertext;
class Evaluator;
} // namespace seal

namespace gemini {

class TensorEncoder;

class HomConv2DSS {
  public:
#ifdef HOM_CONV2D_SS_MAX_THREADS
    static constexpr size_t kMaxThreads = HOM_CONV2D_SS_MAX_THREADS;
#else
    static constexpr size_t kMaxThreads = 16ULL;
#endif

    struct Meta {
        TensorShape ishape;
        TensorShape fshape;
        size_t n_filters;
        Padding padding;
        size_t stride;
        bool is_shared_input;
    };

    explicit HomConv2DSS() = default;

    ~HomConv2DSS() = default;

    Code setUp(const seal::SEALContext& context, std::optional<seal::SecretKey> sk = std::nullopt,
               std::shared_ptr<seal::PublicKey> pk = nullptr);

    [[nodiscard]] seal::scheme_type scheme() const;

    [[nodiscard]] size_t poly_degree() const;

    uint64_t plain_modulus() const;

    Code encryptImage(const Tensor<uint64_t>& in_tensor_share, const Meta& meta,
                      std::vector<seal::Ciphertext>& encrypted_share, size_t nthreads = 1) const;

    Code encryptImage(const Tensor<uint64_t>& in_tensor_share, const Meta& meta,
                      std::vector<seal::Ciphertext>& encrypted_share,
                      std::vector<seal::Plaintext>& encoded_share,
                      size_t nthreads = 1) const; // added

    Code encodeImage(const Tensor<uint64_t>& in_tensor_share, const Meta& meta,
                     std::vector<seal::Plaintext>& encoded_share, size_t nthreads = 1) const;

    Code encryptFilters(const std::vector<Tensor<uint64_t>>& filters, const Meta& meta,
                        std::vector<std::vector<seal::Ciphertext>>& encoded_filters,
                        size_t nthreads = 1) const; // ADDED

    Code encodeFilters(const std::vector<Tensor<uint64_t>>& filters, const Meta& meta,
                       std::vector<std::vector<seal::Plaintext>>& encoded_filters,
                       size_t nthreads = 1) const;

    Code conv2DSS(const std::vector<seal::Ciphertext>& img_share0,
                  const std::vector<seal::Plaintext>& img_share1,
                  const std::vector<std::vector<seal::Plaintext>>& filters, const Meta& meta,
                  std::vector<seal::Ciphertext>& out_share0, Tensor<uint64_t>& out_share1,
                  size_t nthreads = 1) const;

    Code conv2DSS(const std::vector<seal::Ciphertext>& img_share0,
                  const std::vector<std::vector<seal::Plaintext>>& filters, const Meta& meta,
                  const Tensor<uint64_t>& R, std::vector<seal::Ciphertext>& out_share0,
                  size_t nthreads = 1) const; // MODIFIED CONV2DSS

    Code conv2DSS(const std::vector<seal::Plaintext>& img_share0,
                  const std::vector<std::vector<seal::Ciphertext>>& filters, const Meta& meta,
                  std::vector<seal::Ciphertext>& out_share0, size_t nthreads = 1) const; // ADDED

    Code add_plain_inplace(std::vector<seal::Ciphertext>& ciphers,
                           const std::vector<seal::Plaintext>& plain,
                           const size_t& nthreads = 1ULL) const; // ADDED

    Code add_inplace(std::vector<seal::Ciphertext>& A, const std::vector<seal::Ciphertext>& B,
                     const size_t& nthreads = 1ULL) const; // ADDED

    Code decryptToTensor(const std::vector<seal::Ciphertext>& enc_tensor, const Meta& meta,
                         Tensor<uint64_t>& out, size_t nthreads = 1) const;

    Code idealFunctionality(const Tensor<uint64_t>& in_tensor,
                            const std::vector<Tensor<uint64_t>>& filters, const Meta& meta,
                            Tensor<uint64_t>& out_tensor) const;

    static TensorShape GetConv2DOutShape(const HomConv2DSS::Meta& meta);

  protected:
    size_t conv2DOneFilter(const std::vector<seal::Ciphertext>& enc_tensor,
                           const std::vector<seal::Plaintext>& filter, const Meta& meta,
                           seal::Ciphertext* out_buff, size_t out_buff_sze) const;

    size_t conv2DOneFilter(const std::vector<seal::Plaintext>& enc_tensor,
                           const std::vector<seal::Ciphertext>& filter, const Meta& meta,
                           seal::Ciphertext* out_buff, size_t out_buff_sze) const; // ADDED

    Code sampleRandomMask(const std::vector<size_t>& targets, uint64_t* coeffs_buff,
                          size_t buff_size, seal::Plaintext& mask, seal::parms_id_type pid,
                          std::shared_ptr<seal::UniformRandomGenerator> prng, bool is_ntt) const;

    Code addRandomMask(std::vector<seal::Ciphertext>& enc_tensor, Tensor<uint64_t>& mask_tensor,
                       const Meta& meta, size_t nthreads = 1) const;

    Code removeUnusedCoeffs(std::vector<seal::Ciphertext>& ct, const Meta& meta,
                            double* density = nullptr) const;

    Code postProcessInplace(seal::Plaintext& pt, std::vector<size_t>& targets, uint64_t* out_poly,
                            size_t out_buff_size) const;

    std::shared_ptr<seal::SEALContext> context_;
    std::shared_ptr<TensorEncoder> tencoder_{nullptr};
    std::shared_ptr<seal::Evaluator> evaluator_{nullptr};
    std::shared_ptr<seal::Encryptor> encryptor_{nullptr};
    std::shared_ptr<seal::PublicKey> pk_{nullptr};

    std::optional<seal::SecretKey> sk_{std::nullopt};
};

}; // namespace gemini
#endif // GEMINI_CHEETAH_HOMCONVSS_H_
