//  Authors: Wen-jie Lu on 2021/9/11.
// #include "gemini/cheetah/hom_conv2d_ss.cuh"

#include <seal/seal.h>
#include <seal/secretkey.h>

#include <cuda_runtime.h>
#include <troy/troy_cuda.cuh>
// #include <cuda.h>


#define LOG_CUDA 1


// troy::EvaluatorCuda *evaluator;

namespace gemini {

  struct MetaCu {
    size_t n_filters;
    size_t stride;
    bool is_shared_input;
  };

  seal::scheme_type scheme(const std::shared_ptr<seal::SEALContext> context_){
    if (context_) {
      return context_->first_context_data()->parms().scheme();
    } else {
      return seal::scheme_type::none;
    }
  }

  size_t poly_degree(const std::shared_ptr<seal::SEALContext> context_){
    if (context_) {
      return context_->first_context_data()->parms().poly_modulus_degree();
    } else {
      return 0;
    }
  }

  uint64_t plain_modulus(const std::shared_ptr<seal::SEALContext> context_){
    if (context_) {
      return context_->first_context_data()->parms().plain_modulus().value();
    } else {
      return -1;
    }
  }

/*
  size_t maxThreadsHE = 16;
  static Code LaunchWorks(
      ThreadPool &tpool, size_t num_works,
      std::function<Code(long wid, size_t start, size_t end)> program) {
    if (num_works == 0) return Code::OK;
    const long pool_sze = tpool.pool_size();
    if (pool_sze <= 1L) {
      return program(0, 0, num_works);
    } else {
      Code code;
      std::vector<std::future<Code>> futures;
      size_t work_load = (num_works + pool_sze - 1) / pool_sze;
      for (long wid = 0; wid < pool_sze; ++wid) {
        size_t start = wid * work_load;
        size_t end = std::min(start + work_load, num_works);
        futures.push_back(tpool.enqueue(program, wid, start, end));
      }

      code = Code::OK;
      for (auto &&work : futures) {
        Code c = work.get();
        if (code == Code::OK && c != Code::OK) {
          code = c;
        }
      }
      return code;
    }
  }

  void set_up_eval(const std::shared_ptr<seal::SEALContext> context,
            std::shared_ptr<seal::Evaluator> &evaluator) {
    evaluator = std::make_shared<seal::Evaluator>(*context);
  }

  Code filtersToNtt(
      const std::shared_ptr<seal::SEALContext> context,
      const std::shared_ptr<seal::Evaluator> evaluator,
      std::vector<std::vector<seal::Plaintext>> &encoded_filters,
      size_t nthreads) {

    const size_t M = encoded_filters.size();
    const size_t N = encoded_filters.at(0).size();
    
    auto to_ntt_program = [&](long wid, size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        for (size_t j = 0; j < N; ++j) {
          try {
            evaluator->transform_to_ntt_inplace(
              encoded_filters[i][j], 
              context->first_context_data()->parms_id()
            );
          } catch (std::logic_error e) {
            LOG(WARNING) << "SEAL ERROR: " << e.what();
            return Code::ERR_INTERNAL;
          }
        }
      }
      return Code::OK;
    };

    ThreadPool tpool(std::min(std::max(1UL, nthreads), maxThreadsHE));
    return LaunchWorks(tpool, M, to_ntt_program);
  }

  size_t conv2DOneFilter(
      const std::shared_ptr<seal::SEALContext> context,
      const std::shared_ptr<seal::Evaluator> evaluator,
      const std::vector<seal::Ciphertext> &image,
      const std::vector<seal::Plaintext> &filter,
      const MetaCu &meta,
      seal::Ciphertext *out_buff,
      size_t out_buff_sze,
      bool fill_ntt) {

    if (!evaluator) {
      LOG(WARNING) << "conv2DOneFilter: evaluator is absent";
      return size_t(-1);
    }

    size_t nnz = std::accumulate(filter.cbegin(), filter.cend(), 0,
                                [](size_t nnz, const seal::Plaintext &f) {
                                  return nnz + (f.is_zero() ? 0 : 1);
                                });
    if (nnz == 0) {
      LOG(WARNING) << "conv2DOneFilter: filter with all zero is not supported";
      return size_t(-1);
    }

    const size_t out_size = image.size() / filter.size();
    if (out_size < 1) {
      return size_t(-1);
    }

    if (out_size > out_buff_sze || !out_buff) {
      LOG(WARNING) << "conv2DOneFilter: require a larger out_buff";
      return size_t(-1);
    }

    const size_t accum_cnt = filter.size();
    for (size_t i = 0; i < out_size; ++i) {
      out_buff[i].release();
    }

    for (size_t c = 0; c < accum_cnt; ++c) {
      // filter on the margin might be all-zero
      seal::Plaintext _filter = filter[c]; 
      if (_filter.is_zero()) {
        continue;
      }

      if (fill_ntt) {
        evaluator->transform_to_ntt_inplace(_filter, 
                                  context->first_context_data()->parms_id());
      }

      for (size_t i = 0; i < out_size; ++i) {
        size_t ii = c * out_size + i;
        size_t o = ii % out_size;

        if (out_buff[o].size() > 0) {
          // TODO Use FMA. out_buf[o] += tensor[ii] * filter[c];
          auto cpy_ct{image.at(ii)};
          evaluator->multiply_plain_inplace(cpy_ct, _filter);
          evaluator->add_inplace(out_buff[o], cpy_ct);
        } else {
          evaluator->multiply_plain(image.at(ii), _filter, out_buff[o]);
        }
      }
    }
    return out_size;
  }

  Code conv2DSS(
      const std::shared_ptr<seal::SEALContext> context,
      const std::shared_ptr<seal::Evaluator> evaluator,
      const std::vector<seal::Ciphertext> &img_share0,
      const std::vector<seal::Plaintext> &img_share1,
      const std::vector<std::vector<seal::Plaintext>> &filters, const MetaCu &meta,
      std::vector<seal::Ciphertext> &out_share0,
      size_t n_one_channel, size_t nthreads, bool in_ntt, bool fil_ntt, bool out_ntt) {
        
    if (filters.size() != meta.n_filters) {
      LOG(WARNING) << "conv2DSS: #filters " << filters.size()
                  << " != " << meta.n_filters << "\n";
      return Code::ERR_DIM_MISMATCH;
    }

    if (meta.is_shared_input && img_share0.size() != img_share1.size()) {
      LOG(WARNING) << "conv2DSS: #shares " << img_share0.size()
                  << " != " << img_share1.size() << "\n";
      return Code::ERR_DIM_MISMATCH;
    }

    ENSURE_OR_RETURN(filters.size() == meta.n_filters, Code::ERR_DIM_MISMATCH);
    if (meta.is_shared_input) {
      ENSURE_OR_RETURN(img_share0.size() == img_share1.size(),
                      Code::ERR_DIM_MISMATCH);
    }

    auto tl_pool =
        seal::MemoryManager::GetPool(seal::mm_prof_opt::mm_force_thread_local);
        
    ThreadPool tpool(std::min(std::max(1UL, nthreads), maxThreadsHE));

    std::vector<seal::Ciphertext> image;
    auto add_program = [&](long wid, size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        try {
          evaluator->add_plain(img_share0[i], img_share1[i], image[i]);
        } catch (std::logic_error e) {
          LOG(WARNING) << "SEAL ERROR: " << e.what();
          return Code::ERR_INTERNAL;
        }
      }
      return Code::OK;
    };

    if (meta.is_shared_input) {
      image.resize(img_share0.size(), seal::Ciphertext(tl_pool));
      CHECK_ERR(LaunchWorks(tpool, image.size(), add_program), "add");
    } else {
      image = img_share0;
    }

    auto to_ntt_program = [&](long wid, size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        try {
          evaluator->transform_to_ntt_inplace(image[i]);
        } catch (std::logic_error e) {
          LOG(WARNING) << "SEAL ERROR: " << e.what();
          return Code::ERR_INTERNAL;
        }
      }
      return Code::OK;
    };

    if(in_ntt) {
      CHECK_ERR(LaunchWorks(tpool, image.size(), to_ntt_program), "to_ntt");
    }
    
    const size_t n_out_ct = meta.n_filters * n_one_channel;
    out_share0.resize(n_out_ct);
    auto conv_program = [&](long wid, size_t start, size_t end) {
      for (size_t m = start; m < end; ++m) {
        seal::Ciphertext *ct_start = &out_share0.at(m * n_one_channel);
        size_t used = conv2DOneFilter(context, evaluator, image, filters[m], meta, ct_start, 
                                      n_one_channel, fil_ntt);
        if (used == (size_t)-1 || used != n_one_channel) {
          return Code::ERR_INTERNAL;
        }
      };

      return Code::OK;
    };

    CHECK_ERR(LaunchWorks(tpool, meta.n_filters, conv_program), "conv2D");

    auto from_ntt_program = [&](long wid, size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        try {
          evaluator->transform_from_ntt_inplace(out_share0[i]);
        } catch (std::logic_error e) {
          LOG(WARNING) << "SEAL ERROR: " << e.what();
          return Code::ERR_INTERNAL;
        }
      }
      return Code::OK;
    };

    if(out_ntt) {
      CHECK_ERR(LaunchWorks(tpool, out_share0.size(), from_ntt_program), "from_ntt");
    }

    return Code::OK;
  }
*/

  inline bool isInitializedCuda();

  inline void initializeCuda();
  
  inline void seal2cudaContext(const std::shared_ptr<seal::SEALContext> context,
            troy::SEALContextCuda *&contextCu, 
            std::vector<int> moduli_bits = {60, 49},
            int n_spl_prime = 0) {
#if LOG_CUDA
    std::cout << "seal2cudaContext() called..."
      << "\n"
    ; 
#endif
    // recreate troy encryption parms from seal parms
    const size_t polyn_mod = poly_degree(context);
    const uint64_t plain_mod = plain_modulus(context);
    const std::vector<troy::Modulus> coeff_mod = troy::CoeffModulus::Create(polyn_mod, moduli_bits);
    troy::EncryptionParameters parmsTroy(troy::SchemeType::bfv);
    parmsTroy.setPolyModulusDegree(polyn_mod);
    parmsTroy.setCoeffModulus(coeff_mod);
    parmsTroy.setPlainModulus(plain_mod);
    contextCu = new troy::SEALContextCuda(parmsTroy, true, troy::SecurityLevel::tc128);
#if LOG_CUDA
    std::cout << "seal2cudaContext() done!" 
      << "\n"
      << "\t+ " << "polyn_mod = " << polyn_mod
      << "\n"
      << "\t+ " << "plain_mod = " << plain_mod 
      << "\n"
      << "\t+ " << "moduli_bits = " << moduli_bits[0] << ", " << moduli_bits[1] 
      << "\n"
      << "\t+ " << "n_spl_prime = " << n_spl_prime 
      << "\n"
    ;
#endif
  }

  inline void setUpEvalCu(const std::shared_ptr<seal::SEALContext> context,
            troy::SEALContextCuda *&contextCu,
            troy::EvaluatorCuda *&evaluatorCu) {

#if LOG_CUDA
    std::cout << "setUpEvalCu() called..."
      << "\n"
    ;
#endif

    if(!isInitializedCuda()) initializeCuda();

    // convert seal context to cuda context
    seal2cudaContext(context, contextCu);

    // initialize evaluator
    evaluatorCu = new troy::EvaluatorCuda(*contextCu);

#if LOG_CUDA
    std::cout << "setUpEvalCu() done!"
      << "\n"
    ;
#endif

  }

#if LOG_CUDA
  // start a counter for the total bytes saved
  size_t total_bytes_saved = 0;
#endif

  // function to convert seal plaintext to cuda plaintext
  inline troy::PlaintextCuda seal2cudaPt(const seal::Plaintext &plain) {

    troy::PlaintextCuda pt;

    // save the data from seal plaintext to stream
    std::stringstream ss;
    plain.save_wo_header(ss);

#if LOG_CUDA
    total_bytes_saved += ss.str().size();
#endif

    // load the data from stream to cuda plaintext
    pt.load(ss);

    return pt;
  }

  inline std::vector<troy::PlaintextCuda> seal2cudaPt1D(const std::vector<seal::Plaintext> &plain1d) {

    std::vector<troy::PlaintextCuda> pt1d;
    pt1d.resize(plain1d.size());

    for (size_t i = 0; i < plain1d.size(); ++i) {
        pt1d[i] = seal2cudaPt(plain1d[i]);
    }

    return pt1d;
  }

  inline std::vector<std::vector<troy::PlaintextCuda>> seal2cudaPt2D(const std::vector<std::vector<seal::Plaintext>> &plain2d) {

#if LOG_CUDA
    total_bytes_saved = 0;
    std::cout << "seal2cudaPt2D() called..." 
      << "\n" 
    ;
#endif

    std::vector<std::vector<troy::PlaintextCuda>> pt2d;
    pt2d.resize(plain2d.size());

    for (size_t i = 0; i < plain2d.size(); ++i) {
        pt2d[i] = seal2cudaPt1D(plain2d[i]);
    }

#if LOG_CUDA
    std::cout << "seal2cudaPt2D() done!" 
      << "\n" 
      << "\t+ " << "total_bytes_saved = " << total_bytes_saved
      << "\n"
    ;
#endif

    return pt2d;
  }

  // function to convert cuda plaintext to seal plaintext
  inline seal::Plaintext cuda2sealPt(
      const troy::PlaintextCuda &plain) {

    seal::Plaintext pt;

    // save the data from cuda plaintext to stream
    std::stringstream ss;
    plain.save(ss);

#if LOG_CUDA
    total_bytes_saved += ss.str().size();
#endif

    // load the data from stream to seal plaintext
    pt.load_wo_header(ss);
    
    return pt;
  }

  inline std::vector<seal::Plaintext> 
    cuda2sealPt1D(
      const std::vector<troy::PlaintextCuda> &plain1d) {

    std::vector<seal::Plaintext> pt1d;
    pt1d.resize(plain1d.size());

    for (size_t i = 0; i < plain1d.size(); ++i) {
        pt1d[i] = cuda2sealPt(plain1d[i]);
    }

    return pt1d;
  }

  inline std::vector<std::vector<seal::Plaintext>> 
    cuda2sealPt2D(
      const std::vector<std::vector<troy::PlaintextCuda>> &plain2d) {

#if LOG_CUDA
    total_bytes_saved = 0;
    std::cout << "cuda2sealPt2D() called..." 
      << "\n" 
      << "\t+ " << "sizeof(parms_id_type) = " << sizeof(seal::parms_id_type)
      << "\n"
      << "\t+ " << "sizeof(ParmsID) = " << sizeof(troy::ParmsID)
      << "\n"
    ;
#endif

    std::vector<std::vector<seal::Plaintext>> pt2d;
    pt2d.resize(plain2d.size());

    for (size_t i = 0; i < plain2d.size(); ++i) {
        pt2d[i] = cuda2sealPt1D(plain2d[i]);
    }

#if LOG_CUDA
    std::cout << "cuda2sealPt2D() done!" 
      << "\n" 
      << "\t+ " << "total_bytes_saved = " << total_bytes_saved
      << "\n"
    ;
#endif

    return pt2d;
  }

  // Converts a seal ciphertext to a cuda ciphertext
  inline troy::CiphertextCuda seal2cudaCt(const seal::Ciphertext &ct) {

    troy::CiphertextCuda ctCu;

    // save the data from seal ciphertext to stream
    std::stringstream ss;
    ct.save_wo_header(ss);

#if LOG_CUDA
    total_bytes_saved += ss.str().size();
#endif

    // load the data from stream to cuda ciphertext
    ctCu.load(ss);

    return ctCu;
  }

  inline std::vector<troy::CiphertextCuda> seal2cudaCt1D(const std::vector<seal::Ciphertext> &ct1d) {

    std::vector<troy::CiphertextCuda> ct1dCu;
    ct1dCu.resize(ct1d.size());

    for (size_t i = 0; i < ct1d.size(); ++i) {
        ct1dCu[i] = seal2cudaCt(ct1d[i]);
    }

    return ct1dCu;
  }

  inline std::vector<std::vector<troy::CiphertextCuda>> seal2cudaCt2D(const std::vector<std::vector<seal::Ciphertext>> &ct2d) {

#if LOG_CUDA
    total_bytes_saved = 0;
    std::cout << "seal2cudaCt2D() called..." 
      << "\n" 
    ;
#endif

    std::vector<std::vector<troy::CiphertextCuda>> ct2dCu;
    ct2dCu.resize(ct2d.size());

    for (size_t i = 0; i < ct2d.size(); ++i) {
        ct2dCu[i] = seal2cudaCt1D(ct2d[i]);
    }

#if LOG_CUDA
    std::cout << "seal2cudaCt2D() done!" 
      << "\n" 
      << "\t+ " << "total_bytes_saved = " << total_bytes_saved
      << "\n"
    ;
#endif

    return ct2dCu;
  }

  // Converts a cuda ciphertext to a seal ciphertext
  inline seal::Ciphertext cuda2sealCt(const troy::CiphertextCuda &ct) {

    seal::Ciphertext ctSeal;

    // save the data from cuda ciphertext to stream
    std::stringstream ss;
    ct.save(ss);

#if LOG_CUDA
    total_bytes_saved += ss.str().size();
#endif

    // load the data from stream to seal ciphertext
    ctSeal.load_wo_header(ss);

    return ctSeal;
  }

  inline std::vector<seal::Ciphertext> cuda2sealCt1D(const std::vector<troy::CiphertextCuda> &ct1d) {

    std::vector<seal::Ciphertext> ct1dSeal;
    ct1dSeal.resize(ct1d.size());

    for (size_t i = 0; i < ct1d.size(); ++i) {
        ct1dSeal[i] = cuda2sealCt(ct1d[i]);
    }

    return ct1dSeal;
  }

  inline std::vector<std::vector<seal::Ciphertext>> cuda2sealCt2D(const std::vector<std::vector<troy::CiphertextCuda>> &ct2d) {

#if LOG_CUDA
    total_bytes_saved = 0;
    std::cout << "cuda2sealCt2D() called..." 
      << "\n" 
    ;
#endif

    std::vector<std::vector<seal::Ciphertext>> ct2dSeal;
    ct2dSeal.resize(ct2d.size());

    for (size_t i = 0; i < ct2d.size(); ++i) {
        ct2dSeal[i] = cuda2sealCt1D(ct2d[i]);
    }

#if LOG_CUDA
    std::cout << "cuda2sealCt2D() done!" 
      << "\n" 
      << "\t+ " << "total_bytes_saved = " << total_bytes_saved
      << "\n"
    ;
#endif
  
    return ct2dSeal;
  } 


  void filToNttCu(
    const troy::SEALContextCuda *contextCu,
    const troy::EvaluatorCuda *evaluatorCu,
    std::vector<std::vector<troy::PlaintextCuda>> &filPtCu) {

    const size_t M = filPtCu.size();
    const size_t N = filPtCu.at(0).size();
    
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        evaluatorCu->transformToNttInplace(
          filPtCu[i][j], 
          contextCu->firstContextData()->parmsID()
        );
      }
    }
  }

  void filToNttCuSealWrap(
    const std::shared_ptr<seal::SEALContext> context,
    std::vector<std::vector<seal::Plaintext>> &encoded_filters) {
      
#if LOG_CUDA
    std::cout << "filToNttCuSealWrap() called..." 
    << "\n"
    ;
#endif

    troy::SEALContextCuda *contextCu;
    troy::EvaluatorCuda *evaluatorCu;

#if LOG_CUDA
    std::cout 
      << "\t+ " << "contextCu and evaluatorCu initialized"
      << "\n"
      ;
#endif

    setUpEvalCu(context, contextCu, evaluatorCu);

    // convert seal plaintext to cuda plaintext
    std::vector<std::vector<troy::PlaintextCuda>> filPtCu = seal2cudaPt2D(encoded_filters);

    filToNttCu(contextCu, evaluatorCu, filPtCu);

    encoded_filters = cuda2sealPt2D(filPtCu);
  }

  void conv2DOneFilter(
      const troy::SEALContextCuda *contextCu,
      const troy::EvaluatorCuda *evaluatorCu,
      const std::vector<troy::CiphertextCuda> &image,
      const std::vector<troy::PlaintextCuda> &filter,
      troy::CiphertextCuda *out_buff,
      size_t out_buff_sze,
      bool fill_ntt) {

    const size_t out_size = image.size() / filter.size();

    const size_t accum_cnt = filter.size();
    for (size_t i = 0; i < out_size; ++i) {
      out_buff[i].release();
    }

    for (size_t c = 0; c < accum_cnt; ++c) {

      troy::PlaintextCuda _filter = filter[c]; 
      if (_filter.isZero()) {
        continue;
      }

      if (fill_ntt) {
        evaluatorCu->transformToNttInplace(_filter, contextCu->firstContextData()->parmsID());
      }

      for (size_t i = 0; i < out_size; ++i) {
        size_t ii = c * out_size + i;
        size_t o = ii % out_size;

        if (out_buff[o].size() > 0) {
          auto cpy_ct{image.at(ii)};
          evaluatorCu->multiplyPlainInplace(cpy_ct, _filter);
          evaluatorCu->addInplace(out_buff[o], cpy_ct);
        } else {
          evaluatorCu->multiplyPlain(image.at(ii), _filter, out_buff[o]);
        }
      }
    }
  }

  void conv2DCu(
      const troy::SEALContextCuda *contextCu,
      const troy::EvaluatorCuda *evaluatorCu,
      const std::vector<troy::CiphertextCuda> &imgShare0,
      const std::vector<troy::PlaintextCuda> &imgShare1,
      const std::vector<std::vector<troy::PlaintextCuda>> &filPtCu,
      const MetaCu &meta, 
      std::vector<troy::CiphertextCuda> &outShare0, 
      size_t n_one_channel, bool in_ntt, bool fil_ntt, bool out_ntt) {

    std::vector<troy::CiphertextCuda> imageCu;
    if (meta.is_shared_input) {
      imageCu.resize(imgShare0.size());
      for (size_t i = 0; i < imgShare0.size(); ++i) {
        evaluatorCu->addPlain(imgShare0[i], imgShare1[i], imageCu[i]);
      }
    } else {
      imageCu = imgShare0;
    }

    if (in_ntt) {
      for (size_t i = 0; i < imageCu.size(); ++i) {
        evaluatorCu->transformToNttInplace(imageCu[i]);
      }
    }

    const size_t n_out_ct = meta.n_filters * n_one_channel;
    outShare0.resize(n_out_ct);
    for (size_t m = 0; m < meta.n_filters; ++m) {
      troy::CiphertextCuda *ct_start = &outShare0.at(m * n_one_channel);
      conv2DOneFilter(contextCu, evaluatorCu, imageCu, filPtCu[m], ct_start, n_one_channel, fil_ntt);
    }

    if (out_ntt) {
      for (size_t i = 0; i < outShare0.size(); ++i) {
        evaluatorCu->transformFromNttInplace(outShare0[i]);
      }
    }
  }
}  // namespace gemini