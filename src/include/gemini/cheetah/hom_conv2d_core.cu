//  Authors: Wen-jie Lu on 2021/9/11.
#include "gemini/cheetah/hom_conv2d_core.cuh"

// troy::EvaluatorCuda *evaluator;

namespace gemini {

  inline bool isInitializedCuda() {
    return troy::KernelProvider::isInitialized();
  }

  inline void initializeCuda() {
#if LOG_CUDA
    std::cout << "initializeCuda() called..."
      << "\n" 
      << "\t+ " << "isInitializedCuda() = " 
      << std::boolalpha << isInitializedCuda()
      << "\n"
    ;
#endif

    if(!isInitializedCuda()) troy::KernelProvider::initialize();
    else std::cout << "initializeCuda(): troy::KernelProvider already initialized!" << std::endl;
    
#if LOG_CUDA
    std::cout << "initializeCuda() done!"
      << "\n" 
      << "\t+ " << "isInitializedCuda() = "
      << std::boolalpha << isInitializedCuda()
      << "\n"
    ;
#endif
  }

//   inline void seal2cudaContext(const std::shared_ptr<seal::SEALContext> context,
//             troy::SEALContextCuda *&contextCu, 
//             std::vector<int> moduli_bits = {60, 49},
//             int n_spl_prime = 0) {
// #if LOG_CUDA
//     std::cout << "seal2cudaContext() called..."
//       << "\n"
//     ; 
// #endif
//     // recreate troy encryption parms from seal parms
//     const size_t polyn_mod = poly_degree(context);
//     const uint64_t plain_mod = plain_modulus(context);
//     const std::vector<troy::Modulus> coeff_mod = troy::CoeffModulus::Create(polyn_mod, moduli_bits);
//     troy::EncryptionParameters parmsTroy(troy::SchemeType::bfv);
//     parmsTroy.setPolyModulusDegree(polyn_mod);
//     parmsTroy.setCoeffModulus(coeff_mod);
//     parmsTroy.setPlainModulus(plain_mod);
//     contextCu = new troy::SEALContextCuda(parmsTroy, true, troy::SecurityLevel::tc128);
// #if LOG_CUDA
//     std::cout << "seal2cudaContext() done!" 
//       << "\n"
//       << "\t+ " << "polyn_mod = " << polyn_mod
//       << "\n"
//       << "\t+ " << "plain_mod = " << plain_mod 
//       << "\n"
//       << "\t+ " << "moduli_bits = " << moduli_bits[0] << ", " << moduli_bits[1] 
//       << "\n"
//       << "\t+ " << "n_spl_prime = " << n_spl_prime 
//       << "\n"
//     ;
// #endif
//   }

//   inline void setUpEvalCu(const std::shared_ptr<seal::SEALContext> context,
//             troy::SEALContextCuda *&contextCu,
//             troy::EvaluatorCuda *&evaluatorCu) {

// #if LOG_CUDA
//     std::cout << "setUpEvalCu() called..."
//       << "\n"
//     ;
// #endif

//     if(!isInitializedCuda()) initializeCuda();

//     // convert seal context to cuda context
//     seal2cudaContext(context, contextCu);

//     // initialize evaluator
//     evaluatorCu = new troy::EvaluatorCuda(*contextCu);

// #if LOG_CUDA
//     std::cout << "setUpEvalCu() done!"
//       << "\n"
//     ;
// #endif

//   }

// #if LOG_CUDA
//   // start a counter for the total bytes saved
//   size_t total_bytes_saved = 0;
// #endif

//   // function to convert seal plaintext to cuda plaintext
//   inline troy::PlaintextCuda seal2cudaPt(const seal::Plaintext &plain) {

//     troy::PlaintextCuda pt;

//     // save the data from seal plaintext to stream
//     std::stringstream ss;
//     plain.save_wo_header(ss);

// #if LOG_CUDA
//     total_bytes_saved += ss.str().size();
// #endif

//     // load the data from stream to cuda plaintext
//     pt.load(ss);

//     return pt;
//   }

//   inline std::vector<troy::PlaintextCuda> seal2cudaPt1D(const std::vector<seal::Plaintext> &plain1d) {

//     std::vector<troy::PlaintextCuda> pt1d;
//     pt1d.resize(plain1d.size());

//     for (size_t i = 0; i < plain1d.size(); ++i) {
//         pt1d[i] = seal2cudaPt(plain1d[i]);
//     }

//     return pt1d;
//   }

//   inline std::vector<std::vector<troy::PlaintextCuda>> seal2cudaPt2D(const std::vector<std::vector<seal::Plaintext>> &plain2d) {

// #if LOG_CUDA
//     total_bytes_saved = 0;
//     std::cout << "seal2cudaPt2D() called..." 
//       << "\n" 
//     ;
// #endif

//     std::vector<std::vector<troy::PlaintextCuda>> pt2d;
//     pt2d.resize(plain2d.size());

//     for (size_t i = 0; i < plain2d.size(); ++i) {
//         pt2d[i] = seal2cudaPt1D(plain2d[i]);
//     }

// #if LOG_CUDA
//     std::cout << "seal2cudaPt2D() done!" 
//       << "\n" 
//       << "\t+ " << "total_bytes_saved = " << total_bytes_saved
//       << "\n"
//     ;
// #endif

//     return pt2d;
//   }

//   // function to convert cuda plaintext to seal plaintext
//   inline seal::Plaintext cuda2sealPt(
//       const troy::PlaintextCuda &plain) {

//     seal::Plaintext pt;

//     // save the data from cuda plaintext to stream
//     std::stringstream ss;
//     plain.save(ss);

// #if LOG_CUDA
//     total_bytes_saved += ss.str().size();
// #endif

//     // load the data from stream to seal plaintext
//     pt.load_wo_header(ss);
    
//     return pt;
//   }

//   inline std::vector<seal::Plaintext> 
//     cuda2sealPt1D(
//       const std::vector<troy::PlaintextCuda> &plain1d) {

//     std::vector<seal::Plaintext> pt1d;
//     pt1d.resize(plain1d.size());

//     for (size_t i = 0; i < plain1d.size(); ++i) {
//         pt1d[i] = cuda2sealPt(plain1d[i]);
//     }

//     return pt1d;
//   }

//   inline std::vector<std::vector<seal::Plaintext>> 
//     cuda2sealPt2D(
//       const std::vector<std::vector<troy::PlaintextCuda>> &plain2d) {

// #if LOG_CUDA
//     total_bytes_saved = 0;
//     std::cout << "cuda2sealPt2D() called..." 
//       << "\n" 
//       << "\t+ " << "sizeof(parms_id_type) = " << sizeof(seal::parms_id_type)
//       << "\n"
//       << "\t+ " << "sizeof(ParmsID) = " << sizeof(troy::ParmsID)
//       << "\n"
//     ;
// #endif

//     std::vector<std::vector<seal::Plaintext>> pt2d;
//     pt2d.resize(plain2d.size());

//     for (size_t i = 0; i < plain2d.size(); ++i) {
//         pt2d[i] = cuda2sealPt1D(plain2d[i]);
//     }

// #if LOG_CUDA
//     std::cout << "cuda2sealPt2D() done!" 
//       << "\n" 
//       << "\t+ " << "total_bytes_saved = " << total_bytes_saved
//       << "\n"
//     ;
// #endif

//     return pt2d;
//   }

//   // Converts a seal ciphertext to a cuda ciphertext
//   inline troy::CiphertextCuda seal2cudaCt(const seal::Ciphertext &ct) {

//     troy::CiphertextCuda ctCu;

//     // save the data from seal ciphertext to stream
//     std::stringstream ss;
//     ct.save_wo_header(ss);

// #if LOG_CUDA
//     total_bytes_saved += ss.str().size();
// #endif

//     // load the data from stream to cuda ciphertext
//     ctCu.load(ss);

//     return ctCu;
//   }

//   inline std::vector<troy::CiphertextCuda> seal2cudaCt1D(const std::vector<seal::Ciphertext> &ct1d) {

//     std::vector<troy::CiphertextCuda> ct1dCu;
//     ct1dCu.resize(ct1d.size());

//     for (size_t i = 0; i < ct1d.size(); ++i) {
//         ct1dCu[i] = seal2cudaCt(ct1d[i]);
//     }

//     return ct1dCu;
//   }

//   inline std::vector<std::vector<troy::CiphertextCuda>> seal2cudaCt2D(const std::vector<std::vector<seal::Ciphertext>> &ct2d) {

// #if LOG_CUDA
//     total_bytes_saved = 0;
//     std::cout << "seal2cudaCt2D() called..." 
//       << "\n" 
//     ;
// #endif

//     std::vector<std::vector<troy::CiphertextCuda>> ct2dCu;
//     ct2dCu.resize(ct2d.size());

//     for (size_t i = 0; i < ct2d.size(); ++i) {
//         ct2dCu[i] = seal2cudaCt1D(ct2d[i]);
//     }

// #if LOG_CUDA
//     std::cout << "seal2cudaCt2D() done!" 
//       << "\n" 
//       << "\t+ " << "total_bytes_saved = " << total_bytes_saved
//       << "\n"
//     ;
// #endif

//     return ct2dCu;
//   }

//   // Converts a cuda ciphertext to a seal ciphertext
//   inline seal::Ciphertext cuda2sealCt(const troy::CiphertextCuda &ct) {

//     seal::Ciphertext ctSeal;

//     // save the data from cuda ciphertext to stream
//     std::stringstream ss;
//     ct.save(ss);

// #if LOG_CUDA
//     total_bytes_saved += ss.str().size();
// #endif

//     // load the data from stream to seal ciphertext
//     ctSeal.load_wo_header(ss);

//     return ctSeal;
//   }

//   inline std::vector<seal::Ciphertext> cuda2sealCt1D(const std::vector<troy::CiphertextCuda> &ct1d) {

//     std::vector<seal::Ciphertext> ct1dSeal;
//     ct1dSeal.resize(ct1d.size());

//     for (size_t i = 0; i < ct1d.size(); ++i) {
//         ct1dSeal[i] = cuda2sealCt(ct1d[i]);
//     }

//     return ct1dSeal;
//   }

//   inline std::vector<std::vector<seal::Ciphertext>> cuda2sealCt2D(const std::vector<std::vector<troy::CiphertextCuda>> &ct2d) {

// #if LOG_CUDA
//     total_bytes_saved = 0;
//     std::cout << "cuda2sealCt2D() called..." 
//       << "\n" 
//     ;
// #endif

//     std::vector<std::vector<seal::Ciphertext>> ct2dSeal;
//     ct2dSeal.resize(ct2d.size());

//     for (size_t i = 0; i < ct2d.size(); ++i) {
//         ct2dSeal[i] = cuda2sealCt1D(ct2d[i]);
//     }

// #if LOG_CUDA
//     std::cout << "cuda2sealCt2D() done!" 
//       << "\n" 
//       << "\t+ " << "total_bytes_saved = " << total_bytes_saved
//       << "\n"
//     ;
// #endif
  
//     return ct2dSeal;
//   } 


//   void filToNttCu(
//     const troy::SEALContextCuda *contextCu,
//     const troy::EvaluatorCuda *evaluatorCu,
//     std::vector<std::vector<troy::PlaintextCuda>> &filPtCu) {

//     const size_t M = filPtCu.size();
//     const size_t N = filPtCu.at(0).size();
    
//     for (size_t i = 0; i < M; ++i) {
//       for (size_t j = 0; j < N; ++j) {
//         evaluatorCu->transformToNttInplace(
//           filPtCu[i][j], 
//           contextCu->firstContextData()->parmsID()
//         );
//       }
//     }
//   }

//   void filToNttCuSealWrap(
//     const std::shared_ptr<seal::SEALContext> context,
//     std::vector<std::vector<seal::Plaintext>> &encoded_filters) {
      
// #if LOG_CUDA
//     std::cout << "filToNttCuSealWrap() called..." 
//     << "\n"
//     ;
// #endif

//     troy::SEALContextCuda *contextCu;
//     troy::EvaluatorCuda *evaluatorCu;

// #if LOG_CUDA
//     std::cout 
//       << "\t+ " << "contextCu and evaluatorCu initialized"
//       << "\n"
//       ;
// #endif

//     setUpEvalCu(context, contextCu, evaluatorCu);

//     // convert seal plaintext to cuda plaintext
//     std::vector<std::vector<troy::PlaintextCuda>> filPtCu = seal2cudaPt2D(encoded_filters);

//     filToNttCu(contextCu, evaluatorCu, filPtCu);

//     encoded_filters = cuda2sealPt2D(filPtCu);
//   }

//   void conv2DOneFilter(
//       const troy::SEALContextCuda *contextCu,
//       const troy::EvaluatorCuda *evaluatorCu,
//       const std::vector<troy::CiphertextCuda> &image,
//       const std::vector<troy::PlaintextCuda> &filter,
//       troy::CiphertextCuda *out_buff,
//       size_t out_buff_sze,
//       bool fill_ntt) {

//     const size_t out_size = image.size() / filter.size();

//     const size_t accum_cnt = filter.size();
//     for (size_t i = 0; i < out_size; ++i) {
//       out_buff[i].release();
//     }

//     for (size_t c = 0; c < accum_cnt; ++c) {

//       troy::PlaintextCuda _filter = filter[c]; 
//       if (_filter.isZero()) {
//         continue;
//       }

//       if (fill_ntt) {
//         evaluatorCu->transformToNttInplace(_filter, contextCu->firstContextData()->parmsID());
//       }

//       for (size_t i = 0; i < out_size; ++i) {
//         size_t ii = c * out_size + i;
//         size_t o = ii % out_size;

//         if (out_buff[o].size() > 0) {
//           auto cpy_ct{image.at(ii)};
//           evaluatorCu->multiplyPlainInplace(cpy_ct, _filter);
//           evaluatorCu->addInplace(out_buff[o], cpy_ct);
//         } else {
//           evaluatorCu->multiplyPlain(image.at(ii), _filter, out_buff[o]);
//         }
//       }
//     }
//   }

//   void conv2DCu(
//       const troy::SEALContextCuda *contextCu,
//       const troy::EvaluatorCuda *evaluatorCu,
//       const std::vector<troy::CiphertextCuda> &imgShare0,
//       const std::vector<troy::PlaintextCuda> &imgShare1,
//       const std::vector<std::vector<troy::PlaintextCuda>> &filPtCu,
//       const MetaCu &meta, 
//       std::vector<troy::CiphertextCuda> &outShare0, 
//       size_t n_one_channel, bool in_ntt, bool fil_ntt, bool out_ntt) {

//     std::vector<troy::CiphertextCuda> imageCu;
//     if (meta.is_shared_input) {
//       imageCu.resize(imgShare0.size());
//       for (size_t i = 0; i < imgShare0.size(); ++i) {
//         evaluatorCu->addPlain(imgShare0[i], imgShare1[i], imageCu[i]);
//       }
//     } else {
//       imageCu = imgShare0;
//     }

//     if (in_ntt) {
//       for (size_t i = 0; i < imageCu.size(); ++i) {
//         evaluatorCu->transformToNttInplace(imageCu[i]);
//       }
//     }

//     const size_t n_out_ct = meta.n_filters * n_one_channel;
//     outShare0.resize(n_out_ct);
//     for (size_t m = 0; m < meta.n_filters; ++m) {
//       troy::CiphertextCuda *ct_start = &outShare0.at(m * n_one_channel);
//       conv2DOneFilter(contextCu, evaluatorCu, imageCu, filPtCu[m], ct_start, n_one_channel, fil_ntt);
//     }

//     if (out_ntt) {
//       for (size_t i = 0; i < outShare0.size(); ++i) {
//         evaluatorCu->transformFromNttInplace(outShare0[i]);
//       }
//     }
//   }
}  // namespace gemini