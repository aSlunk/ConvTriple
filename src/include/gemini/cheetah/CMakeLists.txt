if(TRIPLE_GPU)
    target_sources(gemini PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/shape_inference.cc
        ${CMAKE_CURRENT_LIST_DIR}/tensor_shape.cc
        ${CMAKE_CURRENT_LIST_DIR}/tensor_encoder.cc
        ${CMAKE_CURRENT_LIST_DIR}/hom_conv2d_ss.cc
        ${CMAKE_CURRENT_LIST_DIR}/hom_conv2d_core.cu
        ${CMAKE_CURRENT_LIST_DIR}/hom_fc_ss.cc
        ${CMAKE_CURRENT_LIST_DIR}/hom_bn_ss.cc
    )
else()
    target_sources(gemini PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/shape_inference.cc
        ${CMAKE_CURRENT_LIST_DIR}/tensor_shape.cc
        ${CMAKE_CURRENT_LIST_DIR}/tensor_encoder.cc
        ${CMAKE_CURRENT_LIST_DIR}/hom_conv2d_ss.cc
        ${CMAKE_CURRENT_LIST_DIR}/hom_fc_ss.cc
        ${CMAKE_CURRENT_LIST_DIR}/hom_bn_ss.cc
    )
endif()