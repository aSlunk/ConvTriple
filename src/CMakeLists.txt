## cheetah-core
add_library(gemini SHARED)
add_subdirectory(include)
target_include_directories(gemini PUBLIC include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gemini PUBLIC SEAL::seal Eigen3::Eigen)

target_compile_options(
    gemini PRIVATE
    "$<$<CONFIG:Debug>:-g3;-ggdb;-Og;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-Wall;-Wextra>"
    "-Wno-unused-parameter"
)


if (USE_APPROX_RESHARE)
    target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=1)
else ()
    target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=0)
endif ()

if(TRIPLE_ZERO)
    target_compile_definitions(gemini PRIVATE ZERO)
endif()

target_include_directories(gemini PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${EMP-OT_INCLUDE_DIRS}
    ${EMP-TOOL_INCLUDE_DIRS}
)

################################################################################
## Interface: static library HE
################################################################################
set(HE_LIB HE)
if(TRIPLE_GPU)
    add_library(${HE_LIB} STATIC
        core/hpmpc_interface.cpp
        core/utils.cpp
        io/send.cpp
        troy/conv2d_gpu.cu
    )

    target_link_libraries(${HE_LIB} PUBLIC Troy::troy CUDA::cudart CUDA::cufft)
    target_compile_definitions(${HE_LIB} PRIVATE USE_CONV_CUDA)
else()
    add_library(${HE_LIB} STATIC
        core/hpmpc_interface.cpp
        core/utils.cpp
        io/send.cpp
    )
endif()
message(STATUS ${EIGEN3_INCLUDE_DIR})
target_include_directories(${HE_LIB} PUBLIC io include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${HE_LIB} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(${HE_LIB} PUBLIC gemini)
target_compile_definitions(${HE_LIB} PUBLIC USE_CHEETAH=1)
target_compile_options(
    ${HE_LIB} PRIVATE
    "$<$<CONFIG:Debug>:-g3;-ggdb;-Og;-Wall;-Wextra;-fsanitize=address>"
    "$<$<CONFIG:Release>:-Wall;-Wextra>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers;-Wno-unused-parameter"
)
target_link_options(${HE_LIB} PUBLIC
    "$<$<CONFIG:Debug>:-fsanitize=address>"
)

if(TRIPLE_VERIFY)
    target_compile_definitions(${HE_LIB} PRIVATE VERIFY)
endif()

if(TRIPLE_COLOR)
    target_compile_definitions(${HE_LIB} PRIVATE COLOR)
endif()

################################################################################
## Server / Client executables
################################################################################
set(server cheetah)
add_executable(${server} server.cpp)
target_link_libraries(${server} PRIVATE ${HE_LIB})
target_compile_options(
    ${server} PRIVATE
    "$<$<CONFIG:Debug>:-g3;-ggdb;-Og;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-Wall;-Wextra>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers;-Wno-unused-parameter"
)
if(TRIPLE_COLOR)
    target_compile_definitions(${server} PRIVATE COLOR)
endif()
if(TRIPLE_GPU)
    target_compile_definitions(${server} PRIVATE USE_CONV_CUDA)
endif()

set(client cheetah_client)
add_executable(${client} client.cpp)
target_link_libraries(${client} PRIVATE ${HE_LIB})
target_compile_definitions(${client} PUBLIC USE_CHEETAH=1)
target_compile_options(
    ${client} PRIVATE
    "$<$<CONFIG:Debug>:-g3;-ggdb;-Og;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-Wall;-Wextra>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers;-Wno-unused-parameter"
)
if(TRIPLE_COLOR)
    target_compile_definitions(${client} PRIVATE COLOR)
endif()
if(TRIPLE_GPU)
    target_compile_definitions(${client} PRIVATE USE_CONV_CUDA)
endif()

message(STATUS "TRIPLE_VERIFY: ${TRIPLE_VERIFY}")
message(STATUS "TRIPLE_COLOR: ${TRIPLE_COLOR}")

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/troy/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/troy/*.cuh"
)
set(ALL_SOURCE_FILES ${ALL_SOURCE_FILES} ${SRC_FILES} PARENT_SCOPE)
