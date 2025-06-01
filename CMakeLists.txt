cmake_minimum_required(VERSION 4.0.1)
project(IDP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED 20)

include(FetchContent)

FetchContent_Declare(
    seal
    GIT_REPOSITORY https://github.com/microsoft/SEAL.git
    GIT_TAG main
)

FetchContent_MakeAvailable(seal)

add_compile_options(
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra;-Wno-unused-parameter>"
)

set(ALL_SOURCE_FILES "")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(cheetah)

add_custom_target(
    format
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND clang-format
        -i ${ALL_SOURCE_FILES}
        --verbose
)
