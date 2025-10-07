cmake_minimum_required(VERSION 3.18)
project(IDP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED 20)

option(USE_APPROX_RESHARE "Enable APPROX RESHARE" OFF)
option(TRIPLE_VERIFY "Enable verification" ON)
option(TRIPLE_COLOR "Enable colored output" ON)
option(TRIPLE_ZERO "Enable computation with ZERO" OFF)

add_compile_options(-fPIC)
# add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

## Dependencies
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/deps;${CMAKE_PREFIX_PATH}")

find_package(Eigen3 NO_MODULE)
find_package(emp-tool REQUIRED)
find_package(emp-ot REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(SEAL 4.0.0 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIR}
#                     ${EMP-OT_INCLUDE_DIRS}
#                     ${EMP-TOOL_INCLUDE_DIRS})


set(ALL_SOURCE_FILES "")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)

add_custom_target(
    format
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND clang-format
        -i ${ALL_SOURCE_FILES}
        --verbose
)
