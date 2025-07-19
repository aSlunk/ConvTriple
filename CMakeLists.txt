cmake_minimum_required(VERSION 3.18)
project(IDP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED 20)

option(USE_APPROX_RESHARE "Enable APPROX RESHARE" OFF)
option(TRIP_VERIFY "Enable verification" ON)
option(TRIP_COLOR "Enable colored output" ON)

include(FetchContent)

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
