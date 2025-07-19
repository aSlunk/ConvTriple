include(CMakeFindDependencyMacro)
include(FetchContent)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/deps;${CMAKE_PREFIX_PATH}")

add_compile_options(-fPIC)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

## Dependencies
find_package(Threads REQUIRED)
find_package(Eigen3 3.3.3 REQUIRED)
find_package(emp-tool REQUIRED)
find_package(emp-ot REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(SEAL REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR}
                    ${EMP-OT_INCLUDE_DIRS}
                    ${EMP-TOOL_INCLUDE_DIRS})

## cheetah-core
add_library(gemini SHARED)
target_include_directories(gemini PUBLIC include)
target_include_directories(gemini PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(include)
target_link_libraries(gemini PUBLIC SEAL::seal Eigen3::Eigen Threads::Threads)

target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=0)

add_library(OT INTERFACE)
target_include_directories(OT INTERFACE ot)
target_link_libraries(OT INTERFACE OpenSSL::SSL OpenSSL::Crypto)

add_library(IO INTERFACE)
target_include_directories(IO INTERFACE io)
target_link_libraries(IO INTERFACE SEAL::seal gemini OT)

set(server cheetah)
add_executable(${server} server.cpp)
target_link_libraries(${server} PRIVATE gemini IO OT)
target_compile_definitions(${server} PUBLIC USE_CHEETAH=1)
target_compile_options(
    ${server} PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra;-Wno-unused-parameter>"
    "$<$<CONFIG:Release>:-Wall;-Wextra;-Wno-unused-parameter>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers"
)

set(client cheetah_client)
add_executable(${client} client.cpp)
target_link_libraries(${client} PRIVATE gemini IO OT)
target_compile_definitions(${client} PUBLIC USE_CHEETAH=1)
target_compile_options(
    ${client} PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra;-Wno-unused-parameter>"
    "$<$<CONFIG:Release>:-Wall;-Wextra;-Wno-unused-parameter>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers"
)

message(STATUS "TRIP_VERIFY: ${TRIP_VERIFY}")
message(STATUS "TRIP_COLOR: ${TRIP_COLOR}")

if(TRIP_VERIFY)
  target_compile_definitions(${server} PRIVATE VERIFY)
  target_compile_definitions(${client} PRIVATE VERIFY)
endif()

if(TRIP_COLOR)
  target_compile_definitions(${server} PRIVATE COLOR)
  target_compile_definitions(${client} PRIVATE COLOR)
endif()

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)
set(ALL_SOURCE_FILES ${ALL_SOURCE_FILES} ${SRC_FILES} PARENT_SCOPE)
