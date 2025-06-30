include(CMakeFindDependencyMacro)
include(FetchContent)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/deps;${CMAKE_PREFIX_PATH}")

add_compile_options(-fPIC)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

## Dependencies
find_package(Threads REQUIRED)
find_package(Eigen3 3.3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

## cheetah-core
add_library(gemini SHARED)
target_include_directories(gemini PUBLIC include)
target_include_directories(gemini PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(include)
target_link_libraries(gemini PUBLIC SEAL::seal Eigen3::Eigen Threads::Threads)

if (USE_APPROX_RESHARE)
  target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=1)
else()
  target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=0)
endif()

add_library(io INTERFACE)
target_include_directories(io INTERFACE io)
target_link_libraries(io INTERFACE SEAL::seal gemini)

set(exe cheetah)
add_executable(${exe} server.cpp)
target_link_libraries(${exe} PRIVATE gemini io)
if(VERIFY)
  target_compile_definitions(${exe} PUBLIC USE_CHEETAH=1 VERIFY=1)
else()
  target_compile_definitions(${exe} PUBLIC USE_CHEETAH=1 VERIFY=0)
endif()
target_compile_options(
    ${exe} PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra;-Wno-unused-parameter>"
)


set(exe cheetah_client)
add_executable(${exe} client.cpp)
target_link_libraries(${exe} PRIVATE gemini io)
if(VERIFY)
  target_compile_definitions(${exe} PUBLIC USE_CHEETAH=1 VERIFY=1)
else()
  target_compile_definitions(${exe} PUBLIC USE_CHEETAH=1 VERIFY=0)
endif()
target_compile_options(
    ${exe} PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra;-Wno-unused-parameter>"
)

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)
set(ALL_SOURCE_FILES ${ALL_SOURCE_FILES} ${SRC_FILES} PARENT_SCOPE)
