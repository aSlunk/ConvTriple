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
include_directories(${EIGEN3_INCLUDE_DIR}
                    ${EMP-OT_INCLUDE_DIRS}
                    ${EMP-TOOL_INCLUDE_DIRS})

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

add_library(OT INTERFACE)
target_include_directories(OT INTERFACE ot)
target_link_libraries(OT INTERFACE OpenSSL::SSL OpenSSL::Crypto)

add_library(IO INTERFACE)
target_include_directories(IO INTERFACE io)
target_link_libraries(IO INTERFACE SEAL::seal gemini OT)

set(exe cheetah)
add_executable(${exe} server.cpp)
target_link_libraries(${exe} PRIVATE gemini IO OT)
if(VERIFY)
  target_compile_definitions(${exe} PUBLIC USE_CHEETAH=1 VERIFY=1)
else()
  target_compile_definitions(${exe} PUBLIC USE_CHEETAH=1 VERIFY=0)
endif()
target_compile_options(
    ${exe} PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra;-Wno-unused-parameter>"
    "$<$<CONFIG:Release>:-Wall;-Wextra;-Wno-unused-parameter>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers"
)


set(exe cheetah_client)
add_executable(${exe} client.cpp)
target_link_libraries(${exe} PRIVATE gemini IO OT)
if(VERIFY)
  target_compile_definitions(${exe} PUBLIC USE_CHEETAH=1 VERIFY=1)
else()
  target_compile_definitions(${exe} PUBLIC USE_CHEETAH=1 VERIFY=0)
endif()
target_compile_options(
    ${exe} PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra;-Wno-unused-parameter>"
    "$<$<CONFIG:Release>:-Wall;-Wextra;-Wno-unused-parameter>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers"
)

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)
set(ALL_SOURCE_FILES ${ALL_SOURCE_FILES} ${SRC_FILES} PARENT_SCOPE)
