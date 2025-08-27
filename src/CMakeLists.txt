include(CMakeFindDependencyMacro)
include(FetchContent)

## cheetah-core
add_library(gemini SHARED)
target_include_directories(gemini PUBLIC include)
target_include_directories(gemini PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(include)
target_link_libraries(gemini PUBLIC SEAL::seal Eigen3::Eigen Threads::Threads)
target_compile_options(
    gemini PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-Wall;-Wextra>"
    "-Wno-unused-parameter"
)

if (USE_APPROX_RESHARE)
  target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=1)
else ()
  target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=0)
endif ()

if(TRIP_ZERO)
  target_compile_definitions(gemini PRIVATE ZERO)
endif()


################################################################################
## Interface: static library
################################################################################
set(HE_LIB HE)
add_library(${HE_LIB} STATIC
  core/hpmpc_interface.cpp
  core/defs.cpp
  io/send.cpp
)
target_include_directories(${HE_LIB} PUBLIC include io ot)
target_link_libraries(${HE_LIB} PUBLIC SEAL::seal gemini OpenSSL::SSL OpenSSL::Crypto)
target_compile_definitions(${HE_LIB} PUBLIC USE_CHEETAH=1)
target_compile_options(
    ${HE_LIB} PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-Wall;-Wextra>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers;-Wno-unused-parameter"
)

if(TRIP_VERIFY)
  target_compile_definitions(${HE_LIB} PRIVATE VERIFY)
endif()

if(TRIP_COLOR)
  target_compile_definitions(${HE_LIB} PRIVATE COLOR)
endif()

################################################################################
## Server / Client executables
################################################################################
set(server cheetah)
add_executable(${server} server.cpp)
target_link_libraries(${server} PRIVATE ${HE_LIB})
target_compile_definitions(${server} PUBLIC USE_CHEETAH=1)
target_compile_options(
    ${server} PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-Wall;-Wextra>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers;-Wno-unused-parameter"
)

set(client cheetah_client)
add_executable(${client} client.cpp)
target_link_libraries(${client} PRIVATE ${HE_LIB})
target_compile_definitions(${client} PUBLIC USE_CHEETAH=1)
target_compile_options(
    ${client} PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-Wall;-Wextra>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers;-Wno-unused-parameter"
)

message(STATUS "TRIP_VERIFY: ${TRIP_VERIFY}")
message(STATUS "TRIP_COLOR: ${TRIP_COLOR}")

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)
set(ALL_SOURCE_FILES ${ALL_SOURCE_FILES} ${SRC_FILES} PARENT_SCOPE)
