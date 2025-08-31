## cheetah-core
add_library(gemini SHARED)
add_subdirectory(include)
target_include_directories(gemini PUBLIC include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gemini PUBLIC SEAL::seal Eigen3::Eigen)
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

if(TRIPLE_ZERO)
    target_compile_definitions(gemini PRIVATE ZERO)
endif()

target_include_directories(gemini PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${EMP-OT_INCLUDE_DIRS}
    ${EMP-TOOL_INCLUDE_DIRS}
)

################################################################################
## Interface: static library
################################################################################
set(HE_LIB HE)
add_library(${HE_LIB} STATIC
    core/hpmpc_interface.cpp
    core/defs.cpp
    io/send.cpp
)
message(STATUS ${EIGEN3_INCLUDE_DIR})
target_include_directories(${HE_LIB} PUBLIC io include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${HE_LIB} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(${HE_LIB} PUBLIC gemini)
target_compile_definitions(${HE_LIB} PUBLIC USE_CHEETAH=1)
target_compile_options(
    ${HE_LIB} PRIVATE
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-Wall;-Wextra>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers;-Wno-unused-parameter"
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
    "$<$<CONFIG:DEBUG>:-g3;-ggdb;-Og;-Wall;-Wextra>"
    "$<$<CONFIG:Release>:-Wall;-Wextra>"
    "-march=native;-Wno-ignored-attributes;-Wno-ignored-qualifiers;-Wno-unused-parameter"
)
if(TRIPLE_COLOR)
    target_compile_definitions(${server} PRIVATE COLOR)
endif()

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
if(TRIPLE_COLOR)
    target_compile_definitions(${client} PRIVATE COLOR)
endif()

message(STATUS "TRIPLE_VERIFY: ${TRIPLE_VERIFY}")
message(STATUS "TRIPLE_COLOR: ${TRIPLE_COLOR}")

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)
set(ALL_SOURCE_FILES ${ALL_SOURCE_FILES} ${SRC_FILES} PARENT_SCOPE)
