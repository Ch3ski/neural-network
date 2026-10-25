cmake_minimum_required(VERSION 3.21.0)
message(NOTICE "Here ${CMAKE_OSX_SYSROOT}")

project(PV021-Neural-Networks CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -Wall
    -Wextra
    -Wold-style-cast
    -Werror
    -Wno-missing-field-initializers
)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Link-time optimization: lets the compiler inline small functions defined in
# other .cpp files (e.g. Vector::operator[] from vector.cpp) into hot loops.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

add_executable(network
    # Headers
    src/neural_network/layer.h
    src/neural_network/network.h
    src/neural_network/activation_functions.h

    src/linear_algebra/vector.h
    src/linear_algebra/matrix.h

    # Sources
    src/main.cpp
    
    src/neural_network/activation_functions.cpp
)

target_include_directories(network PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Tests exist only in Debug builds, so run.sh (Release) never compiles them.
# Build and run:
#   cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
#   cmake --build build-debug && ctest --test-dir build-debug --output-on-failure
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    enable_testing()

    add_executable(vector_tests tests/vector_tests.cpp)
    target_include_directories(vector_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME vector_tests COMMAND vector_tests)
endif()
