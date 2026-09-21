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

add_executable(network
    # Headers

    # Sources
    src/main.cpp
)
