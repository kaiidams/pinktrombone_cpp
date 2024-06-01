cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

project(pinktrombone_cpp LANGUAGES CXX)
find_package(SDL2 CONFIG REQUIRED)
set(VCPKGLIB_BASE_SOURCES)
add_executable(pinktrombone_cpp
    src/test.cpp
    src/noise.cpp
 "include/pinktrombone.h")
target_precompile_headers(pinktrombone_cpp PRIVATE "include/pch.h")

target_include_directories(pinktrombone_cpp PRIVATE include)
target_link_libraries(pinktrombone_cpp
    PRIVATE
    SDL2::SDL2
    SDL2::SDL2main
)
