cmake_minimum_required(VERSION 3.20)
project(paged_kv_traversal_study LANGUAGES CXX)

add_executable(paged_kv_study src/paged_kv_study.cpp)
target_compile_features(paged_kv_study PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(paged_kv_study PRIVATE /O2 /W4)
else()
  target_compile_options(paged_kv_study PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
endif()
