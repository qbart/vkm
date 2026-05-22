cmake_minimum_required(VERSION 3.20)
project(vkm LANGUAGES CXX)

# Header-only interface library: nothing to compile, just headers + the C++23
# requirement and the include path.
add_library(vkm INTERFACE)
add_library(vkm::vkm ALIAS vkm)
target_include_directories(vkm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(vkm INTERFACE cxx_std_23)

# Tests are built only when vkm is the top-level project.
if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
    add_executable(vkm_tests tests/test_vector.cpp)
    target_link_libraries(vkm_tests PRIVATE vkm)
    target_compile_options(vkm_tests PRIVATE -O2 -Wall -Wextra)
    add_test(NAME vkm_tests COMMAND vkm_tests)
endif()
