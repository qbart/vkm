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
    foreach(t vector matrix transform layout common inverse)
        add_executable(test_${t} tests/test_${t}.cpp)
        target_link_libraries(test_${t} PRIVATE vkm)
        target_compile_options(test_${t} PRIVATE -O2 -Wall -Wextra)
        add_test(NAME test_${t} COMMAND test_${t})
    endforeach()
endif()
