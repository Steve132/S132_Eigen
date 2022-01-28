cmake_minimum_required (VERSION 3.16)

project(S132_Eigen VERSION 0.3.0)

add_library(S132_Eigen INTERFACE)
target_include_directories(S132_Eigen INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_compile_features(S132_Eigen INTERFACE cxx_std_17)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
	add_subdirectory(tests)
endif()
