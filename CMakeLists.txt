# dft_simd/CMakeLists.txt -- Stephen Fegan
#
# Copyright 2018, Stephen Fegan <sfegan@llr.in2p3.fr>
# LLR, Ecole Polytechnique, CNRS/IN2P3
#
# This file is part of "dft_simd"
#
# "dft_simd" is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 2 or later, as published by
# the Free Software Foundation.
#
# "dft_simd" is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

cmake_minimum_required(VERSION 3.3)
project(dft_simd)
enable_testing()

# ============================================================= #
# Find the various libraries and packages that calin depends on #
# ============================================================= #

## Use CMake policy to specify complete library path
cmake_policy(SET CMP0060 NEW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

set(BUILD_ARCH_OPT "-march=native")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BUILD_ARCH_OPT} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_ARCH_OPT} --save-temps -std=c++11 -Wall")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")

## Find FFTW libraries
find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDE_DIRS})
set(EXTRA_LIBS ${EXTRA_LIBS} ${FFTW_LIBRARIES} )

## Find OpenCL libraries
find_package(OpenCL)
if(OpenCL_FOUND)
	include_directories(${OpenCL_INCLUDE_DIRS})
	set(EXTRA_LIBS ${EXTRA_LIBS} ${OpenCL_LIBRARIES} )
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OPENCL=1")
endif()

# ========================================== #
# Add the subdirectories to the build system #
# ========================================== #

add_subdirectory(gtest-1.8.0)
include_directories(gtest-1.8.0/include)

set(GTEST_BOTH_LIBRARIES gtest gtest_main)

add_executable(dft_simd dft_simd)
target_link_libraries(dft_simd ${EXTRA_LIBS} ${GTEST_BOTH_LIBRARIES})
