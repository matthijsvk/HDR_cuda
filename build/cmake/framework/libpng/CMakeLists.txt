cmake_minimum_required(VERSION 2.8)

project(lpng)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../dependencies/libpng")

file(GLOB lpng_SOURCES "${SOURCE_DIR}/*.c")

include_directories(${ZLIB_INCLUDE_DIR})
add_library(lpng STATIC ${lpng_SOURCES})

set(LPNG_LIBRARY lpng CACHE INTERNAL "lpng library")
set(LPNG_INCLUDE_DIR "${SOURCE_DIR}" CACHE INTERNAL "lpng include directories")


set_target_properties(lpng PROPERTIES FOLDER "framework/dependencies")
