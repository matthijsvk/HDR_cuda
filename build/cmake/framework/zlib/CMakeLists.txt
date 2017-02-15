cmake_minimum_required(VERSION 2.8)

project(zlib)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../dependencies/zlib")

file(GLOB zlib_SOURCES "${SOURCE_DIR}/*.c")

add_library(zlib STATIC ${zlib_SOURCES})

set(ZLIB_LIBRARY zlib CACHE INTERNAL "zlib library")
set(ZLIB_INCLUDE_DIR "${SOURCE_DIR}" CACHE INTERNAL "zlib include directories")


set_target_properties(zlib PROPERTIES FOLDER "framework/dependencies")
