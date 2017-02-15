cmake_minimum_required(VERSION 3.3)

include("add_configuration.cmake")
include("add_subdirectory_if_exists.cmake")

project(BEST2017)

set(CMAKE_DEBUG_POSTFIX D)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (WIN32)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
	add_definitions(-DGLCORE_STATIC)
else ()
	set(CMAKE_C_FLAGS "-std=c90")
	set(CMAKE_CXX_FLAGS "-std=c++11")
endif ()

add_configuration(Submission RelWithDebInfo)

include(cuda.cmake)

add_subdirectory(framework)

add_subdirectory_if_exists(hdr_pipeline)
