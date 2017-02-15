cmake_minimum_required(VERSION 2.8)

project(framework)


set(DEPENDENCIES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies")

add_definitions(-DZ_SOLO)
add_subdirectory(zlib zlib)
add_subdirectory(libpng libpng)


set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../source/framework")

set(Framework_INCLUDE_DIRS_internal
	${LPNG_INCLUDE_DIR}
	${CUDA_INCLUDE_DIRS}
	${DEPENDENCIES_DIR}
	"${SOURCE_DIR}/../"
)

include_directories(${Framework_INCLUDE_DIRS_internal})


file(GLOB FRAMEWORK_SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.h" "${SOURCE_DIR}/CUDA/*.cpp" "${SOURCE_DIR}/CUDA/*.h")

add_library(framework ${FRAMEWORK_PLATFORM_SOURCES} ${FRAMEWORK_SOURCES})


set(Framework_INCLUDE_DIRS ${Framework_INCLUDE_DIRS_internal} CACHE INTERNAL "framework include directories")
set(Framework_LIBRARIES framework ${LPNG_LIBRARY} ${ZLIB_LIBRARY} ${CMAKE_DL_LIBS} ${CUDA_CUDART_LIBRARY} CACHE INTERNAL "framework libraries")


set_target_properties(framework PROPERTIES FOLDER "framework")
