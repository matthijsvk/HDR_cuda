################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../build/dependencies/zlib/adler32.c \
../build/dependencies/zlib/compress.c \
../build/dependencies/zlib/crc32.c \
../build/dependencies/zlib/deflate.c \
../build/dependencies/zlib/infback.c \
../build/dependencies/zlib/inffast.c \
../build/dependencies/zlib/inflate.c \
../build/dependencies/zlib/inftrees.c \
../build/dependencies/zlib/trees.c \
../build/dependencies/zlib/uncompr.c \
../build/dependencies/zlib/zutil.c 

OBJS += \
./build/dependencies/zlib/adler32.o \
./build/dependencies/zlib/compress.o \
./build/dependencies/zlib/crc32.o \
./build/dependencies/zlib/deflate.o \
./build/dependencies/zlib/infback.o \
./build/dependencies/zlib/inffast.o \
./build/dependencies/zlib/inflate.o \
./build/dependencies/zlib/inftrees.o \
./build/dependencies/zlib/trees.o \
./build/dependencies/zlib/uncompr.o \
./build/dependencies/zlib/zutil.o 

C_DEPS += \
./build/dependencies/zlib/adler32.d \
./build/dependencies/zlib/compress.d \
./build/dependencies/zlib/crc32.d \
./build/dependencies/zlib/deflate.d \
./build/dependencies/zlib/infback.d \
./build/dependencies/zlib/inffast.d \
./build/dependencies/zlib/inflate.d \
./build/dependencies/zlib/inftrees.d \
./build/dependencies/zlib/trees.d \
./build/dependencies/zlib/uncompr.d \
./build/dependencies/zlib/zutil.d 


# Each subdirectory must supply rules for building sources it contributes
build/dependencies/zlib/%.o: ../build/dependencies/zlib/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2   -odir "build/dependencies/zlib" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


