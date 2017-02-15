################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../build/dependencies/libpng/png.c \
../build/dependencies/libpng/pngerror.c \
../build/dependencies/libpng/pngget.c \
../build/dependencies/libpng/pngmem.c \
../build/dependencies/libpng/pngpread.c \
../build/dependencies/libpng/pngread.c \
../build/dependencies/libpng/pngrio.c \
../build/dependencies/libpng/pngrtran.c \
../build/dependencies/libpng/pngrutil.c \
../build/dependencies/libpng/pngset.c \
../build/dependencies/libpng/pngtrans.c \
../build/dependencies/libpng/pngwio.c \
../build/dependencies/libpng/pngwrite.c \
../build/dependencies/libpng/pngwtran.c \
../build/dependencies/libpng/pngwutil.c 

OBJS += \
./build/dependencies/libpng/png.o \
./build/dependencies/libpng/pngerror.o \
./build/dependencies/libpng/pngget.o \
./build/dependencies/libpng/pngmem.o \
./build/dependencies/libpng/pngpread.o \
./build/dependencies/libpng/pngread.o \
./build/dependencies/libpng/pngrio.o \
./build/dependencies/libpng/pngrtran.o \
./build/dependencies/libpng/pngrutil.o \
./build/dependencies/libpng/pngset.o \
./build/dependencies/libpng/pngtrans.o \
./build/dependencies/libpng/pngwio.o \
./build/dependencies/libpng/pngwrite.o \
./build/dependencies/libpng/pngwtran.o \
./build/dependencies/libpng/pngwutil.o 

C_DEPS += \
./build/dependencies/libpng/png.d \
./build/dependencies/libpng/pngerror.d \
./build/dependencies/libpng/pngget.d \
./build/dependencies/libpng/pngmem.d \
./build/dependencies/libpng/pngpread.d \
./build/dependencies/libpng/pngread.d \
./build/dependencies/libpng/pngrio.d \
./build/dependencies/libpng/pngrtran.d \
./build/dependencies/libpng/pngrutil.d \
./build/dependencies/libpng/pngset.d \
./build/dependencies/libpng/pngtrans.d \
./build/dependencies/libpng/pngwio.d \
./build/dependencies/libpng/pngwrite.d \
./build/dependencies/libpng/pngwtran.d \
./build/dependencies/libpng/pngwutil.d 


# Each subdirectory must supply rules for building sources it contributes
build/dependencies/libpng/%.o: ../build/dependencies/libpng/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2   -odir "build/dependencies/libpng" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


