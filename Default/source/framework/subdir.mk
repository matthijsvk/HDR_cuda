################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../source/framework/cmd_args.cpp \
../source/framework/pfm.cpp \
../source/framework/png.cpp 

OBJS += \
./source/framework/cmd_args.o \
./source/framework/pfm.o \
./source/framework/png.o 

CPP_DEPS += \
./source/framework/cmd_args.d \
./source/framework/pfm.d \
./source/framework/png.d 


# Each subdirectory must supply rules for building sources it contributes
source/framework/%.o: ../source/framework/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2   -odir "source/framework" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


