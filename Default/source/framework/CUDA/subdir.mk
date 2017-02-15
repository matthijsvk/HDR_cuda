################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../source/framework/CUDA/error.cpp 

OBJS += \
./source/framework/CUDA/error.o 

CPP_DEPS += \
./source/framework/CUDA/error.d 


# Each subdirectory must supply rules for building sources it contributes
source/framework/CUDA/%.o: ../source/framework/CUDA/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2   -odir "source/framework/CUDA" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


