################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../build/CMakeFiles/3.5.1/CompilerIdC/CMakeCCompilerId.c 

OBJS += \
./build/CMakeFiles/3.5.1/CompilerIdC/CMakeCCompilerId.o 

C_DEPS += \
./build/CMakeFiles/3.5.1/CompilerIdC/CMakeCCompilerId.d 


# Each subdirectory must supply rules for building sources it contributes
build/CMakeFiles/3.5.1/CompilerIdC/%.o: ../build/CMakeFiles/3.5.1/CompilerIdC/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2   -odir "build/CMakeFiles/3.5.1/CompilerIdC" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


