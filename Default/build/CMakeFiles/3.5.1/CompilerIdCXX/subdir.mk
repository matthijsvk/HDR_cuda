################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../build/CMakeFiles/3.5.1/CompilerIdCXX/CMakeCXXCompilerId.cpp 

OBJS += \
./build/CMakeFiles/3.5.1/CompilerIdCXX/CMakeCXXCompilerId.o 

CPP_DEPS += \
./build/CMakeFiles/3.5.1/CompilerIdCXX/CMakeCXXCompilerId.d 


# Each subdirectory must supply rules for building sources it contributes
build/CMakeFiles/3.5.1/CompilerIdCXX/%.o: ../build/CMakeFiles/3.5.1/CompilerIdCXX/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2   -odir "build/CMakeFiles/3.5.1/CompilerIdCXX" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


