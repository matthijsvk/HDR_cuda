################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../source/hdr_pipeline/hdr_pipeline.cu 

CPP_SRCS += \
../source/hdr_pipeline/HDRPipeline.cpp \
../source/hdr_pipeline/main.cpp 

OBJS += \
./source/hdr_pipeline/HDRPipeline.o \
./source/hdr_pipeline/hdr_pipeline.o \
./source/hdr_pipeline/main.o 

CU_DEPS += \
./source/hdr_pipeline/hdr_pipeline.d 

CPP_DEPS += \
./source/hdr_pipeline/HDRPipeline.d \
./source/hdr_pipeline/main.d 


# Each subdirectory must supply rules for building sources it contributes
source/hdr_pipeline/%.o: ../source/hdr_pipeline/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2   -odir "source/hdr_pipeline" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

source/hdr_pipeline/%.o: ../source/hdr_pipeline/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2   -odir "source/hdr_pipeline" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 --compile --relocatable-device-code=false  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


