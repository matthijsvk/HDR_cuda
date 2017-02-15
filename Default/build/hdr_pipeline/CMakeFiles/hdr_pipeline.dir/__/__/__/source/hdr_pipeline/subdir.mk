################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe1.cpp 

C_SRCS += \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe1.c \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe1.stub.c \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe2.c \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe2.stub.c \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe1.c \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe1.stub.c \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe2.c \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe2.stub.c \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.fatbin.c 

O_SRCS += \
../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline_generated_hdr_pipeline.cu.o 

OBJS += \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe1.o \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe1.stub.o \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe2.o \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe2.stub.o \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe1.o \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe1.stub.o \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe2.o \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe2.stub.o \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.fatbin.o 

CPP_DEPS += \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe1.d 

C_DEPS += \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe1.d \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe1.stub.d \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe2.d \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_35.cudafe2.stub.d \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe1.d \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe1.stub.d \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe2.d \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.compute_52.cudafe2.stub.d \
./build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/hdr_pipeline.fatbin.d 


# Each subdirectory must supply rules for building sources it contributes
build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/%.o: ../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2   -odir "build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/%.o: ../build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2   -odir "build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


