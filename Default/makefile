################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include source/hdr_pipeline/subdir.mk
-include source/framework/subdir.mk
-include source/framework/CUDA/subdir.mk
-include build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/mnt/data/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/source/hdr_pipeline/subdir.mk
-include build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline/subdir.mk
-include build/framework/zlib/CMakeFiles/zlib.dir/__/__/dependencies/zlib/subdir.mk
-include build/framework/libpng/CMakeFiles/lpng.dir/__/__/dependencies/libpng/subdir.mk
-include build/framework/CMakeFiles/framework.dir/mnt/data/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/source/framework/subdir.mk
-include build/framework/CMakeFiles/framework.dir/mnt/data/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/source/framework/CUDA/subdir.mk
-include build/dependencies/zlib/subdir.mk
-include build/dependencies/libpng/subdir.mk
-include build/CMakeFiles/subdir.mk
-include build/CMakeFiles/3.5.1/CompilerIdCXX/subdir.mk
-include build/CMakeFiles/3.5.1/CompilerIdC/subdir.mk

OS_SUFFIX := $(subst Linux,linux,$(subst Darwin/x86_64,darwin,$(shell uname -s)/$(shell uname -m)))

-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(CU_DEPS)),)
-include $(CU_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# OpenGL+GLUT OS-specific define
ifeq ($(shell uname -s),Darwin)
GLUT_LIBS := -Xlinker -framework -Xlinker GLUT -Xlinker -framework -Xlinker OpenGL
else
GLUT_LIBS := -lGL -lGLU -lglut 
endif


# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: HDR2

# Tool invocations
HDR2: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	/usr/local/cuda-8.0/bin/nvcc --cudart static --relocatable-device-code=false -link -o  "HDR2" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CC_DEPS)$(C++_DEPS)$(EXECUTABLES)$(C_UPPER_DEPS)$(CXX_DEPS)$(OBJS)$(CU_DEPS)$(CPP_DEPS)$(C_DEPS) HDR2
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
