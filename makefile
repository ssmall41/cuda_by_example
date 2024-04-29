-include ../makefile.init

RM := rm -f

BUILD_ARTIFACT_NAME := cuda_by_example
BUILD_ARTIFACT_EXTENSION :=
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 
SRC := streams.cu
OBJS := objs/streams.o


# All Target
all: main-build

# Main-build Target
main-build: cuda_by_example

# Tool invocations

# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-12.4/bin/nvcc -Xcompiler -fopenmp --device-debug --debug -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -ccbin g++ -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

# Link
cuda_by_example: $(OBJS) $(USER_OBJS) makefile $(OPTIONAL_TOOL_DEPS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC linker'
	/usr/local/cuda-12.4/bin/nvcc -Xcompiler -fopenmp --cudart=static -ccbin g++ -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o "cuda_by_example" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) cuda_by_example
	-$(RM) objs/*
	-@echo ' '

.PHONY: all clean dependents main-build

-include ../makefile.targets
