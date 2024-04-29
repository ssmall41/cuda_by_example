// Tests using an atomic operation to see how it can all go wrong.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void lots_o_incs(int* value)
{
	// Safe
	atomicAdd(value, 1);

	// Not safe
	//(*value) += 1;
}


int main(int argc, char* argv[])
{
	int *dev_value, value = 0;
	cudaMalloc(&dev_value, 1*sizeof(int));
	cudaMemcpy(dev_value, &value, 1*sizeof(int), cudaMemcpyHostToDevice);

	lots_o_incs<<<2, 32>>>(dev_value);

	cudaMemcpy(&value, dev_value, 1*sizeof(int), cudaMemcpyDeviceToHost);

	printf("Value is %i\n", value);

	cudaFree(dev_value);

	return 0;
}
