#include <stdio.h>

__global__ void kernel(int i, int* i_plus_5)
{
	*i_plus_5 = i + 5;
}

int main(int argc, char* argv[])
{
	int i = 2, i_plus_5;
	int *dev_i_plus_5;
	cudaMalloc(&dev_i_plus_5, sizeof(int));
	
	kernel<<<1,1>>>(i, dev_i_plus_5);
	cudaMemcpy(&i_plus_5, dev_i_plus_5, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("The value is %i\n", i_plus_5);
	cudaFree(dev_i_plus_5);
	
	return 0;
}

