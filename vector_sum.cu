#include <stdio.h>

// Adds two vectors. The maximum number of blocks is a limitation.
__global__ void sum_vectors(int* a, int* b, int* c, int size)
{
	int i = blockIdx.x;
	if(i < size)
		c[i] = a[i] + b[i];
}

// Adds two vectors using threads and blocks.
__global__ void sum_vectors_arbitrary(int* a, int* b, int* c, int size)
{
	int i;
	for(i=threadIdx.x+blockIdx.x*blockDim.x; i<size; i+=blockDim.x*gridDim.x)
		c[i] = a[i] + b[i];
}

int main(int argc, char* argv[])
{
	int size = 20, i;
	int* a = (int*) malloc(size*sizeof(int));
	int* b = (int*) malloc(size*sizeof(int));
	int* c = (int*) malloc(size*sizeof(int));
	int *dev_a, *dev_b, *dev_c;
	for(i=0;i<size;i++)
	{
		a[i] = i;
		b[i] = 2*i;
	}

	cudaMalloc(&dev_a, size*sizeof(int));
	cudaMalloc(&dev_b, size*sizeof(int));
	cudaMalloc(&dev_c, size*sizeof(int));
	
	cudaMemcpy(dev_a, a, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size*sizeof(int), cudaMemcpyHostToDevice);
	
	//sum_vectors<<<size, 1>>>(dev_a, dev_b, dev_c, size);
	sum_vectors_arbitrary<<<2, 2>>>(dev_a, dev_b, dev_c, size);

	cudaMemcpy(c, dev_c, size*sizeof(int), cudaMemcpyDeviceToHost);
	
	for(i=0;i<size;i++)
		printf("%i ", c[i]);
	printf("\n");
	
	cudaFree(dev_c);
	cudaFree(dev_b);
	cudaFree(dev_a);
	free(c);
	free(b);
	free(a);
	
	return 0;
}

