#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int num_blocks = 2;
const int num_threads = 2;


// Adds two vectors using threads and blocks.
__global__ void dot_product(int* a, int* b, int* c, int size)
{
	int i;
	__shared__ int block_buffer[num_threads];
	block_buffer[threadIdx.x] = 0;

	// Multiply and sum
	for(i=threadIdx.x+blockIdx.x*blockDim.x; i<size; i+=blockDim.x*gridDim.x)
		block_buffer[threadIdx.x] += a[i] * b[i];

	// Sum up the buffer
	i = blockDim.x / 2; // Assumes blockDim.x is a power of 2
	__syncthreads();
	while(i != 0)
	{
		if(threadIdx.x < i)
			block_buffer[threadIdx.x] += block_buffer[threadIdx.x + i];
		i /= 2;
		__syncthreads();
	}  // block_buffer[0] has the sum for this block.

	// Sum up the sums from each block
	if(threadIdx.x == 0)
		atomicAdd(c, block_buffer[0]);
}


int main(int argc, char* argv[])
{
	int size = 256, i;
	int* a = (int*) malloc(size*sizeof(int));
	int* b = (int*) malloc(size*sizeof(int));
	int c = 0;
	int *dev_a, *dev_b, *dev_c;
	for(i=0;i<size;i++)
	{
		a[i] = i;
		b[i] = 2*i;
	}

	cudaMalloc(&dev_a, size*sizeof(int));
	cudaMalloc(&dev_b, size*sizeof(int));
	cudaMalloc(&dev_c, 1*sizeof(int));

	cudaMemcpy(dev_a, a, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, &c, 1*sizeof(int), cudaMemcpyHostToDevice);

	dot_product<<<num_blocks, num_threads>>>(dev_a, dev_b, dev_c, size);

	cudaMemcpy(&c, dev_c, 1*sizeof(int), cudaMemcpyDeviceToHost);

	int size_m1 = size - 1;
	int actual = (size_m1*(size_m1+1)*(2*size_m1+1)) / 3;
	printf("The final dot product is %i.\n", c);
	printf("The actual result should be %i.\n", actual);

	cudaFree(dev_c);
	cudaFree(dev_b);
	cudaFree(dev_a);
	free(b);
	free(a);

	return 0;
}

