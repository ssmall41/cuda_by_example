#include <stdio.h>
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
	}
	
	if(threadIdx.x == 0)
		c[blockIdx.x] = block_buffer[0];
}

// Sum up all the elements in an array
int reduce_vector_sum(int* a, int size)
{
	int i, sum=0;
	for(i=0;i<size;i++)
		sum += a[i];
	return sum;
}

int main(int argc, char* argv[])
{
	int size = 8, i;
	int* a = (int*) malloc(size*sizeof(int));
	int* b = (int*) malloc(size*sizeof(int));
	int* c = (int*) malloc(num_blocks*sizeof(int));
	int *dev_a, *dev_b, *dev_c;
	for(i=0;i<size;i++)
	{
		a[i] = 1;
		b[i] = 2;
	}

	cudaMalloc(&dev_a, size*sizeof(int));
	cudaMalloc(&dev_b, size*sizeof(int));
	cudaMalloc(&dev_c, num_blocks*sizeof(int));
	
	cudaMemcpy(dev_a, a, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size*sizeof(int), cudaMemcpyHostToDevice);
	
	//sum_vectors<<<size, 1>>>(dev_a, dev_b, dev_c, size);
	dot_product<<<num_blocks, num_threads>>>(dev_a, dev_b, dev_c, size);

	cudaMemcpy(c, dev_c, num_blocks*sizeof(int), cudaMemcpyDeviceToHost);
	
	int dot = reduce_vector_sum(c, num_blocks);
	printf("The final dot product is %i.\n", dot);
	
	cudaFree(dev_c);
	cudaFree(dev_b);
	cudaFree(dev_a);
	free(c);
	free(b);
	free(a);
	
	return 0;
}

