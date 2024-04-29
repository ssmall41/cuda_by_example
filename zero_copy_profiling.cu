// Profiles zero-copy memory against copied memory.
// Even one iteration of the dot product is faster with copied memory.
// For 100 iterations, it's about half the speed.
// For 1 iteration, it's about a bit less than half.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 1024

// Adds two vectors using threads and blocks.
__global__ void dot_product(int* a, int* b, int* c, int size)
{
	int i;
	__shared__ int block_buffer[MAX_THREADS_PER_BLOCK];
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

void copy_to_gpu(int size, int num_blocks, int num_threads, int iterations)
{
	int i;
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

	for(i=0;i<iterations;i++)
	{
		cudaMemset(dev_c, 0, 1*sizeof(int));
		dot_product<<<num_blocks, num_threads>>>(dev_a, dev_b, dev_c, size);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(&c, dev_c, 1*sizeof(int), cudaMemcpyDeviceToHost);

	unsigned int size_m1 = size - 1;
	unsigned int actual = (size_m1*(size_m1+1)*(2*size_m1+1)) / 3;
	printf("The final dot product is %i.\n", c);
	printf("The actual result should be %u.\n", actual);

	cudaFree(dev_c);
	cudaFree(dev_b);
	cudaFree(dev_a);
	free(b);
	free(a);
}

void zero_copy(int size, int num_blocks, int num_threads, int iterations)
{
	int i, *a, *b, *c;

	cudaSetDeviceFlags(cudaDeviceMapHost);
	//cudaHostAlloc(&a, size*sizeof(int), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	//cudaHostAlloc(&b, size*sizeof(int), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc(&a, size*sizeof(int), cudaHostAllocMapped);
	cudaHostAlloc(&b, size*sizeof(int), cudaHostAllocMapped);
	cudaHostAlloc(&c, 1*sizeof(int), cudaHostAllocMapped);
	for(i=0;i<size;i++)
	{
		a[i] = i;
		b[i] = 2*i;
	}

	int *dev_a, *dev_b, *dev_c;
	cudaHostGetDevicePointer(&dev_a, a, 0);
	cudaHostGetDevicePointer(&dev_b, b, 0);
	cudaHostGetDevicePointer(&dev_c, c, 0);

	for(i=0;i<iterations;i++)
	{
		c[0] = 0;
		dot_product<<<num_blocks, num_threads>>>(dev_a, dev_b, dev_c, size);
		cudaDeviceSynchronize();
	}

	unsigned int size_m1 = size - 1;
	unsigned int actual = (size_m1*(size_m1+1)*(2*size_m1+1)) / 3;
	printf("The final dot product is %i.\n", c[0]);
	printf("The actual result should be %u.\n", actual);

	cudaFreeHost(c);
	cudaFreeHost(b);
	cudaFreeHost(a);
}


int main(int argc, char* argv[])
{
	int size = 1024*2;  // Can't go too big. Floats were probably better...
	int num_blocks = 4, num_threads = 128, iterations = 1;
	float elapsed_time;

	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("Running copy version.\n");
	cudaEventRecord(start, 0);
	copy_to_gpu(size, num_blocks, num_threads, iterations);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Total time %3.2f ms\n", elapsed_time);

	printf("Running zero-copy version.\n");
	cudaEventRecord(start, 0);
	zero_copy(size, num_blocks, num_threads, iterations);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Total time %3.2f ms\n", elapsed_time);


	cudaEventDestroy(stop);
	cudaEventDestroy(start);

	return 0;
}
