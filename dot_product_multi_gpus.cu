#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#define MAX_THREADS_PER_BLOCK 1024

struct localData
{
	int deviceID;
	int my_size;
	float *a;
	float *b;
	float partialDot;
};


// Adds two vectors using threads and blocks.
__global__ void dot_product(float* a, float* b, float* c, int size)
{
	int i;
	__shared__ float block_buffer[MAX_THREADS_PER_BLOCK];
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
float reduce_vector_sum(float* a, int size)
{
	int i;
	float sum = 0;
	for(i=0;i<size;i++)
		sum += a[i];
	return sum;
}


void compute_dot_product(localData *myData, int num_blocks, int num_threads)
{
	int my_size = myData->my_size;
	float *my_a = myData->a, *my_b = myData->b;
	float *dev_a, *dev_b, *dev_c, *my_c;

	cudaSetDevice(myData->deviceID);
	cudaMalloc(&dev_a, my_size*sizeof(float));
	cudaMalloc(&dev_b, my_size*sizeof(float));
	cudaMalloc(&dev_c, num_blocks*sizeof(float));
	my_c = (float*) malloc(my_size*sizeof(float));

	cudaMemcpy(dev_a, my_a, my_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, my_b, my_size*sizeof(float), cudaMemcpyHostToDevice);

	dot_product<<<num_blocks, num_threads>>>(dev_a, dev_b, dev_c, my_size);

	cudaMemcpy(my_c, dev_c, num_blocks*sizeof(float), cudaMemcpyDeviceToHost);

	/*
	int i;
	for(i=0;i<num_blocks;i++)
		printf("%f ", my_c[i]);
	printf("\n");
	*/

	float dot = reduce_vector_sum(my_c, num_blocks);
	printf("My dot product is %f.\n", dot);

	free(my_c);
	cudaFree(dev_c);
	cudaFree(dev_b);
	cudaFree(dev_a);

	myData->partialDot = dot;
}

int main(int argc, char* argv[])
{
	int deviceCount, i;
	int N = 1024, num_blocks = 2, num_threads = 32;
	int max_devices = 1;

	// Check the devices
	cudaGetDeviceCount(&deviceCount);
	if(deviceCount == 0)
	{
		printf("No GPU found.\n");
		return 1;
	}
	else
	{
		deviceCount = (deviceCount > max_devices) ? max_devices : deviceCount;
		printf("Using %i GPUs.\n", deviceCount);
	}

	// Reserve memory and initialize the data
	float *a = (float*) malloc(N*sizeof(float));
	float *b = (float*) malloc(N*sizeof(float));
	for(i=0;i<N;i++)
	{
		a[i] = (float) i;
		b[i] = 2.0*i;
	}

	// Initialize the GPU's data
	localData *data = (localData*) malloc(deviceCount*sizeof(localData));
	for(i=0;i<deviceCount;i++)
	{
		data[i].deviceID = i;
		data[i].my_size = N / deviceCount;	// Assumes a nice integer division...
		data[i].a = &(a[data[i].my_size*i]);
		data[i].b = &(b[data[i].my_size*i]);
	}

	// Run the dot products
	// Note: this should be multi-threaded to work correctly. But I only have 1 GPU...
	float dot = 0;
	#pragma omp parallel num_threads(deviceCount)
	{
		//for(i=0;i<deviceCount;i++)
		compute_dot_product(&(data[omp_get_thread_num()]), num_blocks, num_threads);
	}
	for(i=0;i<deviceCount;i++)
		dot += data[i].partialDot;

	// Print the results
	int N_m1 = N - 1;
	float actual = (N_m1*(N_m1+1)*(2*N_m1+1)) / 3.0;
	printf("Final dot product is %f.\n", dot);
	printf("The actual dot product is %f\n", actual);


	// Clean up memory
	free(data);
	free(b);
	free(a);

	return 0;
}

