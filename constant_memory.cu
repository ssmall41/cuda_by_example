// This computes the dot product of a constant vector and a bunch of other vectors.
// It's disigned to check how constant memory works.
// Constant memory seems to be slightly faster if we have a small-ish number of blocks and threads.
// The difference goes away when the blocks/threads gets big.

#include <stdio.h>

const int SIZE = 10000;
__constant__ float dev_core_vector[SIZE];


__global__ void compute_dot_prods(float* vectors_data, float* dots, int num_vectors)
{
	int j, i = threadIdx.x + blockIdx.x*blockDim.x;
	while(i < num_vectors)
	{
		dots[i] = 0.0;
		for(j=0;j<SIZE;j++)
			dots[i] += dev_core_vector[j] * (vectors_data[j + i*SIZE]);
		
		i += blockDim.x*gridDim.x;
	}	
}


int main(int argc, char* argv[])
{
	int i, j, num_vectors = 10000;
	int num_blocks = 8, num_threads = 8;
	cudaEvent_t start, stop;
	
	// Create events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Load up the core_vector
	float* temp_core = (float*) malloc(SIZE*sizeof(float));
	for(i=0;i<SIZE;i++)
		temp_core[i] = i - SIZE/2;
	cudaMemcpyToSymbol(dev_core_vector, temp_core, sizeof(float)*SIZE);
	free(temp_core);
	
	// Load up the other vectors
	float* vectors_data = (float*) malloc(num_vectors*SIZE*sizeof(float));
	float** vectors = (float**) malloc(num_vectors*sizeof(float*));
	for(i=0;i<num_vectors;i++)
		vectors[i] = &vectors_data[i*SIZE];
	for(i=0;i<num_vectors;i++)
		for(j=0;j<SIZE;j++)
			vectors[i][j] = i + j;
	free(vectors);
	
	// Copy vectors to the device
	float* dev_vectors_data;
	cudaMalloc(&dev_vectors_data, num_vectors*SIZE*sizeof(float));
	cudaMemcpy(dev_vectors_data, vectors_data, num_vectors*SIZE*sizeof(float), cudaMemcpyHostToDevice);
	free(vectors_data);
	
	// Make space for the dot products
	float* dots = (float*) malloc(num_vectors*sizeof(float));
	float* dev_dots;
	cudaMalloc(&dev_dots, num_vectors*sizeof(float));
	
	// Do the dot products
	cudaEventRecord(start, 0);
	compute_dot_prods<<<num_blocks, num_threads>>>(dev_vectors_data, dev_dots, num_vectors);
	cudaEventRecord(stop, 0);
	
	// Get the dot products
	cudaMemcpy(dots, dev_dots, num_vectors*sizeof(float), cudaMemcpyDeviceToHost);
	int limit = (num_vectors > 10) ? 10 : num_vectors;
	for(i=0;i<limit;i++)
		printf("%f ", dots[i]);
	printf("\n");
	
	// Free space
	cudaFree(dev_dots);
	free(dots);
	cudaFree(dev_vectors_data);
	
	// Create events
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Total time %3.1f ms\n", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	return 0;
}

