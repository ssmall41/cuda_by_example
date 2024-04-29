// Compares the transfer rate to a device with page enabled and pinned memory on the host.
// It looks like the difference is small:
// page enabled memory is ~1.41 GB/s and pinned memory ~1.52 GB/s.
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


float cuda_page_test(int N, int num_trials)
{
	int *values, *dev_values, i;
	float elapsed_time;

	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Load up the data
	values = (int*) malloc(N*sizeof(int));
	cudaMalloc(&dev_values, N*sizeof(int));
	for(i=0;i<N;i++)	values[i] = i;

	// Try copying to the device
	cudaEventRecord(start, 0);
	for(i=0;i<num_trials;i++)
		cudaMemcpy(dev_values, values, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);

	// Catch the elapsed time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);

	// Free memory
	cudaFree(dev_values);
	free(values);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsed_time;
}

float cuda_pinned_test(int N, int num_trials)
{
	int *values, *dev_values, i;
	float elapsed_time;

	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Load up the data
	cudaHostAlloc(&values, N*sizeof(int), cudaHostAllocDefault);
	cudaMalloc(&dev_values, N*sizeof(int));
	for(i=0;i<N;i++)	values[i] = i;

	// Try copying to the device
	cudaEventRecord(start, 0);
	for(i=0;i<num_trials;i++)
		cudaMemcpy(dev_values, values, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);

	// Catch the elapsed time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);

	// Free memory
	cudaFree(dev_values);
	cudaFreeHost(values);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsed_time;
}

int main(int argc, char* argv[])
{
	int N = 100*1024*1024, num_trials = 100;
	float page_time, pinned_time;
	float convert_to_rate = 1000.0*num_trials*N*sizeof(int)/1024.0/1024.0/1024.0;  // Converts to GB/s

	page_time = cuda_page_test(N, num_trials);
	pinned_time = cuda_pinned_test(N, num_trials);

	printf("Total time for transferring from page enabled memory: %f s\n", page_time/1000);
	printf("Average rate %f GB/s\n", convert_to_rate/page_time);
	printf("Total time for transferring from pinned memory: %f s\n", pinned_time/1000);
	printf("Average rate %f GB/s\n", convert_to_rate/pinned_time);

	return 0;
}
