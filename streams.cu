// Tests out streaming.
// One stream needs ~370ms, while 2 need ~290ms.
// There's not much difference in depth vs breadth first. However,
// for larger data sizes, depth seems to be slightly better.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

/* This is taken directly from the CUDA by Example book, chapter 10.
 * It doesn't really do anything useful, just represents some work
 * on the device. */
__global__ void kernel(int *a, int *b, int *c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < N)
	{
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0;
		c[idx] = (as + bs) / 2;
	}
}


int main(int argc, char* argv[])
{
	int i;
	float elapsed_time;

	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Create streams
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	//stream1 = stream0;

	// Initialize memory
	int *host_a, *host_b, *host_c;
	int *dev_a0, *dev_b0, *dev_c0;
	int *dev_a1, *dev_b1, *dev_c1;
	cudaMalloc(&dev_a0, N*sizeof(int));
	cudaMalloc(&dev_b0, N*sizeof(int));
	cudaMalloc(&dev_c0, N*sizeof(int));
	cudaMalloc(&dev_a1, N*sizeof(int));
	cudaMalloc(&dev_b1, N*sizeof(int));
	cudaMalloc(&dev_c1, N*sizeof(int));
	cudaHostAlloc(&host_a, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc(&host_b, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc(&host_c, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);

	for(i=0;i<FULL_DATA_SIZE;i++)
	{
		host_a[i] = i;
		host_b[i] = i;
	}

	cudaEventRecord(start, 0);

	for(int i=0;i<FULL_DATA_SIZE;i+=N*2)
	{


		// Depth first, incorrect
		// Stream 0
		cudaMemcpyAsync(dev_a0, host_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_b0, host_b+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
		kernel<<<N/256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
		cudaMemcpyAsync(host_c+i, dev_c0, N*sizeof(int), cudaMemcpyDeviceToHost, stream0);

		// Stream 1
		cudaMemcpyAsync(dev_a1, host_a+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(dev_b1, host_b+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
		kernel<<<N/256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
		cudaMemcpyAsync(host_c+i+N, dev_c1, N*sizeof(int), cudaMemcpyDeviceToHost, stream1);

/*
		// Breadth first, correct
		cudaMemcpyAsync(dev_a0, host_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_a1, host_a+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);

		cudaMemcpyAsync(dev_b0, host_b+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_b1, host_b+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);

		kernel<<<N/256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
		kernel<<<N/256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);

		cudaMemcpyAsync(host_c+i, dev_c0, N*sizeof(int), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(host_c+i+N, dev_c1, N*sizeof(int), cudaMemcpyDeviceToHost, stream1);
*/
	}

	// Sync
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	// Check timing
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Total time taken is %3.1f ms.\n", elapsed_time);

	// Trash memory
	cudaFreeHost(host_c);
	cudaFreeHost(host_b);
	cudaFreeHost(host_a);
	cudaFree(dev_c1);
	cudaFree(dev_b1);
	cudaFree(dev_a1);
	cudaFree(dev_c0);
	cudaFree(dev_b0);
	cudaFree(dev_a0);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream0);
	cudaEventDestroy(stop);
	cudaEventDestroy(start);

	return 0;
}
