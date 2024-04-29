//Complie: nvcc info.cu
//Run: ./a.out

#include <stdio.h>

int main()
{
    cudaDeviceProp  prop;

    int count,i;
    cudaError_t val = cudaGetDeviceCount( &count );
    if(val != cudaSuccess)
    {
	printf("Error getting device count:\n");
	if(val == cudaErrorNoDevice)
		printf("No device\n");
	else if(val == cudaErrorInsufficientDriver)
		printf("Insufficient driver\n");
	else
	{
		printf("Got error %i\n", val);
		const char* errorName = cudaGetErrorName(val);
		const char* errorString = cudaGetErrorString(val);
		printf("Error name: %s.\n", errorName);
		printf("Error string: %s.\n", errorString);
	}
	return 1;
    }

    for (i=0; i<count; i++) {
        cudaGetDeviceProperties( &prop, i );
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        
        /* Deprecated for asyncEngineCount
        printf( "Device copy overlap:  " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n");
        */
        
        printf("Number of async engines: %i\n", prop.asyncEngineCount);
            
        printf( "Kernel execution timeout :  " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );
        printf("Concurrent kernel executions: %i\n", prop.concurrentKernels);

        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",
                    prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2] );
        printf( "\n" );
	printf("***********\n");
    }

	return 0;
}

