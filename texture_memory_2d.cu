// Tries 2d texture memory. Can be compared against untexture_memory.cu.
// It runs about as fast as the untextured version.
// btw you must have dim has a multiple of prop.texturePitchAlignment. I was too lazy
// to implement proper padding to make it work properly with arbitrary dim.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void copy_heater_buffer(cudaTextureObject_t tex_heater_buffer, float* in_buffer, int dim)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	float heater_value = tex2D<float>(tex_heater_buffer, x, y);
	if(x < dim && y < dim && heater_value > 0.0)
		in_buffer[offset] = heater_value;
}

__global__ void heat_spread(float* out_buffer, cudaTextureObject_t tex_in_buffer, int dim, float speed)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	if(x > dim || y > dim)	return;
	
	float center = tex2D<float>(tex_in_buffer, x, y);
	float sum_all_directions = tex2D<float>(tex_in_buffer, x-1, y) + tex2D<float>(tex_in_buffer, x+1, y) + tex2D<float>(tex_in_buffer, x, y-1) + tex2D<float>(tex_in_buffer, x, y+1);
	out_buffer[offset] = center + speed * (sum_all_directions - center*4);
}

void swap_tex_buffers(cudaTextureObject_t* tex_in_buffer, cudaTextureObject_t* tex_out_buffer)
{
	cudaTextureObject_t holder = *tex_in_buffer;
	*tex_in_buffer = *tex_out_buffer;
	*tex_out_buffer = holder;
}

void print_buffer(float* buffer, int size)
{
	int i, j;
	printf("#########\n");
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
			printf("%f ", buffer[i + j*size]);
		printf("\n");
	}
}


void print_device_buffer(float* dev_buffer, int size)
{
	float* temp_buffer = (float*) malloc(size*size*sizeof(float));
	cudaMemcpy(temp_buffer, dev_buffer, size*size*sizeof(float), cudaMemcpyDeviceToHost);
	print_buffer(temp_buffer, size);
	free(temp_buffer);
}

cudaTextureObject_t define_texture_memory(float* dev_data, int size)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int aligned_width = prop.texturePitchAlignment;
	while(aligned_width < size)
		aligned_width += prop.texturePitchAlignment;

	// Define the Resource Descriptor
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = dev_data;
	resDesc.res.pitch2D.width = aligned_width;
	resDesc.res.pitch2D.height = size;
	resDesc.res.pitch2D.pitchInBytes = aligned_width*sizeof(float);
	resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.pitch2D.desc.x = 32;
	resDesc.res.pitch2D.desc.y = 32;

	// Define the Texture Descriptor
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	// Create the Texture Object
	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
	
	return tex;
}

float* get_devPtr(cudaTextureObject_t texObject)
{
	cudaResourceDesc desc;
	cudaGetTextureObjectResourceDesc (&desc, texObject);
	return (float*) desc.res.linear.devPtr;
}

int main(int argc, char* argv[])
{
	int i, dim = 32, num_iterations = 20000;
	dim3 num_blocks(2, 2), num_threads(dim/2, dim/2);
	//dim3 num_blocks(1, 1), num_threads(dim, dim);
	float speed = 0.2;
	int buffer_size = dim*dim;
	
	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Reserve memory
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	float* heater_buffer = (float*) calloc(buffer_size, sizeof(float));
	float *dev_heater_buffer, *dev_in_buffer, *dev_out_buffer;
	cudaMalloc(&dev_heater_buffer, buffer_size*sizeof(float));
	cudaMalloc(&dev_in_buffer, buffer_size*sizeof(float));
	cudaMalloc(&dev_out_buffer, buffer_size*sizeof(float));
	cudaTextureObject_t tex_heater_buffer = define_texture_memory(dev_heater_buffer, dim);
	cudaTextureObject_t tex_in_buffer = define_texture_memory(dev_in_buffer, dim);
	cudaTextureObject_t tex_out_buffer = define_texture_memory(dev_out_buffer, dim);
	
	// Initialize the heaters
	heater_buffer[0 + 0*dim] = 1.0;
	heater_buffer[1 + 2*dim] = 1.0;
	/*
	int j;
	for(i=0;i<dim;i++)
		for(j=0;j<dim;j++)
			heater_buffer[i + j*dim] = 1.0;
	*/
	
	// Send data to the GPU
	cudaMemcpy(dev_heater_buffer, heater_buffer, buffer_size*sizeof(float), cudaMemcpyHostToDevice);

	printf("Hello World\n");
	//print_device_buffer(dev_heater_buffer, dim);

	// Let the heat spread
	cudaEventRecord(start, 0);
	for(i=0;i<num_iterations;i++)
	{
		//printf("Iteration %i\n", i);
		copy_heater_buffer<<<num_blocks, num_threads>>>(tex_heater_buffer, dev_in_buffer, dim);
		//print_device_buffer(dev_in_buffer, dim);
		heat_spread<<<num_blocks, num_threads>>>(dev_out_buffer, tex_in_buffer, dim, speed);
		//print_device_buffer(dev_out_buffer, dim);

		swap_tex_buffers(&tex_in_buffer, &tex_out_buffer);
		dev_in_buffer = get_devPtr(tex_in_buffer);  // Needed in order to switch the dev pointers
		dev_out_buffer = get_devPtr(tex_out_buffer);
	}
	copy_heater_buffer<<<num_blocks, num_threads>>>(tex_heater_buffer, dev_in_buffer, dim);
	cudaEventRecord(stop, 0);

	// Copy the input buffer to the host
	cudaMemcpy(heater_buffer, dev_in_buffer, buffer_size*sizeof(float), cudaMemcpyDeviceToHost);
	
	// Print for sanity
	//print_buffer(heater_buffer, dim);

	//Check the runtime
	float elapsed_time;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Total time %3.2f ms\n", elapsed_time);

	// Free memory
	cudaDestroyTextureObject(tex_out_buffer);
	cudaDestroyTextureObject(tex_in_buffer);
	cudaDestroyTextureObject(tex_heater_buffer);
	cudaFree(dev_out_buffer);
	cudaFree(dev_in_buffer);
	cudaFree(dev_heater_buffer);
	free(heater_buffer);

	return 0;
}

