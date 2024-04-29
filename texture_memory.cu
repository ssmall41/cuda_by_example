// Tries texture memory. Can be compared against untexture_memory.cu.
// Using texture memory seems to require MORE time here.
// It seems using dim as a multiple of prop.texturePitchAlignment helps.
// I suspect the resource descriptor or so was suppose to use this somehow.

#include <stdio.h>

__global__ void copy_heater_buffer(cudaTextureObject_t tex_heater_buffer, float* in_buffer, int dim)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	float heater_value = tex1Dfetch<float>(tex_heater_buffer, offset);
	if(x < dim && y < dim && heater_value > 0.0)
		in_buffer[offset] = heater_value;
}

__global__ void heat_spread(float* out_buffer, cudaTextureObject_t tex_in_buffer, int dim, float speed)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	if(x > dim || y > dim)	return;
	
	int l = offset - 1;
	int r = offset + 1;
	if(x == 0) l++;
	if(x == dim-1) r--;
	
	int top = offset - dim;
	int bot = offset + dim;
	if(y == 0) top += dim;
	if(y == dim-1) bot -= dim;
	
	float sum_all_directions = tex1Dfetch<float>(tex_in_buffer, l) + tex1Dfetch<float>(tex_in_buffer, r) + tex1Dfetch<float>(tex_in_buffer, top) + tex1Dfetch<float>(tex_in_buffer, bot);
	out_buffer[offset] = tex1Dfetch<float>(tex_in_buffer, offset) + speed * (sum_all_directions - tex1Dfetch<float>(tex_in_buffer, offset)*4);
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
	// Define the Resource Descriptor
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	//struct cudaResourceDesc* resDesc = (struct cudaResourceDesc*) calloc(1, sizeof(struct cudaResourceDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = dev_data;
	resDesc.res.linear.sizeInBytes = size*size*sizeof(float);
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32;

	// Define the Texture Descriptor
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	//struct cudaTextureDesc* texDesc = (struct cudaTextureDesc*) calloc(1, sizeof(struct cudaTextureDesc));
	texDesc.readMode = cudaReadModeElementType;

	// Create the Texture Object
	cudaTextureObject_t tex;
	//cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
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
	//heater_buffer[2 + 5*dim] = 0.5;
	//heater_buffer[6 + 7*dim] = 1.0;
	
	// Send data to the GPU
	cudaMemcpy(dev_heater_buffer, heater_buffer, buffer_size*sizeof(float), cudaMemcpyHostToDevice);

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

