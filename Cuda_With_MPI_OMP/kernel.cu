#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "helper.h"

#define MAX_THREADS 1024

///---Function declerations---///
cudaError_t calculateWithGPU(int *arr, int *results, unsigned int size);
void checkError(cudaError_t cudaStatus, int *dev_a, int *dev_results, const char* errorMessage);
void freeCudaMemory(int *dev_a, int *dev_results);
__device__ double f_GPU(int i);

__global__ void calcKernel(const int* a, int* res, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; //calculate index of each element in array

	if (f_GPU(a[index]) > 0)
		res[index] = 1;
	else
		res[index] = 0;
}

cudaError_t calculateWithGPU(int *arr, int *results, unsigned int size)
{
	char errorBuffer[100];
    int *dev_a = 0;
	int *dev_results = 0;
	int extra;
	int numOfBlocks, numOfThreads;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
	checkError(cudaStatus, dev_a, dev_results, 
		"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	checkError(cudaStatus, dev_a, dev_results, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_results, size * sizeof(int));
	checkError(cudaStatus, dev_a, dev_results, "cudaMalloc failed!");

    // Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, arr, size * sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus, dev_a, dev_results, "cudaMemcpy failed!");

    // Calculate the number of blocks and threads needed.
	extra = size % MAX_THREADS!=0 ? 1 : 0;
	numOfBlocks = (size/MAX_THREADS+extra);
	numOfThreads = MAX_THREADS>size ? size : MAX_THREADS;
	// Launch a kernel on the GPU with one thread for each element.
    calcKernel<<<numOfBlocks, numOfThreads >>>(dev_a, dev_results, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
	sprintf(errorBuffer, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	checkError(cudaStatus, dev_a, dev_results, errorBuffer);
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
	sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	checkError(cudaStatus, dev_a, dev_results, errorBuffer);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(results, dev_results, size * sizeof(int), cudaMemcpyDeviceToHost);
	checkError(cudaStatus, dev_a, dev_results, "cudaMemcpy failed!");

	freeCudaMemory(dev_a, dev_results);

    return cudaStatus;
}

void freeCudaMemory(int *dev_a, int *dev_results)
{
	cudaFree(dev_a);
	cudaFree(dev_results);
}

void checkError(cudaError_t cudaStatus, int *dev_a, int *dev_results, const char* errorMessage)
{
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, errorMessage);
		fprintf(stderr, "\n");
		freeCudaMemory(dev_a, dev_results);
	}
}

__device__ double f_GPU(int i) {
	int j;
	double value;
	double result = 0;

	for (j = 1; j < MASSIVE; j++) {
		value = (i + 1)*(j % 10);
		result += cos(value);
	}
	return cos(result);
}