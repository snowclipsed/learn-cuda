#include <cuda_runtime.h>
#include <stdio.h>


int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);
  printf("Number of CUDA devices: %d\n", device_count);

  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, i);
    printf("Device %d: %s\n", i, device_prop.name);
    printf("Max Block Dimensions: %d x %d x %d\n",
           device_prop.maxThreadsDim[0],
           device_prop.maxThreadsDim[1],
           device_prop.maxThreadsDim[2]);
    printf("Max Grid Dimensions: %d x %d x %d\n",
           device_prop.maxGridSize[0],
           device_prop.maxGridSize[1],
           device_prop.maxGridSize[2]);
    printf("Warp Size: %d\n", device_prop.warpSize);
    printf("Clock rate: %d\n", device_prop.clockRate);
    printf("Global memory: %ld\n", device_prop.totalGlobalMem);
    printf("Max Threads/Block: %d\n", device_prop.maxThreadsPerBlock);
    printf("Number of multiprocessors:  %d\n", device_prop.multiProcessorCount);
    printf("Registers/Block: %d\n", device_prop.regsPerBlock);
    printf("Number of Threads/multiprocessor: %d\n", device_prop.maxThreadsPerMultiProcessor);    
  }
    

  return 0;
}
