#include<cuda_runtime.h>
#include<stdio.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code: %d) in file %s at line %d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}
