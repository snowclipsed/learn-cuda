#include<cuda_runtime.h>
#include<stdlib.h>
#include "../../tools/helpers.c"

#define CHECK_CUDA_ERROR() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}  // NO backslash here


__global__ 
void vecAddKernel(float* A, float* B, float* C, int n){

    int i = threadIdx.x + blockIdx.x * blockDim.x; 

    if(i<n){
        C[i] = A[i] + B[i];
    }
}


void vecAdd(float* A_h, float* B_h, float* C_h, int n){
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;
    
    // After memory allocations
    cudaMalloc((void **) &A_d, size);
    CHECK_CUDA_ERROR();
    cudaMalloc((void **) &B_d, size);
    CHECK_CUDA_ERROR();
    cudaMalloc((void **) &C_d, size);
    CHECK_CUDA_ERROR();
    
    // After memory copies
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
    
    // After kernel launch
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vecAddKernel<<<blocks, threads>>>(A_d, B_d, C_d, n);
    CHECK_CUDA_ERROR();
    cudaDeviceSynchronize();  // Force kernel completion
    CHECK_CUDA_ERROR();
    
    // After copy back
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR();
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){

    float *a_h, *b_h, *c_h;
    int n = 100000000;

    a_h = create_array_ones(n);
    b_h = create_array_ones(n);
    c_h = (float*)malloc(n * sizeof(float));

    vecAdd(a_h, b_h, c_h, n);
    
    for (size_t i = 0; i < 10; i++) {
        printf("%f\n", c_h[i]);
    }

    free(a_h);
    free(b_h);
    free(c_h);
}