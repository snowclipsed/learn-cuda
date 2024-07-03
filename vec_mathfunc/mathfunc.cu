#include <stdio.h>
#include <stdlib.h>
#include "../tools/helpers.c"
#include "../tools/macros.h"
#include <math.h>
#include <cuda_runtime.h>


// Device functions
__device__ float device_sin(float x) {
    return sinf(x);
}

__device__ float device_tan(float x) {
    return tanf(x);
}

__device__ float device_log(float x) {
    return logf(x);
}



__global__ void apply_func_kernel(float* A, float* result, int n, int choice) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = A[i];
        switch(choice) {
            case 1: result[i] = device_sin(x); break;
            case 2: result[i] = device_tan(x); break;
            case 3: result[i] = device_log(x); break;
            default: result[i] = x; // Default case to avoid undefined behavior
        }
    }
}


float* apply_func(float* A_h, float* result_h, int choice, int n, int threads=256 ){
    
    float *A_d, *result_d;
    int size = n * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&A_d, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&result_d, size));

    CHECK_CUDA_ERROR(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    
    int blocks = (n + threads - 1) / threads;

    switch(choice){
        case 1: apply_func_kernel<<<blocks, threads>>>(A_d, result_d, n, choice);
                break;
        case 2: apply_func_kernel<<<blocks, threads>>>(A_d, result_d, n, choice);
                break;
        case 3: apply_func_kernel<<<blocks, threads>>>(A_d, result_d, n, choice);
                break;
    }

    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(result_h, result_d, size, cudaMemcpyDeviceToHost));

    cudaFree(A_d);
    cudaFree(result_d);
    return result_h;
}


int main(){

    int n = 10000000;
    float* A_h, *result_h;
    
    A_h = create_array_ones(n);
    result_h = (float*)malloc(n* sizeof(float));

    apply_func(A_h, result_h, 2, n);
    
    for (size_t i = 0; i < n; i++) {
        printf("tan(%f) = %f\n", A_h[i], result_h[i]);
    }

    free(A_h);
    free(result_h);

}