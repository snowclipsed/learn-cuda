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



__global__ void apply_func_kernel(float* A, float* result, int rows, int cols, int choice) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j<rows && i<cols) {
        float x = A[j*cols + i];
        switch(choice) {
            case 1: result[j*cols + i] = device_sin(x); break;
            case 2: result[j*cols + i] = device_tan(x); break;
            case 3: result[j*cols + i] = device_log(x); break;
            default: result[j*cols + i] = x; // Default case to avoid undefined behavior
        }
    }
}


float* apply_func(float* A_h, float* result_h, int rows, int cols, int choice, float threads=32.0){
    
    float *A_d, *result_d;
    int size = rows * cols * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&A_d, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&result_d, size));

    CHECK_CUDA_ERROR(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));

    dim3 dimGrid (ceil(cols/threads), ceil(rows/threads), 1);
    dim3 dimBlock (threads, threads, 1); 
    
    apply_func_kernel<<<dimGrid, dimBlock>>>(A_d, result_d, rows, cols, choice);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(result_h, result_d, size, cudaMemcpyDeviceToHost));

    cudaFree(A_d);
    cudaFree(result_d);
    return result_h;
}


int main(){

    int rows, cols;
    rows = 10;
    cols = 10;
    float max, min;
    max = 1.0f;
    min = 0.1f;

    float *A_h, *result_h;
    A_h = create_matrix_random(rows, cols, max, min);
    // A_h = create_matrix_value(rows, cols);
    result_h = (float*)malloc(rows*cols*sizeof(float));

    apply_func(A_h, result_h, rows, cols, 2);

    print_2d_array(result_h, rows, cols);
    free(A_h);
    free(result_h);

}