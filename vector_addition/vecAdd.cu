#include<stdlib.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include "tools/helpers.c"
#include "tools/macros.h"

__global__ void vecAddKernel(float* a, float* b, float* c, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n){
        c[i] = a[i] + b[i];
    }
}


float* vecAdd(float* a_h, float* b_h, float* c_h, int n){

    float* a_d, *b_d, *c_d;
    int size = n*sizeof(int);

    // allocating memory in GPU device
    CHECK_CUDA_ERROR(cudaMalloc((void**)&a_d, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&b_d, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&c_d, size));

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // vector addition
    vecAddKernel<<<ceil(n/256.0), 256>>>(a_d, b_d, c_d, n);


    //copying c_d back to host mem

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    // freeing the allocated memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return c_h;
}

int main(){

    float *a_h, *b_h, *c_h;
    int n = 10000000;

    a_h = create_array_ones(n);
    b_h = create_array_ones(n);
    c_h = (float*)malloc(n * sizeof(float));

    c_h = vecAdd(a_h, b_h, c_h, n);
    
    for (size_t i = 0; i < n; i++) {
        printf("%f\n", c_h[i]);
    }

    free(a_h);
    free(b_h);
    free(c_h);
}