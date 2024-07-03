#include<stdlib.h>
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

__global__ void vecAddKernel(float* a, float* b, float* c, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n){
        c[i] = a[i] + b[i];
    }
}

int* create_random_array(size_t n) {
    // Allocate memory for the array
    int *array = (int*)malloc(n * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Populate the array with random integers
    for (size_t i = 0; i < n; i++) {
        array[i] = rand();
    }

    return array;
}

float* create_array_ones(size_t n){

    float* array = (float*)malloc(n * sizeof(int));

    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Populate the array with 1
    for (size_t i = 0; i < n; i++) {
        array[i] = 1;
    }

    return array;

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