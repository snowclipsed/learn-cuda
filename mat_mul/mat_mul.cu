#include <stdio.h>
#include <stdlib.h>
#include "../tools/helpers.c"
#include "../tools/macros.h"
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

// Simple matmul kernel
__global__ void matmul(float* A, float* B, float* C, int A_height, int common, int B_width) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float C_value = 0;
    if (row < A_height && col < B_width) {
        C_value = 0; // Use float instead of int
        for (int i = 0; i < common; i++) {
            C_value += A[row * common + i] * B[i * B_width + col];
        }
        C[row * B_width + col] = C_value;
    }
}

__host__ void matmulCPU(float* A, float* B, float* C, int A_height, int common, int B_width){
    
    for(int row = 0; row < A_height; row++){
        for(int col = 0; col < B_width; col++){
            C[row * B_width + col] = 0;
            for(int i = 0; i < common; i++){
                C[row* B_width + col] += A[row * common + i] * B[i * B_width + col];
            }
        }
    }
}

// Main function
int main() {
    float* A_h, *B_h, *A_d, *B_d, *C_h, *C_d;
    //float* C_cpu;
    int A_dim[] = {8192, 8192};
    int B_dim[] = {8192, 8192};

    if (A_dim[1] != B_dim[0]) {
        fprintf(stderr, "Error: Matrix size incompatible. Matrices must be i x k @ k x j in dimensions.\n");
        exit(EXIT_FAILURE);
    }

    int size_A = A_dim[0] * A_dim[1] * sizeof(float);
    int size_B = B_dim[0] * B_dim[1] * sizeof(float);
    int size_C = A_dim[0] * B_dim[1] * sizeof(float);

    A_h = create_matrix_random(A_dim[0], A_dim[1], 5.0f, 1.0f);
    B_h = create_matrix_random(B_dim[0], B_dim[1], 5.0f, 1.0f);
    // C_cpu = (float*)malloc(size_C);
    
    // if (C_cpu == NULL) {
    //     fprintf(stderr, "Memory allocation failed for C_cpu\n");
    //     exit(EXIT_FAILURE);
    // }

    C_h = (float*)malloc(size_C); 
    if (C_h == NULL) {
        fprintf(stderr, "Memory allocation failed for C_h\n");
        exit(EXIT_FAILURE);
    }

    CHECK_CUDA_ERROR(cudaMalloc((void**)&A_d, size_A));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&B_d, size_B)); // Fixed allocation for B_d
    CHECK_CUDA_ERROR(cudaMalloc((void**)&C_d, size_C));

    CHECK_CUDA_ERROR(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));

    dim3 blockDim(32, 32, 1);
    dim3 gridDim((B_dim[1] + blockDim.x - 1) / blockDim.x, (A_dim[0] + blockDim.y - 1) / blockDim.y, 1);

    matmul<<<gridDim, blockDim>>>(A_d, B_d, C_d, A_dim[0], A_dim[1], B_dim[1]);

    // cudaDeviceSynchronize(); // Ensure the kernel has completed
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    CHECK_CUDA_ERROR(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));

    // // Verify results (optional)
    // matmulCPU(A_h, B_h, C_cpu, A_dim[0], A_dim[1], B_dim[1]);

    // // After comparing arrays:
    // if (!compare_arrays(C_cpu, C_h, A_dim[0] * B_dim[1])) {
    //     fprintf(stderr, "CPU and GPU results differ!\n");
    //     // You might want to print some of the differing values here
    // }
    // Free memory
    free(A_h);
    free(B_h);
    free(C_h);
    // free(C_cpu);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
