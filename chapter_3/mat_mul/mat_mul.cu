#include <math.h>
#include <stdlib.h>
#include "mat_mul.cuh"
#include "../tools/macros.h"
#include "cuda_runtime.h"

// Simple matmul kernel
__global__ void matmul(float *A, float *B, float *C, int A_height, int common, int B_width) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float C_value = 0;

    if (row < A_height && col < B_width) {
        C_value = 0;
        for (int i = 0; i < common; i++) {
            C_value += A[row * common + i] * B[i * B_width + col];
        }
        C[row * B_width + col] = C_value;
    }
}

// __host__ void matmulCPU(float *A, float *B, float *C, int A_height, int common, int B_width) {
//     for (int row = 0; row < A_height; row++) {
//         for (int col = 0; col < B_width; col++) {
//             C[row * B_width + col] = 0;
//             for (int i = 0; i < common; i++) {
//                 C[row * B_width + col] += A[row * common + i] * B[i * B_width + col];
//             }
//         }
//     }
// }

__global__ void mat_mul_row(float *A, float *B, float *C, int A_height, int common, int B_width) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    float C_value = 0;
    if (row < A_height) {
        for (int col = 0; col < B_width; col++) {
            C_value = 0;
            for (int i = 0; i < common; i++) {
                C_value += A[row * common + i] * B[i * B_width + col];
            }
            C[row * B_width + col] = C_value;
        }
    }
}

__global__ void mat_mul_col(float *A, float *B, float *C, int A_height, int common, int B_width) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float C_value = 0;
    if (col < B_width) {
        for (int row = 0; row < A_height; row++) {
            C_value = 0;
            for (int i = 0; i < common; i++) {
                C_value += A[row * common + i] * B[i * B_width + col];
            }
            C[row * B_width + col] = C_value;
        }
    }
}


// Wrapper function for matmul
extern "C" void launchMatmul(float *A_h, float *B_h, float *C_h, int* A_dim, int* B_dim){
  float *A_d, *B_d, *C_d;
  int size_A = A_dim[0] * A_dim[1] * sizeof(float);
  int size_B = B_dim[0] * B_dim[1] * sizeof(float);
  int size_C = A_dim[0] * B_dim[1] * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc((void **)&A_d, size_A));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&B_d, size_B)); // Fixed allocation for B_d
  CHECK_CUDA_ERROR(cudaMalloc((void **)&C_d, size_C));

  CHECK_CUDA_ERROR(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));

  dim3 blockDim(32, 32, 1);
  dim3 gridDim((A_dim[1] + blockDim.x - 1) / blockDim.x,(A_dim[0] + blockDim.y - 1) / blockDim.y, 1);

  matmul<<<gridDim, blockDim>>>(A_d, B_d, C_d, A_dim[0], A_dim[1], B_dim[1]);

  cudaDeviceSynchronize(); // Ensure the kernel has completed
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
  }

  CHECK_CUDA_ERROR(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

// Wrapper function for mat_mul_row
extern "C" void launchMatMulRow(float *A_h, float *B_h, float *C_h, int* A_dim, int* B_dim) {
  float* A_d, *B_d, *C_d;
  int size_A = A_dim[0] * A_dim[1] * sizeof(float);
  int size_B = B_dim[0] * B_dim[1] * sizeof(float);
  int size_C = A_dim[0] * B_dim[1] * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc((void **)&A_d, size_A));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&B_d, size_B)); // Fixed allocation for B_d
  CHECK_CUDA_ERROR(cudaMalloc((void **)&C_d, size_C));

  CHECK_CUDA_ERROR(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));

  dim3 blockDim(256, 1, 1);
  dim3 gridDim((A_dim[0] + blockDim.x - 1) / blockDim.x, 1,1);

  mat_mul_row<<<gridDim, blockDim>>>(A_d, B_d, C_d, A_dim[0], A_dim[1], B_dim[1]);

  cudaDeviceSynchronize(); // Ensure the kernel has completed
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
  }

  CHECK_CUDA_ERROR(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

extern "C" void launchMatMulCol(float *A_h, float *B_h, float *C_h, int* A_dim, int* B_dim) {
  float* A_d, *B_d, *C_d;
  int size_A = A_dim[0] * A_dim[1] * sizeof(float);
  int size_B = B_dim[0] * B_dim[1] * sizeof(float);
  int size_C = A_dim[0] * B_dim[1] * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc((void **)&A_d, size_A));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&B_d, size_B)); // Fixed allocation for B_d
  CHECK_CUDA_ERROR(cudaMalloc((void **)&C_d, size_C));

  CHECK_CUDA_ERROR(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (B_dim[1] + blockSize - 1) / blockSize;

  mat_mul_col<<<numBlocks, blockSize>>>(A_d, B_d, C_d, A_dim[0], A_dim[1], B_dim[1]);

  cudaDeviceSynchronize(); // Ensure the kernel has completed
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
  }

  CHECK_CUDA_ERROR(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

// // Wrapper function for matmulCPU
// extern "C" void launchMatmulCPU(float *A, float *B, float *C, int A_height, int common, int B_width) {
//     matmulCPU(A, B, C, A_height, common, B_width);
// }