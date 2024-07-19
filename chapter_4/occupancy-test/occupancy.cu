#include <math.h>
#include <stdlib.h>
#include "mat_mul.cuh"
#include "../tools/macros.h"

/**
 * @file occupancy.cu
 * @brief CUDA kernel for matrix multiplication.
 *
 * This file contains the CUDA kernel function `matmul` which performs matrix multiplication
 * of matrices A and B and stores the result in matrix C. The kernel is designed to be launched
 * on a CUDA device using a grid of thread blocks.
 */

/**
 * @brief CUDA kernel for matrix multiplication.
 *
 * This kernel function performs matrix multiplication of matrices A and B and stores the result
 * in matrix C. Each thread in the grid is responsible for computing a single element of the
 * resulting matrix C.
 *
 * @param A         Pointer to the input matrix A.
 * @param B         Pointer to the input matrix B.
 * @param C         Pointer to the output matrix C.
 * @param A_height  The number of rows in matrix A.
 * @param common    The number of columns in matrix A and rows in matrix B.
 * @param B_width   The number of columns in matrix B.
 */
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


/**
 * @brief Launches the matrix multiplication kernel.
 *
 * This function allocates memory on the device, copies the input matrices A and B to the device,
 * launches the matrix multiplication kernel, copies the result matrix C back to the host, and
 * frees the device memory.
 *
 * @param A_h       Pointer to the input matrix A on the host.
 * @param B_h       Pointer to the input matrix B on the host.
 * @param C_h       Pointer to the output matrix C on the host.
 * @param A_dim     The dimensions of matrix A (rows, columns).
 * @param B_dim     The dimensions of matrix B (rows, columns).
 */
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