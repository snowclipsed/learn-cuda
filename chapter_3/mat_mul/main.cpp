// #include <stdio.h>
#include "../../tools/helpers.c"
#include "../../tools/macros.h"
#include <stdlib.h>
#include "mat_mul.cuh"

__global__ void matmul(float *A, float *B, float *C, int A_height, int common, int B_width);
__global__ void mat_mul_row(float *A, float *B, float *C, int A_height, int common, int B_width);


// Main function
int main() {
  float *A_h, *B_h, *C_h_normal, *C_h_row;
  int A_dim[] = {8192, 8192};
  int B_dim[] = {8192, 8192};

  if (A_dim[1] != B_dim[0]) {
    fprintf(stderr, "Error: Matrix size incompatible. Matrices must be i x k @ "
                    "k x j in dimensions.\n");
    exit(EXIT_FAILURE);
  }

  int size_A = A_dim[0] * A_dim[1] * sizeof(float);
  int size_B = B_dim[0] * B_dim[1] * sizeof(float);
  int size_C = A_dim[0] * B_dim[1] * sizeof(float);

  A_h = create_matrix_random(A_dim[0], A_dim[1], 5.0f, 1.0f);
  B_h = create_matrix_random(B_dim[0], B_dim[1], 5.0f, 1.0f);
  C_h_normal = (float *)malloc(size_C);
  C_h_row = (float *)malloc(size_C);

  if (C_h_normal== NULL) {
    fprintf(stderr, "Memory allocation failed for C_h\n");
    exit(EXIT_FAILURE);
  }

  launchMatMulRow(A_h, B_h, C_h_row, A_dim, B_dim);
  launchMatMulCol(A_h, B_h, C_h_normal, A_dim, B_dim);

  // Verify results (optional)
  // After comparing arrays:
  if (!compare_arrays(C_h_normal, C_h_row, A_dim[0] * B_dim[1])) {
      fprintf(stderr, "CPU and GPU results differ!\n");
      // You might want to print some of the differing values here
  }

  // print_2d_array(C_h_normal, A_dim[0], B_dim[1]);
  // Free memory
  free(A_h);
  free(B_h);
  free(C_h_normal);
  free(C_h_row);
}
