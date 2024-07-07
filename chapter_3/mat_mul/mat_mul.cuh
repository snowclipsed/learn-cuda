#ifndef MAT_MUL_CUH
#define MAT_MUL_CUH

#ifdef __cplusplus
extern "C" {
#endif

void launchMatmul(float *A_h, float *B_h, float *C_h, int* A_dim, int* B_dim);
void launchMatMulRow(float *A_h, float *B_h, float *C_h, int* A_dim, int* B_dim);
void launchMatMulCol(float *A_h, float *B_h, float *C_h, int* A_dim, int* B_dim);
void launchMatmulCPU(float *A, float *B, float *C, int A_height, int common, int B_width);

#ifdef __cplusplus
}
#endif

#endif // MAT_MUL_CUH
