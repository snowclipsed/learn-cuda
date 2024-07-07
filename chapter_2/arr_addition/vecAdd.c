#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vecAdd(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
	C[i] = A[i] + B[i];
    }
}

int main(){
	
	clock_t start = clock();
	int n = 1000000;
	float *A = (float*)malloc(n*sizeof(float));
	float *B = (float*)malloc(n*sizeof(float));
	float *C = (float*)malloc(n*sizeof(float));
	
	for(int i = 0; i < n; i++){
		A[i] = i;
		B[i] = i;
	}
	
	vecAdd(A, B, C, n);
	
	clock_t end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	
	printf("Time: %f\n", time);
	free(A);
	free(B);
	free(C);
	
	return 0;

}
