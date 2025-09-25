#include<stdlib.h>
#include<stdio.h>
#include <time.h>

void vecAdd(float *A, float* B, float* C, int n){
	int i = 0;

	for (int i = 0; i < n; i++)
	{
		C[i] = A[i] + B[i];
	}
}


void main(){
	clock_t start = clock();

	int n = 1000;
	float *A = (float*)malloc(n * sizeof(float));
	float *B = (float*)malloc(n * sizeof(float));
	float *C = (float*)malloc(n * sizeof(float));

	int i = 0;
	while(i < n){
		A[i] = i;
		B[i] = 2*i;
		i++;
	}

	vecAdd(A, B, C, n);
	clock_t end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	
	printf("Time: %f\n", time);

	free(A);
	free(B);
	free(C);
}