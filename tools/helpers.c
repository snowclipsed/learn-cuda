#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>


float* create_random_array(size_t n) {
    // Allocate memory for the array
    float *array = (float*)malloc(n * sizeof(float));
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
        array[i] = 1.0f;
    }

    return array;

}

float* create_matrix_random(int rows, int cols, float max, float min) {
    float* array = (float*)malloc(cols * rows * sizeof(float));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            float random_value = min + (float)rand() / RAND_MAX * (max - min);
            array[j * cols + i] = random_value; // Corrected indexing
        }
    }

    return array;
}

float* create_matrix_value(int rows, int cols, float value=1.0f){
    float* array = (float*)malloc(cols * rows * sizeof(float));
    if (array == NULL){
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for(int j = 0; j<rows; j++){
        for(int i=0; i<cols; i++){
            array[i * cols + j] = value;
        }
    }

    return array;
}


void print_2d_array( float* array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f\t", array[i * cols + j]);
        }
        printf("\n");
    }
}

int compare_arrays(float* arr1, float* arr2, int size) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < size; i++) {
        if (fabsf(arr1[i] - arr2[i]) > epsilon) {
            printf("Arrays differ at index %d: %f vs %f\n", i, arr1[i], arr2[i]);
            return 0; // Arrays are not identical
        }
    }
    printf("Arrays are identical within epsilon %f\n", epsilon);
    return 1; // Arrays are identical
}

// cuda kernel performance measurement ------


float measureKernelPerformance(void (*kernel)(void), int numIterations) {
    cudaEvent_t start, stop;
    float totalTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < numIterations; i++) {
        cudaEventRecord(start);
        
        // Call the kernel function
        kernel();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return totalTime / numIterations;
}

void func_timing(void (*func)()) {
    clock_t start = clock();
    func();
    clock_t end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.6f seconds\n", elapsed_time);
}
