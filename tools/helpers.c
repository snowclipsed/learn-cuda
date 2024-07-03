#include <stdio.h>
#include <stdlib.h>

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

float* create_matrix_random(int rows, int cols, float max, float min){
    float* array = (float*)malloc(cols * rows * sizeof(float));
    if (array == NULL){
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for(int j = 0; j<rows; j++){
        for(int i=0; i<cols; i++){
            float random_value = min + ((float)rand() / RAND_MAX) * (max - min);
            array[i * cols + j] = random_value;
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

