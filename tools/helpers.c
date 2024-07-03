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

