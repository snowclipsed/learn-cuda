#ifndef HELPERS_H
#define HELPERS_H

#include <stdlib.h>

// Function to create an array with random values
float* create_random_array(size_t n);

// Function to create an array filled with ones
float* create_array_ones(size_t n);

// Function to create a matrix with random values within a specified range
float* create_matrix_random(int rows, int cols, float max, float min);

// Function to create a matrix filled with a specific value
float* create_matrix_value(int rows, int cols, float value);

// Function to print a 2D array
void print_2d_array(float* array, int rows, int cols);

// Function to compare two arrays
int compare_arrays(float* arr1, float* arr2, int size);

// Function to measure the timing of a given function
void func_timing(void (*func)());

#endif // MY_ARRAY_UTILS_H
