#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
// Include the stb_image library
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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


// concerns loading, saving and freeing images from memory

// Function to load an image as an array of unsigned chars
unsigned char* load_image(const char* filename, int* width, int* height, int* channels) {
    unsigned char* img = stbi_load(filename, width, height, channels, 0);
    
    if (img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", *width, *height, *channels);
    
    return img;
}

// Function to free the memory allocated for the image
void free_image(unsigned char* img) {
    stbi_image_free(img);
}


// Function to save an image
int save_image(const char* filename, int width, int height, int channels, unsigned char* img) {
    int result = 0;
    
    // Get the file extension
    const char* dot = strrchr(filename, '.');
    if (!dot || dot == filename) {
        printf("Error: Invalid filename or missing extension\n");
        return 0;
    }
    
    // Convert to lowercase for easier comparison
    char ext[5];
    strncpy(ext, dot + 1, 4);
    ext[4] = '\0';
    for (int i = 0; ext[i]; i++) {
        ext[i] = tolower(ext[i]);
    }
    
    // Save the image based on the file extension
    if (strcmp(ext, "png") == 0) {
        result = stbi_write_png(filename, width, height, channels, img, width * channels);
    } else if (strcmp(ext, "jpg") == 0 || strcmp(ext, "jpeg") == 0) {
        result = stbi_write_jpg(filename, width, height, channels, img, 100); // 100 is quality (0-100)
    } else if (strcmp(ext, "bmp") == 0) {
        result = stbi_write_bmp(filename, width, height, channels, img);
    } else {
        printf("Error: Unsupported file format. Supported formats are PNG, JPG, and BMP.\n");
        return 0;
    }
    
    if (result == 0) {
        printf("Error: Failed to save the image\n");
        return 0;
    }
    
    printf("Image saved successfully as %s\n", filename);
    return 1;
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



// int main() {
//     int width, height, channels;
//     unsigned char* img = load_image("path_to_your_image.jpg", &width, &height, &channels);
    
//     // Now you can use the img array for further processing
//     // For example, let's print the first pixel's values:
//     if (channels >= 3) {
//         printf("First pixel - R: %d, G: %d, B: %d\n", img[0], img[1], img[2]);
//     }
    
//     // Don't forget to free the memory when you're done
//     free_image(img);
    
//     return 0;
// }