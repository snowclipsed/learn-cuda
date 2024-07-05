#include <stdio.h>
#include <stdlib.h>
#include "../tools/helpers.c"
#include "../tools/macros.h"
#include <math.h>
#include <cuda_runtime.h>

__global__ void colortograyscale_kernel(unsigned char * img_in, unsigned char* img_out, int width, int height, int channels){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(col < width && row < height){
        int grayscaleindex = row*width + col;
        int rgbindex = grayscaleindex * channels;

        unsigned char r = img_in[rgbindex];
        unsigned char g = img_in[rgbindex + 1];
        unsigned char b = img_in[rgbindex + 2];

        img_out[grayscaleindex] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}


int main(){
    int width, height, channels;
     
    unsigned char* img_in_h = load_image("flower.jpg", &width, &height, &channels);
    unsigned char* img_out_h = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char* img_in_d, *img_out_d;

    int size_in = channels * width * height * sizeof(unsigned char);
    int size_out = width * height * sizeof(unsigned char);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&img_in_d, size_in));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&img_out_d, size_out));

    CHECK_CUDA_ERROR(cudaMemcpy(img_in_d, img_in_h, size_in, cudaMemcpyHostToDevice));

    dim3 blockDim(32, 32, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    colortograyscale_kernel<<<gridDim, blockDim>>>(img_in_d, img_out_d, width, height, channels);

    CHECK_CUDA_ERROR(cudaMemcpy(img_out_h, img_out_d, size_out, cudaMemcpyDeviceToHost));

    if (!save_image("grayscale_output.png", width, height, 1, img_out_h)) {
        printf("Failed to save the grayscale image\n");
    }


    cudaFree(img_in_d);
    cudaFree(img_out_d);
    free_image(img_in_h);
    free_image(img_out_h);
}