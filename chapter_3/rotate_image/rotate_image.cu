#include<cuda_runtime.h>
#include "../../tools/helpers.h"
#include "../../tools/image_processing.c"
#include "../../tools/macros.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
// essentially in this kernel we would require to map the pixels from one region to the other
// to perform the shift effect
// of course let's assume we're only doing right angle increments
// original stride formula = r * n + c
// new formula map = c * m + (m-1-r)

__global__ void rotateImage(unsigned char * img_in, unsigned char *img_out, int width, int height, int channels){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row<height && col<width) {
        for(int ch=0; ch < channels; ch++){
            int source_idx = (row * width + col) * channels + ch;
            int dest_idx = (col*height + (height-1-row)) * channels + ch;
            img_out[dest_idx] = img_in[source_idx];
        }
    }
}


int main() {
  int width, height, channels;

  unsigned char *img_in_h =
      load_image("flower.jpg", &width, &height, &channels);
  int size = channels * width * height * sizeof(unsigned char);
  unsigned char *img_out_h = (unsigned char *)malloc(size);
  unsigned char *img_in_d, *img_out_d;

  CHECK_CUDA_ERROR(cudaMalloc((void **)&img_in_d, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&img_out_d, size));
  CHECK_CUDA_ERROR(
      cudaMemcpy(img_in_d, img_in_h, size, cudaMemcpyHostToDevice));

  dim3 blockDim(32, 32, 1);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y, 1);

  rotateImage<<<gridDim, blockDim>>>(img_in_d, img_out_d, width, height,
                                     channels);

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(
      cudaMemcpy(img_out_h, img_out_d, size, cudaMemcpyDeviceToHost));

  if (!save_image("rotate_output.png", height, width, channels, img_out_h)) {
    printf("Failed to save the grayscale image\n");
  }

  cudaFree(img_in_d);
  cudaFree(img_out_d);
  free_image(img_in_h);
  free_image(img_out_h);
}
