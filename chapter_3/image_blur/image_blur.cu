#include "../../tools/helpers.h"
#include "../../tools/image_processing.c"
#include "../../tools/macros.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define BLUR_SIZE 5

__global__ void blur_kernel(unsigned char *img_in, unsigned char *img_out,
                            int width, int height, int channels) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int cur_row = 0;
  int cur_col = 0;
  int pixel_value = 0;
  int num_pixels = 0;

  if (row < height && col < width) {
    for (int ch = 0; ch < channels; ch++) {
      pixel_value = 0;
      num_pixels = 0;
      for (int blur_row = -BLUR_SIZE; blur_row <= BLUR_SIZE; blur_row++) {
        for (int blur_col = -BLUR_SIZE; blur_col <= BLUR_SIZE; blur_col++) {
          cur_row = row + blur_row;
          cur_col = col + blur_col;

          if (cur_row >= 0 && cur_row < height && cur_col >= 0 &&
              cur_col < width) {
            pixel_value += img_in[(cur_row * width + cur_col) * channels + ch];
            num_pixels++;
          }
        }
      }
      img_out[(row * width + col) * channels + ch] =
          (unsigned char)(pixel_value / num_pixels);
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

  blur_kernel<<<gridDim, blockDim>>>(img_in_d, img_out_d, width, height,
                                     channels);

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(
      cudaMemcpy(img_out_h, img_out_d, size, cudaMemcpyDeviceToHost));

  if (!save_image("blur_output.png", width, height, channels, img_out_h)) {
    printf("Failed to save the grayscale image\n");
  }

  cudaFree(img_in_d);
  cudaFree(img_out_d);
  free_image(img_in_h);
  free_image(img_out_h);
}
