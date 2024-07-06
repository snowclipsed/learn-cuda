// this file concerns loading, saving and freeing images from memory


// Include the stb_image library
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
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

