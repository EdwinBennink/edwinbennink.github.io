// Convolutions with different memory access patterns.
// Compile with: gcc -O2 -ftree-vectorize -mavx -std=c99 -o convolution .\convolution.c

// Copyright by Edwin Bennink

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))


// Convolves the image with the kernel. Uses the transposed order of iteration: {x, y, v, u}, in
// which (x, y) are the image coordinates and (u, v) are the kernel coordinates. This is a ordering,
// because of strided memory access.
// The kernel is truncated at the image boundaries.
void convolve_transposed(const float* image, float * output, const int width, const int height,
                         const float* kernel, const int radius) {
    // Iterate over the columns and rows in the image.
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            const int i = y * width + x;  // Index into the output image.
            
            // Initialize the output image with zeros.
            output[i] = 0.0f;
            
            // Iterate over the rows in the kernel. Truncate the iteration at the top and bottom
            // image border.
            for (int v = MAX(0, radius - y); v < MIN(2*radius + 1, radius + height - y); v++) {
                const int k = v * (2*radius + 1);  // Index into the kernel.
                const int j = (y + v - radius) * width + x - radius;  // Index into the input image.
                
                // Iterate over the columns in the kernel. Truncate the iteration at the left and
                // right image border.
                for (int u = MAX(0, radius - x); u < MIN(2*radius + 1, radius + width - x); u++) {
                    // Multiply image with kernel and add to the output.
                    output[i] += image[j + u] * kernel[k + u];
                }
            }
        }
    }
}


// Convolves the image with the kernel. Uses the generally accepted order of iteration: {y, x, v, u},
// in which (x, y) are the image coordinates and (u, v) are the kernel coordinates.
// The kernel is truncated at the image boundaries.
void convolve_default(const float* image, float * output, const int width, const int height,
                      const float* kernel, const int radius) {
    // Iterate over the rows and columns in the image.
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int i = y * width + x;  // Index into the output image.
            
            // Initialize the output image with zeros.
            output[i] = 0.0f;
            
            // Iterate over the rows in the kernel. Truncate the iteration at the top and bottom
            // image border.
            for (int v = MAX(0, radius - y); v < MIN(2*radius + 1, radius + height - y); v++) {
                const int k = v * (2*radius + 1);  // Index into the kernel.
                const int j = (y + v - radius) * width + x - radius;  // Index into the input image.
                
                // Iterate over the columns in the kernel. Truncate the iteration at the left and
                // right image border.
                for (int u = MAX(0, radius - x); u < MIN(2*radius + 1, radius + width - x); u++) {
                    // Multiply image with kernel and add to the output.
                    output[i] += image[j + u] * kernel[k + u];
                }
            }
        }
    }
}


// Convolves the image with the kernel. Uses the more contiguous order of iteration: {y, v, u, x},
// in which (x, y) are the image coordinates and (u, v) are the kernel coordinates. 
// The kernel is truncated at the image boundaries.
void convolve_contiguous(const float* image, float* output, const int width, const int height,
                     const float* kernel, const int radius) {
    // Iterate over the rows in the image.
    for (int y = 0; y < height; y++) {
        
        // Initialize the output image with zeros.
        for (int i = y * width; i < (y + 1) * width; i++) {
            output[i] = 0.0f;
        }
        
        // Iterate over the rows in the kernel. Truncate the iteration at the top and bottom image
        // border.
        for (int v = MAX(0, radius - y); v < MIN(2*radius + 1, radius + height - y); v++) {            
            // Iterate over the columns in the kernel.
            for (int u = 0; u < 2*radius + 1; u++) {
                const int k = v * (2*radius + 1) + u;  // Index into the kernel.
                int i = y * width;  // Index into the input image.
                int j = (y + v - radius) * width + u - radius;  // Index into the output image.
                
                // Iterate over the columns in the image. Note how this for-loop does contiguous 
                // reads/writes in output and image. It allows for efficient vectorization.
                // Truncate the iteration at the left and right image border.
                for (int x = MAX(0, radius - u); x < MIN(width, width - u + radius); x++) {
                    output[i + x] += image[j + x] * kernel[k];
                }
            }
        }
    }
}


int main() {
    const int width = 512;  // Image width
    const int height = 512;  // Image height
    const int radius = 1;  // Kernel radius
    const int kernel_size = (2*radius + 1) * (2*radius + 1);
    const int n_iterations = 1000;  // Number of convolution iterations
    clock_t start, end;
    float duration_ms;
    FILE *fp; 

    // Allocate new image and fill with random values in the range 0-1.
    float *image = malloc(height * width * sizeof(float *));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image[y * width + x] = (float)rand() / RAND_MAX;
        }
    }
    
    // Create a normalized random kernel with a specified radius.
    double sum = 0;
    float kernel[kernel_size];
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = (float)rand() / RAND_MAX;
        sum += kernel[i];
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }
    
    // Save the input image.
    fp = fopen("image.raw", "wb");
    fwrite(image, sizeof(float), width * height, fp);
    fclose(fp);
    
    // Allocate an output image.
    float *output = malloc(height * width * sizeof(float *));
    
    // Apply the convolution operation using the transposed ordering.
    start = clock();
    for (int i = 0; i < n_iterations; i++) {
        convolve_transposed(image, output, width, height, kernel, radius);
    }
    end = clock();
    duration_ms = 1e3f * ((float)(end - start)) / CLOCKS_PER_SEC / n_iterations;
    printf("Transposed: %.2f ms per iteration\n", duration_ms);
    
    // Write the output to file.
    fp = fopen("transposed.raw", "wb");
    fwrite(output, sizeof(float), width * height, fp);
    fclose(fp);
    
    // Apply the convolution operation using default ordering.
    start = clock();
    for (int i = 0; i < n_iterations; i++) {
        convolve_default(image, output, width, height, kernel, radius);
    }
    end = clock();
    duration_ms = 1e3f * ((float)(end - start)) / CLOCKS_PER_SEC / n_iterations;
    printf("Default: %.2f ms per iteration\n", duration_ms);
    
    // Write the output to file.
    fp = fopen("default.raw", "wb");
    fwrite(output, sizeof(float), width * height, fp);
    fclose(fp);
    
    // Apply the convolution operation using a more contiguous ordering.
    start = clock();
    for (int i = 0; i < n_iterations; i++) {
        convolve_contiguous(image, output, width, height, kernel, radius);
    }
    end = clock();
    duration_ms = 1e3f * ((float)(end - start)) / CLOCKS_PER_SEC / n_iterations;
    printf("Contiguous: %.2f ms per iteration\n", duration_ms);
    
    // Write the output to file.
    fp = fopen("contiguous.raw", "wb");
    fwrite(output, sizeof(float), width * height, fp);
    fclose(fp);

    // Cleanup.
    free(image);
    free(output);

    return 0;
}
