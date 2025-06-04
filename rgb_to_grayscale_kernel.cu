// rgb_to_grayscale_kernel.cu
//
// Author: EngEryx
// Date: 03-06-2025
//
// Description:
// This file contains CUDA kernels for converting RGB images to grayscale.
// It includes:
// 1. A reference implementation.
// 2. An optimized implementation using float3 vectorized memory access.
// 3. An optimized implementation using shared memory (bonus).
//
// The grayscale conversion uses the standard luminance formula:
// Y = 0.2989 * R + 0.5870 * G + 0.1140 * B

#include <cuda_runtime.h> // Includes vector_types.h (for float3) and math.h (for fmaf)

// Luminance coefficients
#define R_COEFF 0.2989f
#define G_COEFF 0.5870f
#define B_COEFF 0.1140f

/**
 * @brief Optimized RGB to Grayscale conversion kernel using float3 for vectorized memory access.
 *
 * Each thread processes one pixel. RGB values are loaded as a float3 vector,
 * potentially leading to a single 12-byte memory transaction per thread,
 * which promotes coalesced memory access when threads in a warp access contiguous pixels.
 * The __restrict__ keyword is used to inform the compiler that input and output pointers
 * do not alias, allowing for more aggressive optimizations.
 * Fused Multiply-Add (fmaf) instructions are used for the weighted sum, which can
 * improve performance and precision.
 *
 * @param input_raw Pointer to the input RGB image data (H, W, 3) stored linearly.
 * @param output Pointer to the output grayscale image data (H, W) stored linearly.
 * @param width The width of the image.
 * @param height The height of the image.
 */
__global__ void rgb_to_grayscale_float3(const float *__restrict__ input_raw,
                                      float *__restrict__ output,
                                      int width, int height) {
    // Calculate global 2D thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: ensure the thread is within image dimensions
    if (x < width && y < height) {
        // Calculate 1D index for the current pixel
        int pixel_idx = y * width + x;

        // Cast input pointer to float3 pointer for vectorized loads.
        // Assumes input is (R,G,B, R,G,B, ...) and aligned.
        const float3* input_float3_ptr = (const float3*)input_raw;

        // Load RGB values using a single float3 load instruction
        float3 rgb_val = input_float3_ptr[pixel_idx];

        // Luminance formula using fmaf (fused multiply-add)
        // Y = R_COEFF*R + G_COEFF*G + B_COEFF*B
        float gray_val = R_COEFF * rgb_val.x;            // R component
        gray_val = fmaf(G_COEFF, rgb_val.y, gray_val); // Add G component term
        gray_val = fmaf(B_COEFF, rgb_val.z, gray_val); // Add B component term

        output[pixel_idx] = gray_val;
    }
}

// --- Bonus Kernel: Shared Memory Optimization ---
#define TILE_DIM_X 16 // Tile dimension for shared memory (must match blockDim.x)
#define TILE_DIM_Y 16 // Tile dimension for shared memory (must match blockDim.y)

/**
 * @brief Optimized RGB to Grayscale conversion kernel using shared memory.
 *
 * Each block of threads cooperatively loads a tile of the input RGB image
 * into shared memory. Then, each thread processes its pixel from the faster
 * shared memory. This can improve performance if global memory access is a
 * bottleneck or if there's data reuse within the tile (though less applicable here).
 *
 * @param input_raw Pointer to the input RGB image data.
 * @param output Pointer to the output grayscale image data.
 * @param width The width of the image.
 * @param height The height of the image.
 */
__global__ void rgb_to_grayscale_shared_mem(const float *__restrict__ input_raw,
                                          float *__restrict__ output,
                                          int width, int height) {
    // Declare shared memory for a tile of float3 RGB values
    // Dimensions are TILE_DIM_Y x TILE_DIM_X
    __shared__ float3 tile_s[TILE_DIM_Y][TILE_DIM_X];

    // Thread indices within the block (local coordinates)
    int tx = threadIdx.x; // Thread's x-index within the block
    int ty = threadIdx.y; // Thread's y-index within the block

    // Global coordinates of the pixel this thread will initially load
    int global_load_x = blockIdx.x * TILE_DIM_X + tx;
    int global_load_y = blockIdx.y * TILE_DIM_Y + ty;

    // Cast input pointer to float3 pointer for vectorized loads.
    const float3* input_float3_ptr = (const float3*)input_raw;

    // Load data into shared memory from global memory
    // Each thread loads one pixel's RGB data (as float3) into its corresponding shared memory location.
    // Boundary check for reads from global memory.
    if (global_load_x < width && global_load_y < height) {
        tile_s[ty][tx] = input_float3_ptr[global_load_y * width + global_load_x];
    }

    // Synchronize threads in the block to ensure all data is loaded into shared memory
    // before any thread proceeds to compute.
    __syncthreads();

    // Global coordinates of the pixel this thread will process and write
    // (These are the same as global_load_x/y in this kernel, but kept separate for clarity
    // in more complex tiling schemes where load and process indices might differ)
    int global_process_x = blockIdx.x * TILE_DIM_X + tx;
    int global_process_y = blockIdx.y * TILE_DIM_Y + ty;

    // Process data from shared memory and write to global output
    // Boundary check for writes to global memory.
    if (global_process_x < width && global_process_y < height) {
        float3 rgb_val = tile_s[ty][tx]; // Read from shared memory

        // Luminance formula using fmaf
        float gray_val = R_COEFF * rgb_val.x;
        gray_val = fmaf(G_COEFF, rgb_val.y, gray_val);
        gray_val = fmaf(B_COEFF, rgb_val.z, gray_val);

        output[global_process_y * width + global_process_x] = gray_val;
    }
}

/**
 * @brief Reference RGB to Grayscale conversion kernel.
 *
 * Each thread processes one pixel. RGB values are loaded as three separate float values.
 * This is a basic, unoptimized version for comparison.
 *
 * @param input Pointer to the input RGB image data.
 * @param output Pointer to the output grayscale image data.
 * @param width The width of the image.
 * @param height The height of the image.
 */
__global__ void rgb_to_grayscale_reference(const float *__restrict__ input,
                                         float *__restrict__ output,
                                         int width, int height) {
    // Calculate global 2D thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x < width && y < height) {
        // Calculate 1D index for the current pixel in the output grayscale image
        int gray_idx = y * width + x;
        // Calculate 1D index for the start of RGB components of the current pixel
        // Each pixel has 3 float components (R, G, B).
        int rgb_idx = gray_idx * 3;

        // Load R, G, B components separately
        float r = input[rgb_idx + 0];
        float g = input[rgb_idx + 1];
        float b = input[rgb_idx + 2];

        // Luminance formula
        output[gray_idx] = R_COEFF * r + G_COEFF * g + B_COEFF * b;
    }
}