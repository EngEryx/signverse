# test_script.py
#
# Author: EngEryx
# Date: 03-06-2025
#
# Description:
# This script tests and benchmarks CUDA kernels for RGB to grayscale conversion.
# It loads kernels from 'rgb_to_grayscale_kernel.cu',, runs them on randomly
# generated image data, verifies correctness against a NumPy-based calculation,
# and measures execution time.
#
# Usage:
# python test_script.py

import numpy as np
import pycuda.autoinit # Initializes CUDA context automatically
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import os

# --- Configuration ---
KERNEL_FILE = "rgb_to_grayscale_kernel.cu"
IMAGE_SIZES = [(512, 512), (1024, 1024), (2048, 2048)] # (Width, Height)
NUM_BENCHMARK_RUNS = 100 # Number of kernel executions for averaging time
VERBOSITY_PTXAS = False # Set to True to get verbose PTXAS output during compilation

# --- Helper Functions ---
def load_and_compile_kernels(kernel_file_path, verbose_ptxas=False):
    """Loads CUDA source from a file and compiles it."""
    if not os.path.exists(kernel_file_path):
        print(f"Error: Kernel file '{kernel_file_path}' not found.")
        exit(1)
    with open(kernel_file_path, "r") as f:
        cuda_source_code = f.read()

    compile_options = []
    if verbose_ptxas:
        compile_options.append('-Xptxas=-v') # Shows register usage, etc.

    print(f"Compiling CUDA kernels from '{kernel_file_path}'...")
    try:
        module = SourceModule(cuda_source_code, options=compile_options)
        print("CUDA Kernels Compiled Successfully!")
        return {
            "ref": module.get_function("rgb_to_grayscale_reference"),
            "float3": module.get_function("rgb_to_grayscale_float3"),
            "shared_mem": module.get_function("rgb_to_grayscale_shared_mem")
        }
    except cuda.CompileError as e:
        print("CUDA Compilation Failed!")
        print(e)
        exit(1)

def test_kernel(kernel_func, kernel_name, width, height, input_rgb_np):
    """Tests a given CUDA kernel for correctness and performance."""
    print(f"\n--- Testing {kernel_name} for {width}x{height} image ---")

    output_gray_np = np.zeros((height, width), dtype=np.float32)

    # Allocate GPU memory
    input_gpu = cuda.mem_alloc(input_rgb_np.nbytes)
    output_gpu = cuda.mem_alloc(output_gray_np.nbytes)

    # Copy input data from host (CPU) to device (GPU)
    cuda.memcpy_htod(input_gpu, input_rgb_np)

    # Define CUDA kernel launch parameters
    # For the shared memory kernel, block dimensions must match TILE_DIM_X/Y in .cu
    if kernel_name == "Optimized Kernel (Shared Memory)":
        # These must match #define TILE_DIM_X/Y in the .cu file
        block_dim_x, block_dim_y = 16, 16
    else:
        block_dim_x, block_dim_y = 16, 16 # Common default (256 threads/block)

    block_dims = (block_dim_x, block_dim_y, 1)
    grid_dims = (
        (width + block_dims[0] - 1) // block_dims[0],
        (height + block_dims[1] - 1) // block_dims[1],
        1
    )

    # Warm-up run (important for stable performance measurements)
    kernel_func(input_gpu, output_gpu, np.int32(width), np.int32(height),
                block=block_dims, grid=grid_dims)
    cuda.Context.synchronize() # Ensure warm-up is complete

    # Measure execution time using CUDA events for precision
    start_event = cuda.Event()
    end_event = cuda.Event()

    start_event.record()
    for _ in range(NUM_BENCHMARK_RUNS):
        kernel_func(input_gpu, output_gpu, np.int32(width), np.int32(height),
                    block=block_dims, grid=grid_dims)
    end_event.record()

    end_event.synchronize() # Wait for all kernel executions to complete
    elapsed_ms_total = start_event.time_till(end_event)
    elapsed_ms_per_run = elapsed_ms_total / NUM_BENCHMARK_RUNS

    print(f"Execution time: {elapsed_ms_per_run:.4f} ms (averaged over {NUM_BENCHMARK_RUNS} runs)")

    # Copy output data from device (GPU) to host (CPU)
    cuda.memcpy_dtoh(output_gray_np, output_gpu)

    # Verify correctness against NumPy calculation
    # Luminance coefficients must match those in the CUDA kernel
    R_COEFF, G_COEFF, B_COEFF = 0.2989, 0.5870, 0.1140
    ref_gray_np = (R_COEFF * input_rgb_np[:, :, 0] +
                   G_COEFF * input_rgb_np[:, :, 1] +
                   B_COEFF * input_rgb_np[:, :, 2])

    try:
        # Using a slightly more tolerant decimal for floating point comparisons
        np.testing.assert_almost_equal(output_gray_np, ref_gray_np, decimal=4)
        print("Correctness: PASS")
    except AssertionError as e:
        print("Correctness: FAIL")
        print(e)
        # For debugging, print a small diff:
        # diff = np.abs(output_gray_np - ref_gray_np)
        # print(f"Max difference: {np.max(diff)}")
        # print("Sample differing values (GPU vs CPU):")
        # for i in range(min(5, height)):
        #     for j in range(min(5, width)):
        #         if not np.isclose(output_gray_np[i,j], ref_gray_np[i,j], atol=1e-4):
        #             print(f"Pixel ({i},{j}): GPU={output_gray_np[i,j]:.6f}, CPU={ref_gray_np[i,j]:.6f}")
        #             break
        #     else:
        #         continue
        #     break


    # Free GPU memory
    input_gpu.free()
    output_gpu.free()

    return elapsed_ms_per_run

# --- Main Execution ---
if __name__ == "__main__":
    kernels = load_and_compile_kernels(KERNEL_FILE, verbose_ptxas=VERBOSITY_PTXAS)
    kernel_ref = kernels["ref"]
    kernel_float3 = kernels["float3"]
    kernel_shared_mem = kernels["shared_mem"]

    results_summary = {} # To store {kernel_name: {size_str: time_ms}}

    print("\nStarting benchmarks for RGB to Grayscale conversion kernels...")

    for width, height in IMAGE_SIZES:
        if not (width % 2 == 0 and height % 2 == 0 and width == height):
            print(f"Skipping size {width}x{height} as it does not meet constraints (square, even dimensions).")
            continue

        print(f"\n======================================================")
        print(f"Processing Image Size: {width}x{height}")
        print(f"======================================================")

        # Generate random input image data on CPU
        # Values are float32 in range [0, 1]
        input_rgb_np = np.random.rand(height, width, 3).astype(np.float32)

        kernels_to_test = [
            (kernel_ref, "Reference Kernel"),
            (kernel_float3, "Optimized Kernel (float3)"),
            (kernel_shared_mem, "Optimized Kernel (Shared Memory)")
        ]

        size_str = f"{width}x{height}"
        for kernel_func, kernel_name in kernels_to_test:
            if kernel_name not in results_summary:
                results_summary[kernel_name] = {}

            exec_time = test_kernel(kernel_func, kernel_name, width, height, input_rgb_np)
            results_summary[kernel_name][size_str] = exec_time

    # Print summary table
    print("\n\n--- Performance Summary (ms) ---")
    header_cols = [f"{w}x{h}" for w,h in IMAGE_SIZES if w%2==0 and h%2==0 and w==h]
    # Dynamically determine column width based on longest size string or 'Kernel Name'
    max_col_name_len = max(len("Kernel Name"), max(len(col) for col in header_cols) if header_cols else 0)
    name_col_width = 30 # Fixed width for kernel name column
    data_col_width = max(max_col_name_len, 12) # Data columns width

    header = f"| {'Kernel Name':<{name_col_width}} |" + "".join([f" {col.center(data_col_width)} |" for col in header_cols])
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    # Ensure a consistent order for printing results
    kernel_order = ["Reference Kernel", "Optimized Kernel (float3)", "Optimized Kernel (Shared Memory)"]
    for kernel_name_key in kernel_order:
        if kernel_name_key in results_summary:
            row = f"| {kernel_name_key:<{name_col_width}} |"
            for size_key in header_cols:
                time_val = results_summary[kernel_name_key].get(size_key, "N/A")
                if isinstance(time_val, float):
                    # Format to 4 decimal places and add 'ms'
                    row += f" {f'{time_val:.4f}ms'.rjust(data_col_width)} |"
                else:
                    row += f" {'N/A'.center(data_col_width)} |"
            print(row)
    print(separator)

    print("\nBenchmark complete.")