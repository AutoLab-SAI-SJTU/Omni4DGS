/*
 * CUDA Debug Test Program
 * This program creates test data and calls CUDA rasterization functions
 * for debugging with cuda-gdb
 * Uses PyTorch for memory management to match the actual usage
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/config.h"
#include <torch/extension.h>

// Helper function for buffer resizing (similar to rasterize_points.cu)
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

int main(int argc, char* argv[]) {
    // Set a breakpoint here in cuda-gdb to skip static initialization issues
    // Use: (cuda-gdb) break main
    //      (cuda-gdb) run
    //      (cuda-gdb) continue  (to skip any initialization issues)
    
    std::cout << "=== CUDA Rasterization Debug Test ===" << std::endl;
    
    // Check CUDA availability first
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available!" << std::endl;
        return 1;
    }
    std::cout << "CUDA is available: " << torch::cuda::device_count() << " device(s)" << std::endl;
    
    // Test parameters
    const int P = 100;  // Number of Gaussians
    const int S = 8;    // Feature dimension
    const int D = 3;    // SH degree
    const int M = (D + 1) * (D + 1);  // SH coefficients
    const int W = 800;  // Image width
    const int H = 600;  // Image height
    const float scale_modifier = 1.0f;
    const bool prefiltered = false;
    const bool debug = true;
    
    std::cout << "Parameters:" << std::endl;
    std::cout << "  P (num Gaussians): " << P << std::endl;
    std::cout << "  S (feature dim): " << S << std::endl;
    std::cout << "  D (SH degree): " << D << std::endl;
    std::cout << "  Image size: " << W << "x" << H << std::endl;
    
    // Create PyTorch tensors on CPU first for initialization
    torch::Device device(torch::kCUDA);
    auto cpu_float_opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto cpu_int_opts = torch::TensorOptions().dtype(torch::kInt32);
    auto cpu_bool_opts = torch::TensorOptions().dtype(torch::kBool);
    
    // Create input tensors on CPU (will move to GPU after initialization)
    torch::Tensor means3D = torch::zeros({P, 3}, cpu_float_opts);
    torch::Tensor scales = torch::zeros({P, 3}, cpu_float_opts);
    torch::Tensor rotations = torch::zeros({P, 4}, cpu_float_opts);
    torch::Tensor opacities = torch::zeros({P}, cpu_float_opts);
    torch::Tensor shs = torch::zeros({P, M, 3}, cpu_float_opts);
    torch::Tensor colors_precomp = torch::zeros({P, NUM_CHANNELS}, cpu_float_opts);
    torch::Tensor features = torch::zeros({P, S}, cpu_float_opts);
    torch::Tensor mask = torch::ones({P}, cpu_bool_opts);  // All enabled
    torch::Tensor cov3D_precomp = torch::zeros({P, 6}, cpu_float_opts);
    torch::Tensor background = torch::zeros({NUM_CHANNELS}, cpu_float_opts);
    torch::Tensor viewmatrix = torch::zeros({4, 4}, cpu_float_opts);
    torch::Tensor campos = torch::zeros({3}, cpu_float_opts);
    
    // Initialize test data
    std::cout << "\nInitializing test data..." << std::endl;
    
    // Background color (black) - already zeros
    
    // Camera setup - simple perspective view
    // View matrix (identity-like, looking down -Z)
    auto viewmatrix_a = viewmatrix.accessor<float, 2>();
    viewmatrix_a[0][0] = 1.0f;
    viewmatrix_a[1][1] = 1.0f;
    viewmatrix_a[2][2] = 1.0f;
    viewmatrix_a[3][3] = 1.0f;
    viewmatrix_a[3][2] = -5.0f;  // Translation in Z
    
    // Camera position
    auto campos_a = campos.accessor<float, 1>();
    campos_a[0] = 0.0f;
    campos_a[1] = 0.0f;
    campos_a[2] = 5.0f;
    
    // Generate random Gaussians in a sphere
    srand(42);  // Fixed seed for reproducibility
    auto means3D_a = means3D.accessor<float, 2>();
    auto scales_a = scales.accessor<float, 2>();
    auto rotations_a = rotations.accessor<float, 2>();
    auto opacities_a = opacities.accessor<float, 1>();
    auto shs_a = shs.accessor<float, 3>();
    auto colors_precomp_a = colors_precomp.accessor<float, 2>();
    auto features_a = features.accessor<float, 2>();
    
    for (int i = 0; i < P; i++) {
        // 3D position - random in a sphere
        float theta = (float)rand() / RAND_MAX * 2.0f * M_PI;
        float phi = (float)rand() / RAND_MAX * M_PI;
        float r = (float)rand() / RAND_MAX * 2.0f;
        means3D_a[i][0] = r * sin(phi) * cos(theta);
        means3D_a[i][1] = r * sin(phi) * sin(theta);
        means3D_a[i][2] = r * cos(phi) - 2.0f;  // Shift back
        
        // Scales - small random values
        scales_a[i][0] = 0.01f + (float)rand() / RAND_MAX * 0.05f;
        scales_a[i][1] = 0.01f + (float)rand() / RAND_MAX * 0.05f;
        scales_a[i][2] = 0.01f + (float)rand() / RAND_MAX * 0.05f;
        
        // Rotations - quaternion (identity)
        rotations_a[i][0] = 1.0f;  // w
        rotations_a[i][1] = 0.0f;  // x
        rotations_a[i][2] = 0.0f;  // y
        rotations_a[i][3] = 0.0f;  // z
        
        // Opacity
        opacities_a[i] = 0.5f + (float)rand() / RAND_MAX * 0.5f;
        
        // SH coefficients - random colors
        for (int j = 0; j < M; j++) {
            shs_a[i][j][0] = (float)rand() / RAND_MAX;  // R
            shs_a[i][j][1] = (float)rand() / RAND_MAX;  // G
            shs_a[i][j][2] = (float)rand() / RAND_MAX;  // B
        }
        
        // Precomputed colors (if used)
        colors_precomp_a[i][0] = (float)rand() / RAND_MAX;
        colors_precomp_a[i][1] = (float)rand() / RAND_MAX;
        colors_precomp_a[i][2] = (float)rand() / RAND_MAX;
        
        // Features
        for (int j = 0; j < S; j++) {
            features_a[i][j] = (float)rand() / RAND_MAX;
        }
        
        // Mask - all enabled (already set to ones)
    }
    
    // Move all initialized tensors to GPU
    means3D = means3D.to(device);
    scales = scales.to(device);
    rotations = rotations.to(device);
    opacities = opacities.to(device);
    shs = shs.to(device);
    colors_precomp = colors_precomp.to(device);
    features = features.to(device);
    mask = mask.to(device);
    cov3D_precomp = cov3D_precomp.to(device);
    background = background.to(device);
    viewmatrix = viewmatrix.to(device);
    campos = campos.to(device);
    
    // Create output tensors (similar to rasterize_points.cu)
    std::cout << "Creating output tensors..." << std::endl;
    torch::Tensor out_contrib = torch::full({1, H, W}, 0, cpu_int_opts).to(device);
    torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0f, cpu_float_opts).to(device);
    torch::Tensor out_feature = torch::full({S, H, W}, 0.0f, cpu_float_opts).to(device);
    torch::Tensor out_depth = torch::full({H, W}, 0.0f, cpu_float_opts).to(device);
    torch::Tensor out_acc = torch::full({H, W}, 0.0f, cpu_float_opts).to(device);
    torch::Tensor radii = torch::full({P}, 0, cpu_int_opts).to(device);
    torch::Tensor psi = torch::full({P}, 0.0f, cpu_float_opts).to(device);
    torch::Tensor lat = torch::full({P}, 0.0f, cpu_float_opts).to(device);
    torch::Tensor lon = torch::full({P}, 0.0f, cpu_float_opts).to(device);
    
    // Buffer management for geometry, binning, and image buffers
    torch::TensorOptions buffer_opts(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, buffer_opts).to(device);
    torch::Tensor binningBuffer = torch::empty({0}, buffer_opts).to(device);
    torch::Tensor imgBuffer = torch::empty({0}, buffer_opts).to(device);
    
    // Lambda functions for buffer resizing
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
    
    std::cout << "\n=== Calling CUDA Rasterizer Forward ===" << std::endl;
    std::cout << "You can set breakpoints here for cuda-gdb debugging" << std::endl;
    
    // Call the CUDA rasterizer forward function
    // This is where you can set breakpoints in cuda-gdb
    // Using .contiguous().data<float>() to get raw pointers (same as rasterize_points.cu)
    int rendered = CudaRasterizer::Rasterizer::forward(
        geomFunc,
        binningFunc,
        imgFunc,
        P, S, D, M,
        background.contiguous().data<float>(),
        W, H,
        means3D.contiguous().data<float>(),
        shs.contiguous().data_ptr<float>(),
        colors_precomp.contiguous().data<float>(),
        features.contiguous().data_ptr<float>(),
        opacities.contiguous().data<float>(),
        scales.contiguous().data_ptr<float>(),
        scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        cov3D_precomp.contiguous().data<float>(),
        mask.contiguous().data_ptr<bool>(),
        viewmatrix.contiguous().data<float>(),
        campos.contiguous().data<float>(),
        prefiltered,
        out_contrib.contiguous().data_ptr<int>(),
        out_color.contiguous().data<float>(),
        out_feature.contiguous().data_ptr<float>(),
        out_depth.contiguous().data<float>(),
        out_acc.contiguous().data<float>(),
        radii.contiguous().data<int>(),
        psi.contiguous().data<float>(),
        lat.contiguous().data<float>(),
        lon.contiguous().data<float>(),
        debug
    );
    
    torch::cuda::synchronize();
    
    std::cout << "\nRasterization completed!" << std::endl;
    std::cout << "Rendered Gaussians: " << rendered << std::endl;
    
    // Copy results back to CPU for verification (using PyTorch)
    auto radii_cpu = radii.cpu();
    auto out_acc_cpu = out_acc.cpu();
    auto radii_a = radii_cpu.accessor<int, 1>();
    auto out_acc_a = out_acc_cpu.accessor<float, 2>();
    
    // Print some statistics
    int visible_count = 0;
    for (int i = 0; i < P; i++) {
        if (radii_a[i] > 0) visible_count++;
    }
    std::cout << "Visible Gaussians: " << visible_count << " / " << P << std::endl;
    
    float max_acc = 0.0f;
    float avg_acc = 0.0f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float acc_val = out_acc_a[i][j];
            if (acc_val > max_acc) max_acc = acc_val;
            avg_acc += acc_val;
        }
    }
    avg_acc /= (H * W);
    std::cout << "Max accumulated alpha: " << max_acc << std::endl;
    std::cout << "Average accumulated alpha: " << avg_acc << std::endl;
    
    // Cleanup - PyTorch tensors are automatically freed when they go out of scope
    std::cout << "\nCleaning up..." << std::endl;
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
