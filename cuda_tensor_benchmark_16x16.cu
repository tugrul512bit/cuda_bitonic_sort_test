// 114 TFLOPS for rtx4070

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <device_functions.h>
#include <mma.h>
constexpr int numKernelRepeats = 10;
constexpr int numTensorRepeats = 1000;
constexpr int numMatrixInstances = 1024 * 1024;
__global__ void matrixMul(
    half* a, half* b, half* c
)
{
    const int indexWarp = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int indexBlock = blockIdx.x;
    const int indexThread = threadIdx.x;
    
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int lda = 16;
    constexpr int ldb = 16;
    constexpr int ldc = 16;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    __shared__ half sA[16 * 16 * 32]; // 32 warps per 1024 threads
    __shared__ half sB[16 * 16 * 32];

    // loading data as a single chunk coalesced pattern
    const int num = 32 * 16 * 16;
    const int steps = num / 1024;
    for (int i = 0; i < steps; i++)
    {
        sA[indexThread + i * 1024] = a[indexThread + i * 1024 + indexBlock * 16 * 16 * 32];
        sB[indexThread + i * 1024] = b[indexThread + i * 1024 + indexBlock * 16 * 16 * 32];
    }
    __syncthreads();

    nvcuda::wmma::load_matrix_sync(a_frag, sA + (indexWarp & 31) * 16 * 16, lda);
    nvcuda::wmma::load_matrix_sync(b_frag, sB + (indexWarp & 31) * 16 * 16, ldb);

    // all warp threads need to execute this
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < numTensorRepeats; i++)
    {
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    nvcuda::wmma::load_matrix_sync(c_frag, c + indexWarp * 16 * 16, ldc, nvcuda::wmma::mem_col_major);

    // all warp threads need to execute this
    for (int i = 0; i < c_frag.num_elements; i++)
        c_frag.x[i] += acc_frag.x[i];

    nvcuda::wmma::store_matrix_sync(c + indexWarp * 16 * 16, c_frag, ldc, nvcuda::wmma::mem_col_major);
}

#include<iostream>
void test()
{
    // elements for 1M matrices of size 16x16
    const int n = 16 * 16 * numMatrixInstances;
    half* dvcA, * dvcB, * dvcC;
    cudaMalloc(&dvcA, n * sizeof(half));
    cudaMalloc(&dvcB, n * sizeof(half));
    cudaMalloc(&dvcC, n * sizeof(half));

    half* hstA, * hstB, * hstC;
    cudaHostAlloc(&hstA, n * sizeof(half), cudaHostAllocDefault);
    cudaHostAlloc(&hstB, n * sizeof(half), cudaHostAllocDefault);
    cudaHostAlloc(&hstC, n * sizeof(half), cudaHostAllocDefault);


    for (int i = 0; i < n; i++)
    {
        hstA[i] = i / 10000.0f; // division to not overflow 16-bit floats
        hstB[i] = i / 10000.0f;
        hstC[i] = 0;
    }
    std::cout << "tensor" << std::endl;

    cudaStream_t stream0;
    cudaStreamCreate(&stream0);

    // warm-up 
    for (int i = 0; i < 10; i++)
        matrixMul << <32 * 1024, 1024, 0, stream0 >> > (dvcA, dvcB, dvcC);

    cudaEvent_t evt, evt2;
    cudaEventCreate(&evt);
    cudaEventCreate(&evt2);
    cudaEventRecord(evt, stream0);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    cudaMemcpyAsync(dvcA, hstA, n * sizeof(half), ::cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(dvcB, hstB, n * sizeof(half), ::cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(dvcC, hstC, n * sizeof(half), ::cudaMemcpyHostToDevice, stream0);
    for (int i = 0; i < numKernelRepeats; i++)
    {
        // launch 1M warps (in blocks of 1024 threads or 32 warps)
        matrixMul << <32 * 1024, 1024, 0, stream0 >> > (dvcA, dvcB, dvcC);
    }
    cudaMemcpyAsync(hstC, dvcC, n * sizeof(half), ::cudaMemcpyDeviceToHost, stream0);
    cudaEventRecord(evt2, stream0);
    cudaEventSynchronize(evt2);
    float tim;
    cudaEventElapsedTime(&tim, evt, evt2);
    std::cout << "multiplying 16x16 sized matrices for " << (numMatrixInstances * (size_t)numKernelRepeats * (size_t)numTensorRepeats) << " times took " << tim << " ms" << std::endl;
    std::cout << "each matrix has " << 16 * 16 * 16 * 2 << " 16-bit flop" << std::endl;
    std::cout << "compute performance: " << (numMatrixInstances * (size_t)numKernelRepeats * (size_t)numTensorRepeats * 16 * 16 * 16 * 2) / (tim / 1000.0f) << " flop/s" << std::endl;
    std::cout << (float)hstC[5 + 3 * 16] << std::endl;
    cudaFreeHost(hstA);
    cudaFree(dvcA);

}

int main()
{
    test();
    return 0;
}
