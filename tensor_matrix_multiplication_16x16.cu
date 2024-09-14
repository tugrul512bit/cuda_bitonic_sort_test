#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <device_functions.h>
#include <mma.h>

__global__ void matrixMul(
    half*  a, half*  b, half*  c
    )
{
    const int index = (threadIdx.x + blockIdx.x * blockDim.x);
    cudaStream_t stream0;
    cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
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

    // all warp threads need to execute this
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
    nvcuda::wmma::load_matrix_sync(a_frag, a, lda);
    nvcuda::wmma::load_matrix_sync(b_frag, b, ldb);
    nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    nvcuda::wmma::load_matrix_sync(c_frag, c, ldc, nvcuda::wmma::mem_col_major);
    
    // all warp threads need to execute this
    for (int i = 0; i < c_frag.num_elements; i++)    
        c_frag.x[i] += acc_frag.x[i];
    
    nvcuda::wmma::store_matrix_sync(c, c_frag, ldc, nvcuda::wmma::mem_col_major);
}

#include<iostream>
void test2()
{
    const int n = 16*16;
    half *dvcA,*dvcB,*dvcC;
    cudaMalloc(&dvcA, n * sizeof(half));
    cudaMalloc(&dvcB, n * sizeof(half));
    cudaMalloc(&dvcC, n * sizeof(half));

    half *hstA,*hstB,*hstC;
    cudaHostAlloc(&hstA, n * sizeof(half), cudaHostAllocDefault);
    cudaHostAlloc(&hstB, n * sizeof(half), cudaHostAllocDefault);
    cudaHostAlloc(&hstC, n * sizeof(half), cudaHostAllocDefault);
    
 
    for (int i = 0; i < n; i++)
    {
        hstA[i] = i / 100.0f; // division to not overflow 16-bit floats
        hstB[i] = i / 100.0f;
        hstC[i] = 0;
    }
    std::cout << "tensor" << std::endl;

    cudaStream_t stream0;
    cudaStreamCreate(&stream0);

    cudaEvent_t evt,evt2;
    cudaEventCreate(&evt);
    cudaEventCreate(&evt2);
    cudaEventRecord(evt, stream0);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    for (int i = 0; i < 1; i++)
    {
        cudaMemcpyAsync(dvcA, hstA, n * sizeof(half), ::cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dvcB, hstB, n * sizeof(half), ::cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dvcC, hstC, n * sizeof(half), ::cudaMemcpyHostToDevice, stream0);

        // launch single warp (may not be 32 in future)
        matrixMul<<<1, props.warpSize,0, stream0>>>(dvcA,dvcB,dvcC);
        

        cudaMemcpyAsync(hstC, dvcC, n * sizeof(half), ::cudaMemcpyDeviceToHost, stream0);
    }
    
    cudaEventRecord(evt2, stream0);
    cudaEventSynchronize(evt2);
    float tim;
    cudaEventElapsedTime(&tim, evt, evt2);


    std::cout<<(float)hstC[5+3*16]<< std::endl;
    float acc = 0.0f;
    for (int i = 0; i < 16; i++)
    {
        acc += (i+3*16) * (i * 16+5);
    }
    std::cout << acc << std::endl;

    cudaFreeHost(hstA);
    cudaFree(dvcA);
    
}

int main()
{
    test2();
	return 0;
}
