// each kernel is single-block and computes a chunk currently (starting chunk = whole array)
// when a chunk has more than 64 elements, it does a step of quicksort
// after many steps, chunks get smaller, 
// when 64 or less sized, they are sorted in brute-force algorithm (odd-even parallel sort, not even optimized)
// this reduces number of launched kernels from ~1M to 60k. This improves performance 15x-20x compared to pure-quicksort.
// todo: insert child kernel parameters into an array
//          then launch single kernel with multiple blocks, to compute with 1 block per child kernel parameter
//          expected speedup=2^nDepth on kernel-launch-overhead

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <device_functions.h>

#include<iostream>
#include<vector>

__global__ void quickSortWithoutStreamCompaction(unsigned int* arr, unsigned int* leftMem, unsigned int* rightMem, int startIncluded, int stopIncluded, int depth, unsigned int * numKernels)
{
    const int id = threadIdx.x; // 1 block
    
    const int bd = blockDim.x;
    int left = 0;
    int right = 0;
    unsigned int pivot = arr[stopIncluded];
    const int num = stopIncluded - startIncluded + 1;
    if (num <= 0)
        return;

    if (id == 0) atomicAdd(&numKernels[0], 1);
    if (id == 0 && num <= 64) atomicAdd(&numKernels[1], 1);

    // if number of elements are 64 or less (90% of kernels for 1M-element array are these)
    // switch to brute-force to reduce total kernel launch overhead
    __shared__ unsigned int brute[64];
    unsigned int rank = 0;
    if (num <= 64)
    {
        if (id < num)
        {
            brute[id] = arr[startIncluded + id];
        }

    }
    __syncthreads();

    if (num <= 64)
    for (int i = 0; i < num; i++)
    {
        if (id < num/2)
        {
            if (i & 1)
            {
                if(id * 2 + 1<num)
                if (brute[id * 2 + 1] < brute[id * 2])
                {
                    int tmp = brute[id * 2 + 1];
                    brute[id * 2 + 1] = brute[id * 2];
                    brute[id * 2] = tmp;
                }
            }
            else
            {
                if(id * 2 + 2<num)
                if (brute[id * 2 + 2] < brute[id * 2+1])
                {
                    int tmp = brute[id * 2 + 2];
                    brute[id * 2 + 2] = brute[id * 2+1];
                    brute[id * 2+1] = tmp;
                }
            }
        }
        __syncthreads();
    }
    if (num <= 64)
    {
        if (id < num)
        {
            arr[startIncluded + id] = brute[rank];
        }
    }

    if (num <= 64)
        return;
  
    __shared__ int indexLeft;
    __shared__ int indexRight;
    int indexLeftR = 0;
    int indexRightR = 0;
    if (id == 0)
    {
        indexLeft = 0;
        indexRight = 0;
    }
    __syncthreads();

    {
        const int steps = (num / bd) + 1;
        for (int i = 0; i < steps; i++)
        {
            const int curId = i * bd + id;
            if (curId < num)
            {
                const auto data = arr[curId + startIncluded];
                if (data < pivot)
                    leftMem[startIncluded + atomicAdd(&indexLeft, 1)] = data;
                else
                    rightMem[startIncluded + atomicAdd(&indexRight, 1)] = data;
            }
        }
    }
    __syncthreads();
    indexLeftR = indexLeft;
    indexRightR = indexRight;
    if (indexLeftR > 0)
    {
        const int steps = (indexLeftR / bd) + 1;
        for (int i = 0; i < steps; i++)
        {
            const int curId = i * bd + id;
            if (curId < indexLeftR)
            {
                arr[curId + startIncluded] = leftMem[startIncluded + curId];
            }
        }
    }

    if (indexRightR > 0)
    {
        const int steps = (indexRightR / bd) + 1;
        for (int i = 0; i < steps; i++)
        {
            const int curId = i * bd + id;
            if (curId < indexRightR)
            {
                arr[curId + indexLeftR + startIncluded] = rightMem[startIncluded + curId];
            }
        }
    }

    __syncthreads();

    if (id == 0)
    {
        if (indexLeftR + indexRightR != num)
            printf("( error %i %i %i) ", indexLeftR, indexRightR, num);
        auto tmp = arr[indexLeftR + startIncluded];
        arr[stopIncluded] = tmp;
        arr[indexLeftR + startIncluded] = pivot;
    }

    __syncthreads();
    auto nLeft = indexLeftR;
    auto nRight = indexRightR;
    const int bdDynamicSizeLeft = (nLeft > 1024 ? 1024 : (nLeft > 512 ? 512 : (nLeft > 256 ? 256 : (nLeft > 128 ? 128 : 64))));
    const int bdDynamicSizeRight = (nRight > 1024 ? 1024 : (nRight > 512 ? 512 : (nRight > 256 ? 256 : (nRight > 128 ? 128 : 64))));

    if (id == 0)
    {
        // to add limited kernel parallelism, child kernels are launched independently with their own streams
        // maximum 32 (depth 4 ==> left + right) kernels launched concurrently
        if (depth < 5)
        {
            cudaStream_t stream0;
            cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
            cudaStream_t stream1;
            cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
            if (nLeft > 1)
                quickSortWithoutStreamCompaction <<<1, bdDynamicSizeLeft, 0, stream0 >>> (arr, leftMem, rightMem, startIncluded, startIncluded + nLeft - 1, depth + 1,numKernels);


            if (nRight > 1)
                quickSortWithoutStreamCompaction <<<1, bdDynamicSizeRight, 0, stream1 >>> (arr, leftMem, rightMem, startIncluded + nLeft + 1, stopIncluded, depth + 1, numKernels);
            cudaStreamDestroy(stream0);
            cudaStreamDestroy(stream1);
        }
        else // deep nodes are too many so they are computed serially
        {
            if (nLeft > 1)
                quickSortWithoutStreamCompaction <<<1, bdDynamicSizeLeft >>> (arr, leftMem, rightMem, startIncluded, startIncluded + nLeft - 1, depth + 1, numKernels);


            if (nRight > 1)
                quickSortWithoutStreamCompaction <<<1, bdDynamicSizeRight >>> (arr, leftMem, rightMem, startIncluded + nLeft + 1, stopIncluded, depth + 1, numKernels);
        }
    }
}

void test()
{
    constexpr int n = 1024 * 32;
    unsigned int *data, *left, * right,*numKernels;
    std::vector<unsigned int> hostData(n);
    cudaMalloc(&data, n * sizeof(unsigned int));
    cudaMalloc(&left, n * sizeof(unsigned int));
    cudaMalloc(&right, n * sizeof(unsigned int));
    cudaMalloc(&numKernels, 2*sizeof(unsigned int));
    int numKernelsHost[2];
    for (int j = 0; j < 5; j++)
    {
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand();
        }
        numKernelsHost[0] = 0;
        numKernelsHost[1] = 0;
        cudaMemcpy((void*)data, hostData.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)numKernels, &numKernelsHost, 2*sizeof(unsigned int), cudaMemcpyHostToDevice);
        quickSortWithoutStreamCompaction <<<1, 1024 >>> (data, left, right, 0, n - 1, 0,numKernels);
        cudaDeviceSynchronize();
        cudaMemcpy(hostData.data(), (void*)data, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&numKernelsHost, (void*)numKernels,  2*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        std::cout << "number of kernel launches = " << numKernelsHost[0] << std::endl;
        std::cout << "number of kernel launches with <= 64 elements= " << numKernelsHost[1] << std::endl;
    }
   
    bool err = false;
    for (int i = 0; i < n - 1; i++)
        if (hostData[i] > hostData[i + 1])
        {
            std::cout << "error at: " << i << ": " << hostData[i] << std::endl;
            err = true;
            break;
        }
    if (!err)
    {
        std::cout << "quicksort completed successfully" << std::endl;
    }
    cudaFree(data);
    cudaFree(left);
    cudaFree(right);
}

int main()
{
    test();

    return 0;
}
