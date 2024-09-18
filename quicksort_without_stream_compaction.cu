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

__global__ void quickSortWithoutStreamCompaction(unsigned int* arr, unsigned int* leftMem, unsigned int* rightMem, int startIncluded, int stopIncluded,int depth)
{
    const int id = threadIdx.x; // 1 block
    const int bd = blockDim.x;
    int left = 0;
    int right = 0;
    unsigned int pivot = arr[stopIncluded];
    const int num = stopIncluded - startIncluded + 1;
    if (num <= 0)
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
                arr[curId + startIncluded] = leftMem[startIncluded+curId];
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
                arr[curId + indexLeftR + startIncluded] = rightMem[startIncluded+ curId];
            }
        }
    }

    __syncthreads();

    if(id==0)
    {             
        if (indexLeftR + indexRightR != num)
            printf("( error %i %i %i) ",indexLeftR,indexRightR,num);
        auto tmp = arr[indexLeftR + startIncluded];
        arr[stopIncluded] = tmp;
        arr[indexLeftR + startIncluded] = pivot;     
    }
    
    __syncthreads();
    auto nLeft = indexLeftR;
    auto nRight = indexRightR;
    const int bdDynamicSizeLeft = (nLeft > 1024 ? 1024 : (nLeft > 512 ? 512 : (nLeft > 256 ? 256 : (nLeft > 128 ? 128 : (nLeft > 64 ? 64 : 32)))));
    const int bdDynamicSizeRight = (nRight > 1024 ? 1024 : (nRight > 512 ? 512 : (nRight > 256 ? 256 : (nRight > 128 ? 128 : (nRight > 64 ? 64 : 32)))));

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
                quickSortWithoutStreamCompaction <<<1, bdDynamicSizeLeft, 0, stream0 >>> (arr, leftMem, rightMem, startIncluded, startIncluded + nLeft - 1, depth + 1);


            if (nRight > 1)
                quickSortWithoutStreamCompaction <<<1, bdDynamicSizeRight, 0, stream1 >>> (arr, leftMem, rightMem, startIncluded + nLeft + 1, stopIncluded, depth + 1);
            cudaStreamDestroy(stream0);
            cudaStreamDestroy(stream1);
        }
        else // deep nodes are too many so they are computed serially
        {
            if (nLeft > 1)
                quickSortWithoutStreamCompaction <<<1, bdDynamicSizeLeft >>> (arr, leftMem, rightMem, startIncluded, startIncluded + nLeft - 1, depth + 1);


            if (nRight > 1)
                quickSortWithoutStreamCompaction <<<1, bdDynamicSizeRight>>> (arr, leftMem, rightMem, startIncluded + nLeft + 1, stopIncluded, depth + 1);
        }
    }
}

void test()
{
    constexpr int n = 1024 * 1024;
    unsigned int *data,*left,*right;
    std::vector<unsigned int> hostData(n);
    cudaMalloc(&data, n * sizeof(unsigned int));
    cudaMalloc(&left, n * sizeof(unsigned int));
    cudaMalloc(&right, n * sizeof(unsigned int));
  
    for (int j = 0; j < 5; j++)
    {
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand();
        }
        cudaMemcpy((void*)data, hostData.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice);
        quickSortWithoutStreamCompaction <<<1, 1024 >>> (data,left,right, 0, n - 1, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(hostData.data(), (void*)data, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    }
    bool err = false;
    for(int i=0;i< n -1;i++)
        if (hostData[i] > hostData[i + 1])
        {
            std::cout<<"error at: " << i << ": " << hostData[i] << std::endl;
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

    bool cpuStreamCompactionTest = false;
    if (cpuStreamCompactionTest)
    {
        int t[100];
        for (int i = 0; i < 100; i++)
        {
            t[i] = (i % 8) - 4;
            std::cout << t[i] << " ";
        }
        std::cout << std::endl;

        int t2[100], t3[100];
        for (int i = 0; i < 100; i++)
            if (t[i] >= 0)
                t2[i] = 1;
            else
                t2[i] = 0;

        for (int i = 0; i < 100; i++)
            t3[i] = t2[i];

        int step = 1;
        while (step < 100)
        {
            for (int i = 0; i < 100; i++)
                if (i - step >= 0)
                    t2[i] += t3[i - step];

            for (int i = 0; i < 100; i++)
                t3[i] = t2[i];
            step *= 2;
        }
        for (int i = 0; i < 100; i++)
            t3[i] = -1;

        for (int i = 98; i >= 0; i--)
            t2[i + 1] = t2[i];
        t2[0] = 0;

        std::cout << "---------------" << std::endl;
        for (int i = 0; i < 100; i++)
        {
            std::cout << t2[i] << " ";
        }
        std::cout << std::endl;

        for (int i = 1; i < 100; i++)
        {
            if (t2[i] == t2[i - 1] + 1)
            {
                t3[t2[i] - 1] = t[i - 1];
            }
        }
        std::cout << "---------------" << std::endl;
        for (int i = 0; i < 100; i++)
        {
            std::cout << t3[i] << " ";
        }
        std::cout << std::endl;
    }
    
    
    return 0;
}
