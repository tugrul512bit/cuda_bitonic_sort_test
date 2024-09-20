// hybrid quicksort (16 milliseconds for 4M elements inside RTX4070)
// when chunk size is greater than 1024, it does quicksort steps
// continues splitting chunks
// when chunk size is 1024 or less, executes parallel odd-even sort

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

__global__ void quickSortWithoutStreamCompaction(
    unsigned int* arr, unsigned int* leftMem, unsigned int* rightMem, int depth, unsigned int* numTasks,
    int* tasks, int* tasks2);

__global__ void resetNumTasks(unsigned int* arr, unsigned int* leftMem, unsigned int* rightMem, int depth, unsigned int* numTasks,
    int* tasks, int* tasks2)
{
    const int n = numTasks[0];

    if (n > 0)
    {
        numTasks[0] = 0;
        //printf("\n %i \n", n);

        quickSortWithoutStreamCompaction << <n, 1024, 0, cudaStreamTailLaunch >> > (arr, leftMem, rightMem, depth, numTasks, tasks, tasks2);
    }

}

__global__ void copyTasksBack(unsigned int* arr, unsigned int* leftMem, unsigned int* rightMem, int depth, unsigned int* numTasks,
    int* tasks, int* tasks2)
{
    const int id = threadIdx.x;
    const int n = numTasks[0];
    const int steps = 1 + n / 1024;

    for (int i = 0; i < steps; i++)
    {
        const int curId = id + i * 1024;
        if (curId < n)
        {
            tasks[curId * 2] = tasks2[curId * 2];
            tasks[curId * 2 + 1] = tasks2[curId * 2 + 1];
        }
    }
    if (id == 0)
    {
        resetNumTasks << <1, 1, 0, cudaStreamTailLaunch >> > (arr, leftMem, rightMem, depth, numTasks, tasks, tasks2);
    }
}

// task pattern: 
//              task 0      task 1      task 2      task 3      ---> array chunks to sort (no overlap)
//              start stop  start stop  start stop  start stop  ---> tasks buffer
//              block 0     block 1     block 2     block 3     ---> cuda blocks
__global__ void quickSortWithoutStreamCompaction(
    unsigned int* arr, unsigned int* leftMem, unsigned int* rightMem, int depth, unsigned int* numTasks,
    int* tasks, int* tasks2)
{
    const int gr = gridDim.x;

    // 1 block = 1 chunk of data
    const int gid = blockIdx.x;
    const int id = threadIdx.x;

    if (id == 0 && gid == 0)
        copyTasksBack << <1, 1024, 0, cudaStreamTailLaunch >> > (arr, leftMem, rightMem, depth, numTasks, tasks, tasks2);

    const int startIncluded = tasks[gid * 2];
    const int stopIncluded = tasks[gid * 2 + 1];
    const int num = stopIncluded - startIncluded + 1;


    if (num < 2)
        return;

    if (num == 2)
    {
        if (id == 0)
        {
            if (arr[startIncluded] > arr[startIncluded + 1])
            {
                unsigned int tmp = arr[startIncluded];
                arr[startIncluded] = arr[startIncluded + 1];
                arr[startIncluded + 1] = tmp;
            }
        }

        return;
    }


    const int bd = blockDim.x;


    int left = 0;
    int right = 0;
    unsigned int pivot = arr[stopIncluded];

    // if chunk size is 1024 or less, do brute-force sorting
    __shared__ unsigned int cache[1024];

    if (num <= 1024)
    {
        if (id < num)
        {
            cache[id] = arr[startIncluded + id];
        }
    }
    __syncthreads();
    if (num <= 1024)
    {
        for (int i = 0; i < num; i++)
        {
            if (id + 1 < num && (id % 2 == 0))
                if (cache[id + 1] < cache[id])
                {
                    unsigned int tmp = cache[id + 1];
                    cache[id + 1] = cache[id];
                    cache[id] = tmp;
                }
            __syncthreads();
            if (id + 1 < num && !(id % 2 == 0))
                if (cache[id + 1] < cache[id])
                {
                    unsigned int tmp = cache[id + 1];
                    cache[id + 1] = cache[id];
                    cache[id] = tmp;
                }
            __syncthreads();
        }
    }

    if (num <= 1024)
    {
        if (id < num)
        {
            arr[startIncluded + id] = cache[id];
        }
    }
    if (num <= 1024)
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
                {
                    if (curId + startIncluded != stopIncluded)
                    {
                        rightMem[startIncluded + atomicAdd(&indexRight, 1)] = data;
                    }
                }
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
    if (id == 0)
    {
        arr[startIncluded + indexLeftR] = pivot;
    }
    if (indexRightR > 0)
    {
        const int steps = (indexRightR / bd) + 1;
        for (int i = 0; i < steps; i++)
        {
            const int curId = i * bd + id;
            if (curId + indexLeftR + startIncluded + 1 <= stopIncluded)
            {
                arr[curId + indexLeftR + startIncluded + 1] = rightMem[startIncluded + curId];
            }
        }
    }
    __syncthreads();
    auto nLeft = indexLeftR;
    auto nRight = indexRightR;

    if (id == 0)
    {
        if (nLeft > 1)
        {
            if (startIncluded + nLeft - 1 > startIncluded)
            {
                const int index = atomicAdd(&numTasks[0], 1);
                tasks2[index * 2] = startIncluded;
                tasks2[index * 2 + 1] = startIncluded + nLeft - 1;

            }
        }


        if (nRight > 1)
        {
            if (stopIncluded > startIncluded + nLeft + 1)
            {
                const int index = atomicAdd(&numTasks[0], 1);
                tasks2[index * 2] = startIncluded + nLeft + 1;
                tasks2[index * 2 + 1] = stopIncluded;

            }
        }
    }


}



__global__ void qSortMain(
    unsigned int* arr, unsigned int* leftMem, unsigned int* rightMem, int depth, unsigned int* numTasks,
    int* tasks, int* tasks2)
{

    quickSortWithoutStreamCompaction << <1, 1024 >> > (arr, leftMem, rightMem, depth, numTasks, tasks, tasks2);
}

void test()
{
    constexpr int n = 1024 * 1024 * 4;
    unsigned int* data, * left, * right, * numTasks;
    int* tasks, * tasks2;
    std::vector<unsigned int> hostData(n);
    std::vector<int> hostTasks(2);
    cudaMalloc(&data, n * sizeof(unsigned int));
    cudaMalloc(&left, n * sizeof(unsigned int));
    cudaMalloc(&right, n * sizeof(unsigned int));
    cudaMalloc(&numTasks, 2 * sizeof(unsigned int));
    cudaMalloc(&tasks, n * sizeof(int));
    cudaMalloc(&tasks2, n * sizeof(int));
    unsigned int numTasksHost[2];
    for (int j = 0; j < 5; j++)
    {
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand();
        }
        numTasksHost[0] = 1; // launch 1 block first
        numTasksHost[1] = 0;
        hostTasks[0] = 0;
        hostTasks[1] = n - 1; // first block's chunk limits: 0 - n-1
        cudaMemcpy((void*)data, hostData.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)numTasks, numTasksHost, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)tasks, hostTasks.data(), 2 * sizeof(int), cudaMemcpyHostToDevice); // host only gives 1 task with 2 parameters
        qSortMain << <1, 1 >> > (data, left, right, 0, numTasks, tasks, tasks2);
        cudaDeviceSynchronize();
        cudaMemcpy(hostData.data(), (void*)data, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(numTasksHost, (void*)numTasks, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

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
    cudaFree(tasks);
    cudaFree(tasks2);
    cudaFree(numTasks);
}

int main()
{
    test();

    return 0;
}
