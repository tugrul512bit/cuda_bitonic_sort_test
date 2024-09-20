// hybrid quicksort + rtx4070
// with dynamic parallelism: 16 milliseconds for 4M elements
// with host-launch : 30 milliseconds for 4M elements
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void resetNumTasks( int* numTasks)
{
    numTasks[0]=0;
    numTasks[1]=0;
}

__global__ void copyTasksBack( int* arr,  int* leftMem,  int* rightMem,  int* numTasks,
    int* tasks, int* tasks2, int* tasks3, int* tasks4)
{
    const int id = threadIdx.x;
    const int n = numTasks[0];
    const int n2 = numTasks[1];
    const int steps = 1 + n / 1024;
    const int steps2 = 1 + n2 / 1024;

    for (int i = 0; i < steps; i++)
    {
        const int curId = id + i * 1024;
        if (curId < n)
        {
            tasks[curId * 2] = tasks2[curId * 2];
            tasks[curId * 2 + 1] = tasks2[curId * 2 + 1];
        }
    }



    for (int i = 0; i < steps2; i++)
    {
        const int curId = id + i * 1024;
        if (curId < n2)
        {
            tasks3[curId * 2] = tasks4[curId * 2];
            tasks3[curId * 2 + 1] = tasks4[curId * 2 + 1];
        }
    }

}

#define check(x)  if (x != cudaSuccess) std::cout << __LINE__ << " " << cudaGetErrorString(x) << std::endl
#define kcheck(x) if (x != cudaSuccess) printf("device: %s\n", cudaGetErrorString(x))

__global__ void bruteSort(int * __restrict__ arr, int* __restrict__ tasks3)
{
    const int id = threadIdx.x;
    const int gid = blockIdx.x;
    const int startIncluded = tasks3[gid * 2];
    const int stopIncluded = tasks3[gid * 2 + 1];
    const int num = stopIncluded - startIncluded + 1;

    __shared__ int cache[1024];
    if (id < num && startIncluded + id <= stopIncluded)
    {
        cache[id] = arr[startIncluded+id];
    }
    __syncthreads();
    for (int i = 0; i < num; i++)
    {
        if (id +1< num)
        {

            if ((id % 2 == 0) && (cache[id + 1] < cache[id]))
            {
                cache[id] ^= cache[id + 1];
                cache[id + 1] ^= cache[id];
                cache[id] ^= cache[id + 1];
            }
        }
        __syncthreads();
        if (id +1 < num)
        {

            if ((id % 2 == 1) && (cache[id + 1] < cache[id]))
            {
                cache[id] ^= cache[id + 1];
                cache[id + 1] ^= cache[id];
                cache[id] ^= cache[id + 1];

            }
        }
        __syncthreads();
    }
    if (id < num && startIncluded + id <= stopIncluded)
    {
        arr[startIncluded + id]= cache[id];
    }
}

// task pattern: 
//              task 0      task 1      task 2      task 3      ---> array chunks to sort (no overlap)
//              start stop  start stop  start stop  start stop  ---> tasks buffer
//              block 0     block 1     block 2     block 3     ---> cuda blocks
__global__ void quickSortWithoutStreamCompaction(
    int* __restrict__ arr, int* __restrict__ leftMem, int* __restrict__ rightMem, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks4)
{
    // 1 block = 1 chunk of data
    const int gid = blockIdx.x;
    const int id = threadIdx.x;


    const int startIncluded = tasks[gid * 2];
    const int stopIncluded = tasks[gid * 2 + 1];
    const int num = stopIncluded - startIncluded + 1;

    __shared__ int indexLeft;
    __shared__ int indexRight;

    if (num <= 1)
        return;



    const int bd = blockDim.x;

    int pivot = arr[stopIncluded];



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
        // push new "quick" task
        if (nLeft > 1)
        {
            if (startIncluded + nLeft - 1 > startIncluded)
            {
                if (startIncluded + nLeft - 1 - startIncluded + 1 <= 1024)
                {
                    const int index = atomicAdd(&numTasks[1], 1);
                    tasks4[index * 2] = startIncluded;
                    tasks4[index * 2 + 1] = startIncluded + nLeft - 1;
                }
                else
                {
                    const int index = atomicAdd(&numTasks[0], 1);
                    tasks2[index * 2] = startIncluded;
                    tasks2[index * 2 + 1] = startIncluded + nLeft - 1;
                }
            }
        }

        // push new "quick" task
        if (nRight > 1)
        {
            if (stopIncluded > startIncluded + nLeft + 1)
            {
                if (stopIncluded - (startIncluded + nLeft + 1) + 1 <= 1024)
                {
                    const int index = atomicAdd(&numTasks[1], 1);
                    tasks4[index * 2] = startIncluded + nLeft + 1;
                    tasks4[index * 2 + 1] = stopIncluded;
                }
                else
                {
                    const int index = atomicAdd(&numTasks[0], 1);
                    tasks2[index * 2] = startIncluded + nLeft + 1;
                    tasks2[index * 2 + 1] = stopIncluded;
                }
            }
        }
    }
}


__global__ void qSortMain(
     int* arr,  int* leftMem,  int* rightMem, int depth,  int* numTasks,
    int* tasks, int* tasks2)
{
    



}

void test()
{
    constexpr int n = 1024 * 1024*4;
     int* data, * left, * right, * numTasks;
    int* tasks, * tasks2,*tasks3,*tasks4;
    std::vector< int> hostData(n);
    std::vector<int> hostTasks(2);
    gpuErrchk( cudaSetDevice(0));
    gpuErrchk( cudaDeviceSynchronize());
    gpuErrchk( cudaMalloc(&data, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&left, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&right, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&numTasks, 2 * sizeof(int)));
    gpuErrchk( cudaMalloc(&tasks, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&tasks2, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&tasks3, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&tasks4, n * sizeof(int)));
    int numTasksHost[2];
    int nQuickTask = 1;
    int nBruteTask =0;
        
    for (int j = 0; j < 40; j++)
    {
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand();//n-i; //rand();
        }

        numTasksHost[0] = 1; // launch 1 block first
        numTasksHost[1] = 0;
        hostTasks[0] = 0;
        hostTasks[1] = n - 1; // first block's chunk limits: 0 - n-1
       gpuErrchk( cudaMemcpy((void*)data, hostData.data(), n * sizeof( int), cudaMemcpyHostToDevice));
       gpuErrchk( cudaMemcpy((void*)numTasks, numTasksHost, 2 * sizeof( int), cudaMemcpyHostToDevice));
       gpuErrchk( cudaMemcpy((void*)tasks, hostTasks.data(), 2 * sizeof(int), cudaMemcpyHostToDevice)); // host only gives 1 task with 2 parameters
        nQuickTask = 1;
        nBruteTask = 0;

        while (nQuickTask > 0 || nBruteTask >0)
        {
            cudaDeviceSynchronize();
            //qSortMain << <1, 1 >> > (data, left, right, 0, numTasks, tasks, tasks2);
            if (nQuickTask > 0)
            quickSortWithoutStreamCompaction <<<nQuickTask, 1024>>> (data, left, right, numTasks, tasks, tasks2,tasks4);
            gpuErrchk(cudaDeviceSynchronize());


            if (nBruteTask > 0)
                bruteSort <<<nBruteTask, 1024>>> (data, tasks3);
            gpuErrchk(cudaDeviceSynchronize());
            kcheck(cudaPeekAtLastError());

            

            //std::cout << "m=" << nQuickTask;
           // std::cout << "  n=" << nBruteTask << std::endl;
            gpuErrchk(cudaMemcpy(numTasksHost, (void*)numTasks, 2 * sizeof(int), cudaMemcpyDeviceToHost));
            nQuickTask = numTasksHost[0];
            nBruteTask = numTasksHost[1];

            copyTasksBack << <1, 1024 >> > (data, left, right, numTasks, tasks, tasks2,tasks3,tasks4);
            resetNumTasks << <1, 1 >> > (numTasks);
            kcheck(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

        }
        gpuErrchk(cudaMemcpy(hostData.data(), (void*)data, n * sizeof( int), cudaMemcpyDeviceToHost));
        bool err = false;
        for (int i = 0; i < n - 2; i++)
            if (hostData[i] > hostData[i + 1])
            {
                std::cout << "error at: " << i << ": " << hostData[i]<<" "<<hostData[i+1]<<" "<<hostData[i+2]  << std::endl;
                err = true;
                j = 1000000;
                break;
            }
        if (!err)
        {
            std::cout << "quicksort completed successfully "<<j << std::endl;
           // for (int i = 0; i < 35; i++)
            //    std::cout << hostData[i] << " ";
        }
    }

    cudaFree(data);
    cudaFree(left);
    cudaFree(right);
    cudaFree(tasks);
    cudaFree(tasks2);
    cudaFree(tasks3);
    cudaFree(tasks4);
    cudaFree(numTasks);
}

int main()
{
    test();

    return 0;
}
