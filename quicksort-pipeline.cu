// hybrid quicksort pipeline that uses CPU threads to sort multiple arrays on GPU concurrently
// it overlaps multiple kernels to fill empty/idle SM units
// for example, quicksort starts with 1 block then spawns 2 blocks then increases to thousands of blocks
// but single sorting means 1 block uses only 2% of GPU
// so another sorting kernel can fill empty SM units and use GPU fully
// concurrency and maximum array size to sort can be adjusted from template parameters

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
                if (cache[id+1] < cache[id])
                {
                    unsigned int tmp = cache[id+1];
                    cache[id+1] = cache[id];
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
             arr[startIncluded + id]= cache[id];
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

        auto tmp = arr[indexLeftR + startIncluded];
        arr[stopIncluded] = tmp;
        arr[indexLeftR + startIncluded] = pivot;
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




// sorts n arrays at once
// to overlap more work in gpu pipelines
// also used for sorting bigger arrays quicker than a single sorter
// by merging independently sorted chunks
// merge sort ---> multiple threads ---> quick-sort ---> cuda kernel ---> odd-even-sort ---> cuda block
#include<thread>
#include<memory>
#include<mutex>
#include<queue>
#include<condition_variable>
#include<functional>
/* 
    N = number of concurrent sorts
    S = max number of elements per sort
*/
struct Sorter;
template<int N, int S>
struct SortingPipeline
{
    std::shared_ptr<Sorter> sorter[N];
    std::vector<std::thread> threads;
    std::queue<std::vector<unsigned int>> data;
    std::function<void(std::vector<unsigned int>)> curCallback;
    std::mutex mut;
    std::condition_variable cond;
    bool status;
    int msg;
    int available;
    SortingPipeline()
    {
        {
            std::unique_lock<std::mutex> lck(mut);
            status = true;
            available = N;
        }
        for (int i = 0; i < N; i++)
        {
            int k = i;
            sorter[i] = std::make_shared<Sorter>(S);
            threads.emplace_back([&,k](){
                bool work = true;
                std::function<void(std::vector<unsigned int>)> callback;
                while (work)
                {
                    std::vector<unsigned int> dataToSort;
                    {
                        std::unique_lock<std::mutex> lck(mut);    
                        
                        cond.notify_all();
                        cond.wait(lck, [&]() { return msg>0; });   
                        work = status;
                        if (msg > 0)
                        {
                            msg--;
                            
                            if (data.size() > 0)
                            {
                                dataToSort = data.front();
                                data.pop();
                                callback = curCallback;
                            }
                            cond.notify_all();
                        }
                    }

                    if (dataToSort.size() > 0)
                    {
                        sorter[k]->Push(dataToSort);     
                        sorter[k]->Run();
                        sorter[k]->Pop(dataToSort);
                        callback(dataToSort);
                        available++;
                    }
                }
                std::cout << "sorter thread shutting down" << std::endl;
            });
        }
    }

    void Sort(std::vector<unsigned int> dataToSort,std::function<void(std::vector<unsigned int>)> callback)
    {
        std::unique_lock<std::mutex> lck(mut);
        msg++;
        data.push(dataToSort);
        curCallback = callback;
        available--;
        cond.notify_all();
    }

    void RunFuncSynchronized(std::function<void(void)> func)
    {
        std::unique_lock<std::mutex> lck(mut);
        func();
    }

    void Wait()
    {
        while (true)
        {
            std::unique_lock<std::mutex> lck(mut);
            msg+=N;
            cond.notify_all(); 
            cond.wait(lck, [&]() { return available == N; });     
            
            if (available == N)
                return;            
        }
    }

    ~SortingPipeline()
    {
        {
            std::unique_lock<std::mutex> lck(mut);
            status = false;
            msg += N;
        }
        while(true)
        {
            std::unique_lock<std::mutex> lck(mut);
            cond.notify_all();
            cond.wait(lck, [&]() { return msg == 0; });
            if (msg == 0)
                break;
        }
        for (int i = 0; i < N; i++)
        {
            
            threads[i].join();
        }
    }
};
struct Sorter
{
    int n;
    unsigned int* data, * left, * right, * numTasks;
    int* tasks, * tasks2;
    cudaStream_t stream;
    Sorter(int numElementsMax)
    {
        cudaSetDevice(0);
        cudaStreamCreate(&stream);
        n = numElementsMax;
        cudaMallocHost(&data, n * sizeof(unsigned int));
        cudaMallocHost(&left, n * sizeof(unsigned int));
        cudaMallocHost(&right, n * sizeof(unsigned int));
        cudaMallocHost(&numTasks, 2 * sizeof(unsigned int));
        cudaMallocHost(&tasks, n * sizeof(int));
        cudaMallocHost(&tasks2, n * sizeof(int));
    }

    void Push(std::vector<unsigned int>& hostData)
    {
        unsigned int hostTasks[2];
        unsigned int numTasksHost[2];
        numTasksHost[0] = 1; // launch 1 block first
        numTasksHost[1] = 0;
        hostTasks[0] = 0;
        hostTasks[1] = hostData.size()-1; // first block's chunk limits: 0 - n-1
        cudaMemcpyAsync((void*)data, hostData.data(), hostData.size() * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync((void*)numTasks, numTasksHost, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync((void*)tasks, hostTasks, 2 * sizeof(int), cudaMemcpyHostToDevice, stream); // host only gives 1 task with 2 parameters
    }
    void Run()
    {
        qSortMain <<<1, 1,0, stream >>> (data, left, right, 0, numTasks, tasks, tasks2);
    }

    void Pop(std::vector<unsigned int>& hostData)
    {
        cudaMemcpyAsync(hostData.data(), (void*)data, hostData.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    ~Sorter()
    {
        cudaFreeHost(data);
        cudaFreeHost(left);
        cudaFreeHost(right);
        cudaFreeHost(tasks);
        cudaFreeHost(tasks2);
        cudaFreeHost(numTasks);
        cudaStreamDestroy(stream);
    }
};

// sorts 15 arrays, with 3 concurrency and max 8M elements per array
// in CUDA GPU (quick-sort algorithm)
void test()
{
    std::shared_ptr<SortingPipeline<3, 1024 * 1024*8>> pipeline = std::make_shared< SortingPipeline<3, 1024 * 1024*8>>();
    constexpr int n = 1024 * 1024*8;
    for (int j = 0; j < 15; j++)
    {
        std::vector<unsigned int> hostData(n);
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand();
        }
        pipeline->Sort(hostData,
            [n](std::vector<unsigned int> sorted)
            {
                bool err = false;
                for (int i = 0; i < n - 1; i++)
                    if (sorted[i] > sorted[i + 1])
                    {
                        std::cout << "error at: " << i << ": " << sorted[i] << std::endl;
                        err = true;
                        break;
                    }
                if (!err)
                {
                    std::cout << "quicksort completed successfully" << std::endl;
                }
            });
        
    }

    pipeline->Wait();
    
}

int main()
{
    test();

    return 0;
}
