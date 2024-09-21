// hybrid quicksort
// when chunk size is greater than 1024, it does quicksort steps
// continues splitting chunks like: left, middle(just count), right
// when chunk size is 1024 or less, executes parallel odd-even sort (todo: shear-sort 1024 + network 32)

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


// task pattern: 
//              task 0      task 1      task 2      task 3      ---> array chunks to sort (no overlap)
//              start stop  start stop  start stop  start stop  ---> tasks buffer
//              block 0     block 1     block 2     block 3     ---> cuda blocks
__global__ void quickSortWithoutStreamCompaction(
    int* __restrict__ arr, int* __restrict__ left, int* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4);

__global__ void bruteSort(int* __restrict__ arr, int* __restrict__ left, int* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4);

__global__ void resetNumTasks(int* __restrict__ data, int* __restrict__ left, int* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4)
{
    if (threadIdx.x == 0)
    {
        numTasks[2] = numTasks[0];
        numTasks[3] = numTasks[1];
        numTasks[0] = 0;
        numTasks[1] = 0;
        __syncthreads();
        
        if (numTasks[3] > 0)
            bruteSort <<<numTasks[3], 128, 0, cudaStreamTailLaunch >>> (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);
       
        if (numTasks[2] > 0)
            quickSortWithoutStreamCompaction <<<numTasks[2], 1024, 0, cudaStreamTailLaunch >>> (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);

    }
}

__global__ void copyTasksBack(int* __restrict__ data, int* __restrict__ left, int* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4)
{
    const int id = threadIdx.x;
    const int n = numTasks[0];
    const int n2 = numTasks[1];
    const int steps = 1 + n / 1024;
    const int steps2 = 1 + n2 / 1024;


    // make quick-sort tasks usable
    for (int i = 0; i < steps; i++)
    {
        const int curId = id + i * 1024;
        if (curId < n)
        {            
            tasks[curId * 2] = tasks2[curId * 2];
            tasks[curId * 2 + 1] = tasks2[curId * 2 + 1];
        }
    }


    // make brute-force tasks usable
    for (int i = 0; i < steps2; i++)
    {
        const int curId = id + i * 1024;
        if (curId < n2)
        {
            tasks3[curId * 2] = tasks4[curId * 2];
            tasks3[curId * 2 + 1] = tasks4[curId * 2 + 1];
        }
    }

    if (id == 0)
    {
        
        resetNumTasks <<<1, 1, 0, cudaStreamTailLaunch >>> (data, left, right, numTasks, tasks, tasks2, tasks3,tasks4);
    }

}


#define compareSwap(a,x,y) if(a[y]<a[x]){a[x]^=a[y];a[y]^=a[x];a[x]^=a[y];}

__global__ void bruteSort(int* __restrict__ arr, int* __restrict__ left, int* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4)
{
  
    const int id = threadIdx.x;
    const int gid = blockIdx.x;
    __shared__ int taskIdCacheStart;
    __shared__ int taskIdCacheStop;
    if (id == 0)
    {
        taskIdCacheStart = tasks3[gid * 2];
        taskIdCacheStop = tasks3[gid * 2 + 1];
        tasks3[gid * 2] = 0;
        tasks3[gid * 2 + 1] = 0;
    }
    __syncthreads();
    const int startIncluded = taskIdCacheStart;
    const int stopIncluded = taskIdCacheStop;
    const int num = stopIncluded - startIncluded + 1;
    if (startIncluded == 0 && stopIncluded == 0)
    {
        if(id == 0)
            printf("\n brute-force task id error: %i \n",gid);
        return;
    }
    
    __shared__ int cache[128];
    if (startIncluded + id <= stopIncluded)
    {
        cache[id] = arr[startIncluded+id];
    }
    __syncthreads();

    
    for (int i = 0; i < num; i++)
    {
        if (id +1< num)
        {
            if ((id % 2 == 0))
            {
                compareSwap(cache, id, id+1)
            }
        }
        __syncthreads();
        if (id +1 < num)
        {
            if ((id % 2 == 1))
            {
                compareSwap(cache, id, id + 1)
            }
        }
        __syncthreads();
    }
    

    if (startIncluded + id <= stopIncluded)
    {
        arr[startIncluded + id]= cache[id];
    }
}

// task pattern: 
//              task 0      task 1      task 2      task 3      ---> array chunks to sort (no overlap)
//              start stop  start stop  start stop  start stop  ---> tasks buffer
//              block 0     block 1     block 2     block 3     ---> cuda blocks
__global__ void quickSortWithoutStreamCompaction(
    int* __restrict__ arr, int* __restrict__ left, int* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4)
{
    // 1 block = 1 chunk of data
    const int gid = blockIdx.x;
    const int id = threadIdx.x;
   
    if(id == 0 && gid == 0)
        copyTasksBack <<<1, 1024, 0, cudaStreamTailLaunch >>> (arr, left, right, numTasks, tasks, tasks2, tasks3, tasks4);


    __shared__ int taskIdCacheStart;
    __shared__ int taskIdCacheStop;
    if (id == 0)
    {
        taskIdCacheStart = tasks[gid * 2];
        taskIdCacheStop = tasks[gid * 2 + 1];
        tasks[gid * 2] = 0;
        tasks[gid * 2 + 1] = 0;
    }
    __syncthreads();
    const int startIncluded = taskIdCacheStart;
    const int stopIncluded = taskIdCacheStop;
    const int num = stopIncluded - startIncluded + 1;


    if (startIncluded == 0 && stopIncluded == 0)
    {
        if(id==0)
            printf("\n quicksort task id error: %i \n", gid);
        return;
    }
        
    __shared__ int indexLeft;
    __shared__ int indexMid;
    __shared__ int indexRight;


    const int bd = blockDim.x;
    const int pivot = arr[stopIncluded];


    int nLeft = 0;
    int nMid = 0;
    int nRight = 0;
    if (id == 0)
    {
        indexLeft = 0;
        indexMid = 0;
        indexRight = 0;
    }
    __syncthreads();
    
    const int stepsArray = (num / bd) + 1;
    for (int i = 0; i < stepsArray; i++)
    {
        const int curId = i * bd + id;
        if (curId < num)
        {
            const auto data = arr[curId + startIncluded];
            if (data < pivot)
                left[startIncluded + atomicAdd(&indexLeft, 1)] = data;
            else if (data > pivot)
                right[startIncluded + atomicAdd(&indexRight, 1)] = data;
            else
                atomicAdd(&indexMid, 1); // this is a counting-sort-like optimization for one of worst-cases
        }
    }
    

    __syncthreads();
    nLeft = indexLeft;
    nMid = indexMid;
    nRight = indexRight;

    // move left
    const int stepsLeft = (nLeft / bd) + 1;
    for (int i = 0; i < stepsLeft; i++)
    {
        const int curId = i * bd + id;
        if (curId < nLeft)
        {
            arr[curId + startIncluded] = left[startIncluded + curId];

        }
    }
    

    // move mid
    const int stepsMid = (nMid / bd) + 1;
    for (int i = 0; i < stepsMid; i++)
    {
        const int curId = i * bd + id;
        if (curId < nMid)
        {
            arr[curId + startIncluded+nLeft] = pivot;

        }
    }
    

    
    // move right
    const int stepsRight = (nRight / bd) + 1;
    for (int i = 0; i < stepsRight; i++)
    {
        const int curId = i * bd + id;
        if (curId< nRight)
        {
            arr[curId + startIncluded + nLeft + nMid] = right[startIncluded + curId];
        }
    }
    
    __syncthreads();

    if (nLeft + nRight + nMid != num)
        printf(" @@ ERROR: wrong partition values @@");
    if (id == 0)
    {
        // push new "quick" task
        if (nLeft > 1)
        {            
            if (nLeft <= 128) // push new "brute-force" task
            {
                const int index = atomicAdd(&numTasks[1], 1);
                tasks4[index * 2] = startIncluded;
                tasks4[index * 2 + 1] = startIncluded + nLeft-1;
            }
            else// push new "quick" task
            {
                const int index = atomicAdd(&numTasks[0], 1);
                tasks2[index * 2] = startIncluded;
                tasks2[index * 2 + 1] = startIncluded + nLeft-1;
            }            
        }
        

        
        if (nRight > 1)
        {

            if (nRight <= 128) // push new "brute-force" task
            {
                const int index = atomicAdd(&numTasks[1], 1);
                tasks4[index * 2] = stopIncluded-nRight+1;
                tasks4[index * 2 + 1] = stopIncluded;
            }
            else // push new "quick" task
            {
                const int index = atomicAdd(&numTasks[0], 1);
                tasks2[index * 2] = stopIncluded - nRight+1;
                tasks2[index * 2 + 1] = stopIncluded;
            }
            
        }
    }
}

__global__ void resetTasks(int * tasks, int * tasks2, int * tasks3, int * tasks4, const int n)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        tasks[id] = 0;
        tasks2[id] = 0;
        tasks3[id] = 0;
        tasks4[id] = 0;
    }
}

__global__ void quickSortMain(
    int n,
    int* __restrict__ data, int* __restrict__ left, int* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4)
{
    
    int nQuickTask = 1;
    int nBruteTask = 0;
    tasks2[0] = 0;
    tasks2[1] = n - 1;
    __syncthreads();
    int ctr = 0;
    cudaStream_t stream0;
    cudaStreamCreateWithFlags(&stream0,(unsigned int) cudaStreamNonBlocking);

    while (true)
    {
        if (ctr > 1000000)
            break;
        copyTasksBack << <1, 1024,0,stream0 >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);
        resetNumTasks << <1, 1, 0, stream0 >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);
        
        if (nQuickTask > 0)
            quickSortWithoutStreamCompaction << <nQuickTask, 1024, 0, stream0 >> > (data, left, right, numTasks, tasks, tasks2,tasks3, tasks4);
        if (nBruteTask > 0)
            bruteSort <<<nBruteTask, 128, 0, stream0 >>> (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);
        cudaEvent_t event0;
        cudaEventCreateWithFlags(&event0, cudaEventBlockingSync);
        cudaEventRecord(event0, stream0);
        cudaStreamWaitEvent(stream0, event0, 0);
        cudaEventDestroy(event0);
     
        nQuickTask = numTasks[0];
        nBruteTask = numTasks[1];
        printf(" %i %i \n", nQuickTask, nBruteTask);

        if (nQuickTask + nBruteTask <= 0)
            break;
    }
    cudaStreamDestroy(stream0);
}
void test()
{
    constexpr int n = 1024*1024*4;
    int* data, * left, * right, * numTasks;
    int* tasks, * tasks2,*tasks3,*tasks4;
    std::vector< int> hostData(n),backup(n);
    std::vector<int> hostTasks(2);
   
    gpuErrchk( cudaSetDevice(0));
    gpuErrchk( cudaDeviceSynchronize());
    gpuErrchk( cudaMalloc(&data, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&left, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&right, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&numTasks, 4 * sizeof(int)));
    gpuErrchk( cudaMalloc(&tasks, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&tasks2, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&tasks3, n * sizeof(int)));
    gpuErrchk( cudaMalloc(&tasks4, n * sizeof(int)));
    resetTasks << <1 + n / 1024, 1024 >> > (tasks, tasks2, tasks3, tasks4, n);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int numTasksHost[2];
    int nQuickTask = 1;
    int nBruteTask =0;
        
    for (int j = 0; j < 100; j++)
    {
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand();//n-i; //rand();
            backup[i] = hostData[i];
        }
        auto qSort = [&]() {

            numTasksHost[0] = 1; // launch 1 block first
            numTasksHost[1] = 0;
            numTasksHost[2] = 1; // launch 1 block first
            numTasksHost[3] = 0;
            hostTasks[0] = 0;
            hostTasks[1] = n - 1; // first block's chunk limits: 0 - n-1
            gpuErrchk(cudaMemcpy((void*)data, hostData.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy((void*)numTasks, numTasksHost, 4 * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy((void*)tasks2, hostTasks.data(), 2 * sizeof(int), cudaMemcpyHostToDevice)); // host only gives 1 task with 2 parameters
            nQuickTask = 1;
            nBruteTask = 0;
            int ctr = 0;

            //quickSortMain<<<1,1>>>(n,data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);
            copyTasksBack << <1, 1024>> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);
            gpuErrchk(cudaDeviceSynchronize());
            if(false)
            while (nQuickTask > 0 || nBruteTask > 0)
            {
                if (!(nQuickTask > 0 || nBruteTask > 0))
                    break;
                gpuErrchk(cudaMemcpy(numTasksHost, (void*)numTasks, 2 * sizeof(int), cudaMemcpyDeviceToHost));
                //gpuErrchk(cudaMemcpy(hostTasks.data(), (void*)tasks, 2 * sizeof(int), cudaMemcpyDeviceToHost));
                nQuickTask = numTasksHost[0];
                nBruteTask = numTasksHost[1];



                copyTasksBack << <1, 1024 >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);
                
                resetNumTasks << <1, 1 >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);
                


                gpuErrchk(cudaDeviceSynchronize());


                //std::cout << "n=" << nQuickTask << " m=" << nBruteTask << "        t1=" << hostTasks[0] << " t2=" << hostTasks[1] << std::endl;

                //qSortMain << <1, 1 >> > (data, left, right, 0, numTasks, tasks, tasks2);
                if (nQuickTask > 0)
                    quickSortWithoutStreamCompaction <<<nQuickTask, 1024 >>> (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);
                gpuErrchk(cudaGetLastError());
             
                if (nBruteTask > 0)
                    bruteSort <<<nBruteTask, 128 >>> (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());

                
              
            }
            gpuErrchk(cudaMemcpy(hostData.data(), (void*)data, n * sizeof(int), cudaMemcpyDeviceToHost));
        };

        qSort();
        bool err = false,err2=false;
        for (int i = 0; i < n - 2; i++)
            if (hostData[i] > hostData[i + 1])
            {
                std::cout << "error at: " << i << ": " << hostData[i] << " " << hostData[i + 1] << " " << hostData[i + 2] << std::endl;
                err = true;
                j = 1000000;
                // re-testing with same input:
                std::cout << "re-testing with exactly same input elements:" << std::endl;
                for (int i = 0; i < n; i++)
                {
                    hostData[i] = backup[i];
                }
                qSort();
                err = false;
                for (int i = 0; i < n - 2; i++)
                    if (hostData[i] > hostData[i + 1])
                    {
                        std::cout << "Error happened again!" << std::endl;
                        err = true;
                        
                        break;
                    }

                if (!err)
                {
                    std::cout << "quicksort completed successfully with same input!!!" << std::endl;
                    // for (int i = 0; i < 35; i++)
                     //    std::cout << hostData[i] << " ";
                    err2 = true;
                }
                break;
            }

        if (!err && !err2)
        {
            std::cout << "quicksort completed successfully " << j << std::endl;
            // for (int i = 0; i < 35; i++)
             //    std::cout << hostData[i] << " ";
        }
    }

    gpuErrchk(cudaFree(data));
    gpuErrchk(cudaFree(left));
    gpuErrchk(cudaFree(right));
    gpuErrchk(cudaFree(tasks));
    gpuErrchk(cudaFree(tasks2));
    gpuErrchk(cudaFree(tasks3));
    gpuErrchk(cudaFree(tasks4));
    gpuErrchk(cudaFree(numTasks));

}

int main()
{
    test();

    return 0;
}
