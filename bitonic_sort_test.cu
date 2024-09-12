    // test codes for sorting 64M float keys
    // uses float array to sort in-place.
    // uses dynamic parallelism feature of cuda
    // array size needs to be integer power of 2
    // arary size needs to be at least 8192
    // repeats for 10 times and writes total time (and 100th element after sort that is 101)
    // benchmark data:
    
    /*
    Array elements  	GT1030		    		std::sort 	        GTX1080ti            		RTX4070(from nsight-compute profiler) 
			(benchmark)   			(1 core )             (guesstimate)
			(no overclock)
    1024		not applicable                  -
    2048		not applicable			-
    4096		not applicable			-
    8192		363 µs		  		114  µs		      -
    16k			463 us		  		248  µs		      -
    32k			746 us		 		536  µs		      -
    64k			1.23 ms		  		1.15 ms		      -
    128k		2.32 ms		  		2.46 ms		      -
    256k		4.87 ms		 		5.4  ms			~1.5+ 0.3	ms
    512k		8.72 ms		  		11.7 ms			~3	+ 0.5	ms
    1M			18.3 ms		  		22   ms			~6  + 1.2	ms
    2M			39 ms		  		48   ms			~12 + 2.7	ms
    4M			86 ms		  		101  ms			~23 + 6.3	ms
    8M			187 ms		  		211  ms			~47 + 14	ms
    16M			407 ms		  		451  ms			~95 + 32	ms
    32M			883 ms		  		940  ms			~190+ 70	ms
    64M			1.93 s		  		2.0  s		    	~380+ 150	ms            	119 ms kernel, 21 ms buffer copy (pinned buffers)
    (float keys)    (copy+kernel )			(copy + kernel)

    pcie v2.0 4x: 1.4GB/s
    fx8150 @ 3.6GHz
    4GB RAM 1333MHz
    (single channel DDR3)
    */
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <device_functions.h>

__global__ void kernel(char * data)
{
	if (threadIdx.x == 0)
		data[0] += 35;
}

// hello-world type test
int test()
{
    char* dvc;
	cudaMalloc<char>(&dvc, 1);
	char hst=30;
	cudaMemcpy(dvc, &hst, 1, ::cudaMemcpyHostToDevice);
	kernel<<<64, 64 >>> (dvc);	
	cudaMemcpy(&hst, dvc, 1, ::cudaMemcpyDeviceToHost);
	return hst;
}


// bitonic-sort
const int n = 67108864; // 64M elements
const int l2n = 26;  // log2(n)


// shared memory per block, also number of work per block (2048=minimum, 4096=moderate, 8192=maximum).
const int sharedSize = 8192;
const int l22k = 13; // log2(sharedSize)
__device__ void compareSwap(float& var1, float& var2, bool dir)
{
    if (var1 > var2 && dir)
    {
        float tmp = var1;
        var1 = var2;
        var2 = tmp;
    }
    else if (var1 < var2 && !dir)
    {
        float tmp = var1;
        var1 = var2;
        var2 = tmp;
    }
}
__global__ void computeBox(float* __restrict__ data, const int boxSize, const int leapSize)
{
    const int index = (threadIdx.x + blockIdx.x * blockDim.x);
    const bool dir = ((index % boxSize) < (boxSize / 2));
    const int indexOffset = (index / leapSize) * leapSize;

    compareSwap(data[index + indexOffset], data[index + indexOffset + leapSize], dir);
}
__global__ void computeBoxForward(float* __restrict__ data, const int boxSize, const int leapSize)
{
    const int index = (threadIdx.x + blockIdx.x * blockDim.x);
    const bool dir = true;
    const int indexOffset = (index / leapSize) * leapSize;

    compareSwap(data[index + indexOffset], data[index + indexOffset + leapSize], dir);
}
__device__ void computeBoxShared(float* __restrict__ data, const int boxSize, const int leapSize, const int work)
{
    const int index = threadIdx.x + work * 1024;
    const bool dir = ((index % boxSize) < (boxSize / 2));
    const int indexOffset = (index / leapSize) * leapSize;

    compareSwap(data[index + indexOffset], data[index + indexOffset + leapSize], dir);
}
__device__ void computeBoxForwardShared(float* __restrict__ data, const int boxSize, const int leapSize, const int work)
{
    const int index = threadIdx.x + work * 1024;
    const bool dir = true;
    const int indexOffset = (index / leapSize) * leapSize;

    compareSwap(data[index + indexOffset], data[index + indexOffset + leapSize], dir);
}
__global__ void bitonicSharedSort(float* __restrict__ data)
{
    const int offset = blockIdx.x * sharedSize;
    __shared__ float sm[sharedSize];
    const int nCopy = sharedSize / 1024;
    const int nWork = sharedSize / 2048;
    for (int i = 0; i < nCopy; i++)
    {
        sm[threadIdx.x + i * 1024] = data[threadIdx.x + offset + i * 1024];
    }
    __syncthreads();
    int boxSize = 2;
    for (int i = 0; i < l22k - 1; i++)
    {
        for (int leapSize = boxSize / 2; leapSize > 0; leapSize /= 2)
        {
            for (int work = 0; work < nWork; work++)
            {
                computeBoxShared(sm, boxSize, leapSize, work);
            }
            __syncthreads();
        }
        boxSize *= 2;
    }

    for (int leapSize = boxSize / 2; leapSize > 0; leapSize /= 2)
    {
        for (int work = 0; work < nWork; work++)
        {
            computeBoxForwardShared(sm, boxSize, leapSize, work);
        }
        __syncthreads();
    }

    for (int i = 0; i < nCopy; i++)
    {
        data[threadIdx.x + offset + i * 1024] = sm[threadIdx.x + i * 1024];
    }
}
__global__ void bitonicSharedMergeLeaps(float* __restrict__ data, const int boxSizeP, const int leapSizeP)
{
    const int offset = blockIdx.x * sharedSize;
    __shared__ float sm[sharedSize];
    const int nCopy = sharedSize / 1024;
    const int nWork = sharedSize / 2048;
    for (int i = 0; i < nCopy; i++)
    {
        sm[threadIdx.x + i * 1024] = data[threadIdx.x + offset + i * 1024];
    }
    __syncthreads();

    for (int leapSize = leapSizeP; leapSize > 0; leapSize /= 2)
    {
        for (int work = 0; work < nWork; work++)
        {
            const int index = threadIdx.x + work * 1024;
            const int index2 = threadIdx.x + work * 1024 + blockIdx.x * blockDim.x * nWork;
            const bool dir = ((index2 % boxSizeP) < (boxSizeP / 2));
            const int indexOffset = (index / leapSize) * leapSize;

            compareSwap(sm[index + indexOffset], sm[index + indexOffset + leapSize], dir);
        }
        __syncthreads();
    }

    for (int i = 0; i < nCopy; i++)
    {
        data[threadIdx.x + offset + i * 1024] = sm[threadIdx.x + i * 1024];
    }
}

// launch this with 1 cuda thread
// dynamic parallelism = needs something newer than cc v3.0

__global__ void bitonicSort(float* __restrict__ data)
{
    cudaStream_t stream0;
    cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
    bitonicSharedSort <<<(n / sharedSize), 1024,0,stream0 >>> (data);
    int boxSize = sharedSize;
    for (int i = l22k - 1; i < l2n - 1; i++)
    {
        if (boxSize > sharedSize)
        {
            int leapSize = boxSize / 2;
            for (; leapSize > sharedSize / 2; leapSize /= 2)
            {
                computeBox <<<(n / 1024) / 2, 1024, 0, stream0 >>> (data, boxSize, leapSize);
            }
            bitonicSharedMergeLeaps <<<(n / sharedSize), 1024, 0, stream0 >>> (data, boxSize, leapSize);
        }
        else
        {
            bitonicSharedMergeLeaps <<<(n / sharedSize), 1024, 0, stream0 >>> (data, boxSize, sharedSize / 2);
        }
        boxSize *= 2;
    }


    for (int leapSize = boxSize / 2; leapSize > 0; leapSize /= 2)
    {
        computeBoxForward <<<(n / 1024) / 2, 1024, 0, stream0 >>> (data, boxSize, leapSize);
    }

    cudaEvent_t event0;
    cudaEventCreateWithFlags(&event0, cudaEventBlockingSync);
    cudaEventRecord(event0, stream0);
    cudaStreamWaitEvent(stream0, event0, 0);
    cudaStreamDestroy(stream0);
}

#include<vector>
#include<iostream>
void test2()
{
    float* dvc;
    cudaMalloc(&dvc, n * sizeof(float));

    float* hst;
    auto cerr = cudaHostAlloc(&hst, n * sizeof(float), cudaHostAllocDefault);
    std::cout << cerr << std::endl;
 
    for (int i = 0; i < n; i++)
        hst[i] = n - i;
  
    cudaStream_t stream0;
    cudaStreamCreate(&stream0);

    cudaEvent_t evt,evt2;

    cudaEventCreate(&evt);
    cudaEventCreate(&evt2);

    cudaEventRecord(evt, stream0);
    
    for (int i = 0; i < 10; i++)
    {

        cudaMemcpyAsync(dvc, hst, n * sizeof(float), ::cudaMemcpyHostToDevice, stream0);
        bitonicSort <<<1, 1,0, stream0 >>> (dvc);
        cudaMemcpyAsync(hst, dvc, n * sizeof(float), ::cudaMemcpyDeviceToHost, stream0);
    }
    
    cudaEventRecord(evt2, stream0);
    cudaEventSynchronize(evt2);
    float tim;
    cudaEventElapsedTime(&tim, evt, evt2);

    std::cout << hst[100]<<"  "<<tim<< std::endl;
    
    cudaFreeHost(hst);
    cudaFree(dvc);
    
}

int main()
{
    test2();
	return 0;
}
