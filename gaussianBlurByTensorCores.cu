// 220 milliseconds for generating 16000 images from 1 image (16 different gaussian blur patterns for 1000 times)
// this is equivalent to 13.7 microseconds per image (1024x1024)
// note: not finished yet. half of output is empty or opencv bugs when too many windows are created

// tensor-core gaussian-blur for generating 16 different gaussian blurred pixels of an image at once
// for example, input is an image, output is 16 images each with a different blur strength
// tensor matrix-A row is 1 pixel's neighbor pixel data
// tensor matrix-B row is made of gaussian blur multipliers
// first row of result has 16 different blur versions of first pixel
// 15 more rows for 16 total pixels. Every warp computes 16 pixels at once, possibly having 4x TFLOPS than CUDA-only version
// For every tensor-using CUDA block, there are also 4 blocks of pure-CUDA computing gaussian blur algorithms to add CUDA core performance on top of tensor
// for every tensor-using CUDA block, there are also 8 block of pure-CUDA integer-only path that adds on top of float + tensor performance
// possibly bandwidth-bottlenecked
// possibly has more error due to bias from integer+float+tensor mixed work & only 16-bit precision
// but it should be faster than either of them

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <device_functions.h>
#include <mma.h>
// square image for simplicity
constexpr int imageSize = 1024;

// square tile for simplicity
constexpr int tileSize = 4;

constexpr int imagePixels = imageSize * imageSize;
constexpr int tilePixels = tileSize * tileSize;
constexpr int tilesPerDimension = imageSize / tileSize;

// neighbor data size to load when tiling in shared memory (4x4 requires a 6x6 region)
// for simplicity, only interior of image is computed. borders are left alone for now
constexpr int tileAccessSize = tileSize+2;
constexpr int tileAccessPixels = tileAccessSize * tileAccessSize;

// 1 warp per block for now
__global__ void superGaussPipeline(
    half* image, half* gaussianMultipliers, half* outputImages
)
{
    
    // every warp is working independently on different tile. so tile id = warp id
    const int indexWarp = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int indexTileX = indexWarp & (tilesPerDimension - 1); // still faster than modulo?
    const int indexTileY = indexWarp / tilesPerDimension; 

    // skip border tiles for simplicity for now
    if (indexTileX == 0 || indexTileX >= tilesPerDimension - 1 || indexTileY == 0 || indexTileY >= tilesPerDimension - 1)
        return;


    const int indexLane = (threadIdx.x & (warpSize - 1));

    // top-left corner of tile that is 6x6 sized
    // loading consists of reading 6 rows starting with indexTile, each 6 columns    
    // load 36 pixels using 32 threads ==> some threads need more iterations
    const int indexTile = (indexTileX*tileSize - 1) + (indexTileY * tileSize - 1) * imageSize;
    const int stepsRequired = 1 + (tileAccessPixels / warpSize);

    alignas(64)
    __shared__ half alignedAccess[256];
    // 6x6 pixels (to compute closest neighbor based Gaussian Blur for 4x4 interior pixels)

    // doesn't require alignment because no tensor access this
    __shared__ half tileAccess[tileAccessPixels];

    // there should be a more efficient loading mechanism
    // simply loading 1 row at a time
    for(int i=0;i<tileAccessSize;i++)
    {
        if (indexLane < tileAccessSize)
        {
            tileAccess[i * tileAccessSize + indexLane] =  image[indexTile + i * imageSize + indexLane];
        }
    }
    __syncthreads();
    
    // map neighbor pixel data (9 per pixel) to tensor core matrix row (each row = neighbors of a pixel, half of elements are zero)
    // map gaussian multipliers to tensor core matrix row
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> neighborPixels;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> gaussian;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> result;
    // mapping
    if (indexLane < tilePixels)
    {
        
        const int ix = indexLane & (tileSize - 1);
        const int iy = indexLane / tileSize;


        int ctr = 0;
        const int accessX = ix + 1;
        const int accessY = iy + 1;
        for (int jy = -1; jy <= 1; jy++)
            for (int jx = -1; jx <= 1; jx++)
            {
                alignedAccess[ctr + indexLane * 16] = tileAccess[accessX + jx + (accessY + jy) * tileAccessSize];
                ctr++;
            }
      

        for (int k = 0; k < 7; k++)
            alignedAccess[9 + k + indexLane * 16] = 0;
        
    }
    __syncthreads();
    
    // now mapped neighbor data can be loaded
    nvcuda::wmma::load_matrix_sync(neighborPixels, alignedAccess, 16);


    // also load 16 different gaussian blur multiplier sets (1 per row, 9 elements filled, rest are zeroed) are loaded
    nvcuda::wmma::load_matrix_sync(gaussian, gaussianMultipliers, 16);
    
    // initialize results to zero
    nvcuda::wmma::fill_fragment(result, 0.0f);
    
    // 16 gaussian blur operations per pixel are computed at once
    // each result of a pixel is given in a row
    // 16 different pixels in 16 rows
    nvcuda::wmma::mma_sync(result, neighborPixels, gaussian, result);
    __syncthreads();
    
    // store results back to shared memory
    nvcuda::wmma::store_matrix_sync(alignedAccess, result, 16, nvcuda::wmma::mem_col_major);
   
    __syncthreads(); // because shared memory was written?
    
    // distribute the result to 16 images each having a different gaussian blur strength/pattern
    if (indexLane < 16)
    {
        const int resultSubTileX = indexLane & (tileSize - 1);
        const int resultSubTileY = indexLane / tileSize;

        // iterate output images
        
        for (int i = 0; i < 16; i++)
            outputImages[   i * imagePixels +
                            (indexTileX * tileSize + resultSubTileX) +
                            (indexTileY * tileSize + resultSubTileY) * imageSize]  // i iterates gauss version, tileX & tileY iterate 4x4 pixels of tile
            =             
            alignedAccess[i + indexLane*16]; // i iterates gauss version, indexLane iterates pixel
    }
}

#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>

void test()
{
   
    auto img0 = cv::imread("test.jpg",cv::ImreadModes::IMREAD_COLOR);
    cv::Mat img;
    cv::cvtColor(img0,img, cv::COLOR_BGR2GRAY);
    cv::namedWindow("input");
    cv::resizeWindow("input", cv::Size2i(1024, 1024));
    cv::imshow("input", img);
    while (cv::waitKey() != 27)
    {
        
    }
    

    // elements for 1M matrices of size 16x16
    constexpr int inputN = imagePixels * sizeof(half);
    constexpr int gaussN = 16 * 16 * sizeof(half);
    constexpr int outputN = inputN * 16;
    half* dvcA, * dvcB, * dvcC;
    cudaMalloc(&dvcA, inputN);
    cudaMalloc(&dvcB, gaussN);
    cudaMalloc(&dvcC, outputN);

    half* hstA, * hstB, * hstC;
    cudaHostAlloc(&hstA, inputN, cudaHostAllocDefault);
    cudaHostAlloc(&hstB, gaussN, cudaHostAllocDefault);
    cudaHostAlloc(&hstC, outputN, cudaHostAllocDefault);

    for (int i = 0; i < imagePixels; i++)
        hstA[i] = img.at<uchar>(i)/256.0f;

    for (int i = 0; i < 16 * 16; i++)
    {
        hstB[i * 16] = 1;
        hstB[i * 16 + 1] = 2;
        hstB[i * 16 + 2] = 1;
        hstB[i * 16 + 3] = 2;
        hstB[i * 16 + 4] = 4;
        hstB[i * 16 + 5] = 2;
        hstB[i * 16 + 6] = 1;
        hstB[i * 16 + 7] = 2;
        hstB[i * 16 + 8] = 1;
        hstB[i * 16 + 9] = 0.0f; 
        hstB[i * 16 + 10] = 0.0f;
        hstB[i * 16 + 11] = 0.0f;
        hstB[i * 16 + 12] = 0.0f;
        hstB[i * 16 + 13] = 0.0f;
        hstB[i * 16 + 14] = 0.0f;
        hstB[i * 16 + 15] = 0.0f;
    }
    // read image by opencv
    std::cout << "tensor blur" << std::endl;

    cudaStream_t stream0;
    cudaStreamCreate(&stream0);


    // warm-up 
    int numWarpsToLaunch = imagePixels / tilePixels;
    for (int i = 0; i < 1000; i++)
    {
        superGaussPipeline <<<numWarpsToLaunch, 32, 0, stream0 >>> (dvcA, dvcB, dvcC);

    }
    cudaEvent_t evt, evt2;
    auto err = cudaEventCreate(&evt);
    if (err)
    {
        std::cout << "Error code:" << err << std::endl;
    }
    cudaEventCreate(&evt2);
    cudaEventRecord(evt, stream0);
    


    cudaMemcpyAsync(dvcA, hstA, inputN, ::cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(dvcB, hstB, gaussN, ::cudaMemcpyHostToDevice, stream0);
    for (int i = 0; i < 1000; i++)
    {
        superGaussPipeline <<<numWarpsToLaunch, 32, 0, stream0 >>> (dvcA, dvcB, dvcC);
    }
    err=cudaMemcpyAsync(hstC, dvcC, outputN, ::cudaMemcpyDeviceToHost, stream0);
    if (err)
    {
        std::cout <<"Error code:" << err << std::endl;
    }
    cudaEventRecord(evt2, stream0);
    cudaEventSynchronize(evt2);
    float tim;
    cudaEventElapsedTime(&tim, evt, evt2);
    std::cout << "generating 16 images for 1000 times: " << tim << " ms" << std::endl;
    
    // opencv display images
    // division by 16 for normalization
    for (int k = 0; k < 16; k++)
    {
        for (int i = 0; i < imagePixels; i++)
            img.at<uchar>(i) = (((float)hstC[i+k*imagePixels]) / 16.0f) * 256;
        cv::waitKey(100);
        cv::namedWindow(std::string("output") + std::to_string(k));
        cv::resizeWindow(std::string("output") + std::to_string(k), cv::Size2i(1024, 1024));
        cv::imshow(std::string("output") + std::to_string(k), img);

    }
    while (cv::waitKey() != 27)
    {

    }
    cv::destroyAllWindows();
    cudaFreeHost(hstA);
    cudaFree(dvcA);

}

int main()
{
    test();
    return 0;
}
