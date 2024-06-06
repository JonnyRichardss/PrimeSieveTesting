#include "SieveFactorsCUDA.cuh"


#include <sm_35_atomic_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

SieveFactorsCUDA::SieveFactorsCUDA(long long upTo) : PrimeSieve(upTo, "CUDA Factor based sieve"), bitArray(upTo, true)
{
}
__global__ void EliminateMultiples(long long upTo,int maxFactor,int* lowPrimes, size_t numLowPrimes, GPUBitset* bitArray) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //for a min we assume numBlocks * numThreads exceeds the number of low primes, this may not be true
    //TODO assign each prime a block and go up the multiples thread-wise (eg. thread 0 does factor^2 - thread 1 does factor^2 + 2 etc)
    if (id >= numLowPrimes) return;
    for (int factor = lowPrimes[id]; factor < maxFactor; factor += 2) {
        if (!bitArray->GetBitDevice(factor,id)) continue;//if its already a multiple of something else - skip
        for (int product = factor * factor; product < upTo; product += factor * 2) {
            bitArray->ClearBitDevice(product);//every odd multiple of factor is now set to false
        }
    }
}
int SieveFactorsCUDA::Calculate()
{
    //reset
    bitArray.Reset(true);
    lastResult = 1;//start by including #2
    int cpuMax = sqrt(upTo);
    int maxFactor = sqrt(cpuMax);
    std::vector<int> lowPrimes;
    for (int factor = 3; factor < maxFactor; factor += 2) {//only checking odd numbers
        if (!bitArray.GetBitHost(factor)) continue;//if its already a multiple of something else - skip
        for (int product = factor * factor; product < cpuMax; product += factor * 2) {
            bitArray.ClearBitHost(product);//every odd multiple of factor is now set to false
        }
    }
    for (int i = 3; i < cpuMax; i += 2) {
        if (bitArray.GetBitHost(i))lowPrimes.push_back(i);
    }
    //we can copy the bitArray and lowPrimes to the GPU
    //declare GPU ptrs
    size_t* arrayDataGPU;
    int* LowPrimesGPU;
    GPUBitset* bitArrayGPU;

    size_t dataGPUsize = bitArray.data_size * sizeof(size_t);
    size_t intsGPUsize = lowPrimes.size() * sizeof(int);
    size_t bitsGPUsize = sizeof(GPUBitset);

    cudaMalloc((void**) & arrayDataGPU, dataGPUsize);
    cudaMalloc((void**) & LowPrimesGPU, intsGPUsize);
    cudaMalloc((void**) & bitArrayGPU, bitsGPUsize);

    size_t* bitArrayHostData = bitArray.data;

    bitArray.data = arrayDataGPU;//this is heinous
        cudaMemcpy(arrayDataGPU, bitArrayHostData, dataGPUsize, cudaMemcpyHostToDevice);
        cudaMemcpy(LowPrimesGPU, &lowPrimes[0], intsGPUsize, cudaMemcpyHostToDevice);
        cudaMemcpy(bitArrayGPU, &bitArray, bitsGPUsize, cudaMemcpyHostToDevice);
    bitArray.data = bitArrayHostData;


    //do the checks on each of the lowPrimes in parallel

    EliminateMultiples << < 2048, 1024 >> > (upTo,sqrt(upTo),LowPrimesGPU,lowPrimes.size(),bitArrayGPU);
    //copy the bitArray back
    cudaMemcpy( bitArrayHostData, arrayDataGPU, dataGPUsize, cudaMemcpyDeviceToHost);
    cudaMemcpy( bitArrayGPU, &bitArray, bitsGPUsize, cudaMemcpyDeviceToHost);

    //free gpu stuff
    cudaFree(arrayDataGPU);
    cudaFree(LowPrimesGPU);
    cudaFree(bitArrayGPU);
    for (int i = 3; i < upTo; i += 2) {
        if (bitArray.GetBitHost(i)) lastResult++;
    }
    return lastResult;
}