#include "SieveFactorsCUDA.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
//working with multiple CUDA files is a nightmare so this is getting manually #included
#define NUM_BLOCKS 4096
#define SIZE_T_BITS 64 //not that this is gonna change for me personally but oh well
struct GPUBitset {
    size_t* data;
    //numbits is the size
    size_t data_size;
    size_t numBits;
    GPUBitset(size_t numBits, bool init) : numBits(numBits) {
        float sizef = numBits / SIZE_T_BITS;
        data_size = ceil(sizef);
        //data = (size_t*)(sizeof(size_t) * data_size);
        cudaMallocHost((void**)&data, sizeof(size_t) * data_size);
        size_t setvalue = init ? -1 : 0;
        memset(data, setvalue, sizeof(size_t) * data_size);
    }
    ~GPUBitset() {
        free(data);
    }
    void Reset(bool init) {
        cudaFreeHost(data);
        //data = (size_t*)malloc(sizeof(size_t) * data_size);
        cudaMallocHost((void**)&data,sizeof(size_t) * data_size);
        size_t setvalue = init ? -1 : 0;
        memset(data, setvalue, sizeof(size_t) * data_size);
    }
    bool GetBitHost(size_t index) {
        if (index > numBits) printf("FUCKY WUCKY");
        size_t data_loc = index / SIZE_T_BITS;
        size_t loc_offset = index % SIZE_T_BITS;
        //printf("%zu[%zu]\n", data, data_loc);
        return (data[data_loc] >> loc_offset) & 1;
    }
    __device__ bool GetBitDevice(size_t index) {

        size_t data_loc = index / SIZE_T_BITS;
        size_t loc_offset = index % SIZE_T_BITS;
        return (data[data_loc] >> loc_offset) & 1;
    }
    void SetBitHost(size_t index) {
        size_t data_loc = index / SIZE_T_BITS;
        size_t loc_offset = index % SIZE_T_BITS;
        data[data_loc] = data[data_loc] | (size_t)1 << loc_offset;
    }
    void ClearBitHost(size_t index) {
        size_t data_loc = index / SIZE_T_BITS;
        size_t loc_offset = index % SIZE_T_BITS;
        data[data_loc] = data[data_loc] & ~((size_t)1 << loc_offset);
    }
    __device__ void SetBitDevice(size_t index) {
        size_t data_loc = index / SIZE_T_BITS;
        size_t loc_offset = index % SIZE_T_BITS;
        atomicOr(&data[data_loc], (size_t)1 << loc_offset);//this should work idk
    }
    __device__ void ClearBitDevice(size_t index) {
        size_t data_loc = index / SIZE_T_BITS;
        size_t loc_offset = index % SIZE_T_BITS;
        atomicAnd(&data[data_loc], ~((size_t)1 << loc_offset));//this should work idk
    }

};
SieveFactorsCUDA::SieveFactorsCUDA(long long upTo) : PrimeSieve(upTo, "CUDA Factor based sieve")
{
    bitArray = new GPUBitset(upTo, true);
}
__global__ void EliminateMultiples(long long upTo, int maxFactor, int* lowPrimes, size_t numLowPrimes, GPUBitset* bitArray) {
    //TODO assign blocks in 2d
    if (blockIdx.x >= numLowPrimes) return;
    for (int primeToCheck = blockIdx.x; primeToCheck < numLowPrimes; primeToCheck+= NUM_BLOCKS) {
        int factor = lowPrimes[primeToCheck];
    if (!bitArray->GetBitDevice(factor)) return;//if its already a multiple of something else - skip
    for (int product = (factor * factor) + factor * threadIdx.x; product < upTo; product += factor * blockDim.x) {
        bitArray->ClearBitDevice(product);//every odd multiple of factor is now set to false
    }
}
    
}
int SieveFactorsCUDA::Calculate()
{
    //reset
    bitArray->Reset(true);
    lastResult = 1;//start by including #2
    int cpuMax = sqrt(upTo);
    int maxFactor = sqrt(cpuMax);
    std::vector<int> lowPrimes;
    for (int factor = 3; factor < maxFactor; factor += 2) {//only checking odd numbers
        if (!bitArray->GetBitHost(factor)) continue;//if its already a multiple of something else - skip
        for (int product = factor * factor; product < cpuMax; product += factor * 2) {
            bitArray->ClearBitHost(product);//every odd multiple of factor is now set to false
        }
    }
    for (int i = 3; i < cpuMax; i += 2) {
        if (bitArray->GetBitHost(i))lowPrimes.push_back(i);
    }
    //we can copy the bitArray and lowPrimes to the GPU
    //declare GPU ptrs
    size_t* arrayDataGPU;
    int* LowPrimesGPU;
    GPUBitset* bitArrayGPU;

    size_t dataGPUsize = bitArray->data_size * sizeof(size_t);
    size_t intsGPUsize = lowPrimes.size() * sizeof(int);
    size_t bitsGPUsize = sizeof(GPUBitset);

    cudaMalloc((void**) & arrayDataGPU, dataGPUsize);
    cudaMalloc((void**) & LowPrimesGPU, intsGPUsize);
    cudaMalloc((void**) & bitArrayGPU, bitsGPUsize);

    size_t* bitArrayHostData = bitArray->data;

    bitArray->data = arrayDataGPU;//this is heinous
        cudaMemcpy(arrayDataGPU, bitArrayHostData, dataGPUsize, cudaMemcpyHostToDevice);
        //cudaMemset(arrayDataGPU, 0, dataGPUsize);
        cudaMemcpy(LowPrimesGPU, &lowPrimes[0], intsGPUsize, cudaMemcpyHostToDevice);
        cudaMemcpy(bitArrayGPU, bitArray, bitsGPUsize, cudaMemcpyHostToDevice);
    bitArray->data = bitArrayHostData;


    //do the checks on each of the lowPrimes in parallel

    EliminateMultiples << < NUM_BLOCKS, 1024 >> > (upTo,sqrt(upTo),LowPrimesGPU,lowPrimes.size(),bitArrayGPU);
    //copy the bitArray back
    cudaMemcpy( bitArrayHostData, arrayDataGPU, dataGPUsize, cudaMemcpyDeviceToHost);
    cudaMemcpy( bitArrayGPU, bitArray, bitsGPUsize, cudaMemcpyDeviceToHost);

    //free gpu stuff
    cudaFree(arrayDataGPU);
    cudaFree(LowPrimesGPU);
    cudaFree(bitArrayGPU);
    for (size_t i = 3; i < upTo; i += 2) {
        if (bitArray->GetBitHost(i)) lastResult++;
    }
    return lastResult;
}