#pragma once
#include <cmath>
#ifndef __host__
#define __host__
#define __device__
#endif
//#include <device_atomic_functions.hpp>
#define SIZE_T_BITS 64 //not that this is gonna change for me personally but oh well

extern unsigned long long atomicOr(unsigned long long* address, unsigned long long val);
extern unsigned long long atomicAnd(unsigned long long* address, unsigned long long val);
struct GPUBitset {
	size_t* data;
	//numbits is the size
	size_t data_size;
	size_t numBits;
	GPUBitset(size_t numBits, bool init): numBits(numBits) {
		float sizef = numBits / SIZE_T_BITS;
		data_size = ceil(sizef);
		data = (size_t*)malloc(sizeof(size_t) * data_size);
		size_t setvalue = init ? -1 : 0;
		memset(data, setvalue, sizeof(size_t) * data_size);
	}
	 ~GPUBitset() {
		free(data);
	}
	 void Reset(bool init) {
		free(data);
		data = (size_t*)malloc(sizeof(size_t) * data_size);
		size_t setvalue = init ? -1 : 0;
		memset(data, setvalue, sizeof(size_t) * data_size);
	}
	bool GetBitHost(size_t index) {
		//if (index > numBits) printf("FUCKY WUCKY");
		size_t data_loc =index / SIZE_T_BITS;
		size_t loc_offset = index % SIZE_T_BITS;
		//printf("%zu[%zu]\n", data, data_loc);
		return (data[data_loc] >> loc_offset) & 1;
	}
	__device__ bool GetBitDevice(size_t index,int id) {

		size_t data_loc = index / SIZE_T_BITS;
		size_t loc_offset = index % SIZE_T_BITS;
		return (data[data_loc] >> loc_offset) & 1;
	}
	void SetBitHost(size_t index) {
		size_t data_loc =index / SIZE_T_BITS;
		size_t loc_offset = index % SIZE_T_BITS;
		data[data_loc] = data[data_loc] | (size_t)1 <<loc_offset;
	}
	void ClearBitHost(size_t index) {
		size_t data_loc =index / SIZE_T_BITS;
		size_t loc_offset = index % SIZE_T_BITS;
		data[data_loc] = data[data_loc] & ~((size_t)1 << loc_offset);
	}
	__device__ void SetBitDevice(size_t index) {
		size_t data_loc =index / SIZE_T_BITS;
		size_t loc_offset = index % SIZE_T_BITS;
		atomicOr(&data[data_loc], (size_t)1 << loc_offset);//this should work idk
	}
	__device__ void ClearBitDevice(size_t index) {
		size_t data_loc =index / SIZE_T_BITS;
		size_t loc_offset = index % SIZE_T_BITS;
		atomicAnd(&data[data_loc], ~((size_t)1 << loc_offset));//this should work idk
	}

};