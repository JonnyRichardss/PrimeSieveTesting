#pragma once
#include "PrimeSieve.h"
#include "GPUbitset.cuh"
class SieveFactorsCUDA : public PrimeSieve 
{
public:
	SieveFactorsCUDA(long long upTo);
	int Calculate();
protected:
	GPUBitset bitArray;
};

