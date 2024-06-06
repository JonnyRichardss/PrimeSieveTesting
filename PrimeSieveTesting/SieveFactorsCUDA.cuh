#pragma once
#include "PrimeSieve.h"
struct GPUBitset;
class SieveFactorsCUDA : public PrimeSieve 
{
public:
	SieveFactorsCUDA(long long upTo);
	int Calculate();
protected:
	GPUBitset* bitArray;
};

