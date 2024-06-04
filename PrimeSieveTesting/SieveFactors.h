#pragma once
#include "PrimeSieve.h"
#include <vector>
class SieveFactors : public PrimeSieve
{
public:
	SieveFactors(long long upTo);
	int Calculate();
protected:
	std::vector<bool> bitArray;
};

