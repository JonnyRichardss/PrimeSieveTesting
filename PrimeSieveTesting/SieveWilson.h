#pragma once
#include "PrimeSieve.h"
class SieveWilson : public PrimeSieve
{
public:
	SieveWilson(long long upTo);
	int Calculate();
};

