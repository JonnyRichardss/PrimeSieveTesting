#include "PrimeSieve.h"
#include <cstdio>
PrimeSieve::PrimeSieve(long long upTo) : upTo(upTo),last_result(0)
{
}

PrimeSieve::~PrimeSieve()
{
}

bool PrimeSieve::Validate()
{
    auto correctValIterator = resultsDictionary.find(upTo);
    if (correctValIterator == resultsDictionary.end()) {
        printf("No historical answer for max value %lli! -- Assuming answer is correct.\n", upTo);
        return true;
    }
    if (correctValIterator->second == last_result) {
        printf("Correctly found %i primes up to %lli!\n", last_result, upTo);
        return true;
    }
    else {
        printf("expected to find %i primes up to %lli, got %i!\n", correctValIterator->second, upTo,last_result);
        return false;
    }
}
