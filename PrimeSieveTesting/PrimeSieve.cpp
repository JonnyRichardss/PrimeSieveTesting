#include "PrimeSieve.h"
#include <cstdio>
#include "EnablePrints.h"
PrimeSieve::PrimeSieve(long long upTo,const char* name) : upTo(upTo), lastResult(0), name(name)
{
}

PrimeSieve::~PrimeSieve()
{
}

bool PrimeSieve::Validate()
{
    auto correctValIterator = resultsDictionary.find(upTo);
    if (correctValIterator == resultsDictionary.end()) {
        if (!DONT_PRINT) printf("No historical answer for max value %lli! -- Assuming answer is correct.\n", upTo);
        return true;
    }
    if (correctValIterator->second == lastResult) {
        if (!DONT_PRINT) printf("Correctly found %i primes up to %lli!\n", lastResult, upTo);
        return true;
    }
    else {
        if (!DONT_PRINT) printf("expected to find %i primes up to %lli, got %i!\n", correctValIterator->second, upTo,lastResult);
        return false;
    }
}

int PrimeSieve::GetExpected()
{
    auto correctValIterator = resultsDictionary.find(upTo);
    if (correctValIterator == resultsDictionary.end()) return -1;
    else return correctValIterator->second;
}
