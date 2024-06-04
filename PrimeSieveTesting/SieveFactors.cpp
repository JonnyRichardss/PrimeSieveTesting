#include "SieveFactors.h"

SieveFactors::SieveFactors(long long upTo) : PrimeSieve(upTo), bitArray(upTo,true)
{
}

int SieveFactors::Calculate()
{
    //reset
    std::fill(bitArray.begin(), bitArray.end(), true);
    last_result = 0;

    int maxFactor = sqrt(upTo);
    for (int factor = 3; factor < maxFactor; factor += 2) {//only checking odd numbers
        if (!bitArray[factor]) continue;//if its already a multiple of something else - skip
        for (int product = factor * factor; product < upTo; product += factor * 2) {
            bitArray[product] = false;//every odd multiple of factor is now set to false
        }
    }
    for (int i = 0; i < upTo; i++) {
        if (bitArray[i]) last_result++;
    }
    return last_result;
}
