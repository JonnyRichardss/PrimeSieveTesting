#include "SieveWilson.h"
#include "gmpxx.h"
SieveWilson::SieveWilson(long long upTo) : PrimeSieve(upTo, "Wilson's formula based sieve")
{
}

int SieveWilson::Calculate()
{
    
    //reset
    lastResult = 0;
    mpz_class currentFactorial = 1; //formula relies on (n-1)! +1 so we keep track of (n-1) factorial
    //i think theres not really any point skipping evens, they need to be multiplied in anyway and filtering them out already costs a mod, which we do right after anyway
    //its possible that % 2 is a special case at a machine level and it *is* worth doing but im going to leave it for now
    for (int i = 2; i < upTo; i ++) {
        currentFactorial *= i - 1;
        if ((currentFactorial + 1) % i == 0)
        {
            lastResult++;
        }
    }
    return lastResult;
}
