#include <cstdio>
#include <chrono>
#include <vector>
#include "SieveFactors.h"
#include "SieveWilson.h"
#include "SieveFactorsCUDA.cuh"
using namespace std::chrono_literals;

void Test_Sieve(PrimeSieve& sieve, std::chrono::seconds testTime) {
    //deviating from what (i think) DavePL does here - instead i'm going to run-validate-reset
    //he only validates once at the end but i want to run and validate so my sieves will reset themselves when they start running
    auto startT = std::chrono::steady_clock::now();
    int passes = 0;
    std::vector<bool> passValidations;
    while (std::chrono::steady_clock::now() - startT < testTime) {
        sieve.Calculate();
        passes++;
        passValidations.push_back(sieve.Validate());
    }
    int correctCount = 0; for (bool b : passValidations) if (b) correctCount++;
    double duration = (std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::steady_clock::now() - startT).count()) / 1000000.0;
    printf("Sieve '%s' results: \n%s \nNumber of passes: %i \nNumber of correct passes: %i \nExpected count: %i \nLast pass count: %i \nTotal time: %lf \nAverage time: %lf \nSieve size: %lli \n%s\n",
        sieve.name,
        "------------------",
        passes,
        correctCount,
        sieve.GetExpected(),
        sieve.lastResult,
        duration,
        duration / passes,
        sieve.upTo,
        "------------------");
    if (sieve.GetExpected() == -1) {
        printf("Expected value of -1 means that the value of 'upto' is different to those stored in the validation table.\n");
    }
}
int main()
{
    long long sieveSize = 1000000LL;
    auto testTime = 2s;
    



    SieveFactors factorSieve(sieveSize);
    SieveWilson wilsonSieve(sieveSize);
    SieveFactorsCUDA cudaSieve(sieveSize);
    Test_Sieve(cudaSieve, testTime);
    Test_Sieve(factorSieve,testTime);
    //Test_Sieve(wilsonSieve, testTime);   
}