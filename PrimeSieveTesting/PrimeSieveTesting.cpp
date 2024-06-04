#include <cstdio>
#include <chrono>
#include <vector>
#include "SieveFactors.h"
using namespace std::chrono_literals;

void Test_Sieve(PrimeSieve& sieve, std::chrono::seconds testTime) {
    //deviating from what (i think) DavePL does here - instead i'm going to run-validate-reset
    //as far as i can tell his repeated passes will be much faster since its working on a completed bit array
    //but i want to run and validate since i have other methods coming that can't work the same way
    auto startT = std::chrono::steady_clock::now();
    int passes = 0;
    std::vector<bool> passValidations;
    while (std::chrono::steady_clock::now() - startT < testTime) {
        sieve.Calculate();
        passes++;
        passValidations.push_back(sieve.Validate());
    }
    int correctCount = 0; for (bool b : passValidations) if (b) correctCount++;
    double duration = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::steady_clock::now() - startT).count() / 1000000;
    printf("Sieve %s results: \n%s \nNumber of passes: %i \nNumber of correct passes: %i \nExpected count: %i \nLast pass count: %i \nTotal time: %lf \nAverage time: %lf \nSieve size: %lli \n",
        sieve.name,
        "------------------",
        passes,
        correctCount,
        sieve.GetExpected(),
        sieve.lastResult,
        duration,
        duration / passes,
        sieve.upTo);
    if (sieve.GetExpected() == -1) {
        printf("Expected value of -1 means that the value of 'upto' is different to those stored in the validation table.\n");
    }
}
int main()
{
    SieveFactors factorSieve(1000000L);
    Test_Sieve(factorSieve,2s);
    //printf("Hello, World!\n");
    
}