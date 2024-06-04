#pragma once
#include <map>
class PrimeSieve
{
public:
	PrimeSieve(long long upTo, const char* name = "Unnamed Sieve class");
	virtual ~PrimeSieve();
	//returns the number of primes below upTo (set at construction of the sieve) and saves it for validation
	virtual int Calculate() = 0;
	//returns true if the last calculated value is equal to the pre-defined correct value
	bool Validate();
    int GetExpected();
    const char* name;
	int lastResult;
    const long long upTo;
protected:
    //https://github.com/PlummersSoftwareLLC/Primes/blob/original/PrimeCPP/PrimeCPP.cpp
    const std::map<const long long, const int> resultsDictionary =
    {
          {          10LL, 4         },               // Historical data for validating our results - the number of primes
          {          20LL, 8         },
          {         100LL, 25        },               // to be found under some limit, such as 168 primes under 1000
          {        1000LL, 168       },
          {       10000LL, 1229      },
          {      100000LL, 9592      },
          {     1000000LL, 78498     },
          {    10000000LL, 664579    },
          {   100000000LL, 5761455   },
          {  1000000000LL, 50847534  },
          { 10000000000LL, 455052511 },
    };
};

