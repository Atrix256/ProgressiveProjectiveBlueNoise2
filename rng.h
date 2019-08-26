#pragma once

#include <random>

std::mt19937 GetRNG(uint32_t contextIndex)
{
    uint32_t seed[8] =
    {
        377243142 ^ contextIndex,
        3379348808 ^ contextIndex,
        3529983610 ^ contextIndex,
        2483992069 ^ contextIndex,
        2532182207 ^ contextIndex,
        600251115 ^ contextIndex,
        3649906436 ^ contextIndex,
        3065633521 ^ contextIndex
    };

    #if RANDOMIZE_SEEDS()
        std::random_device rd("dev/random");
        for (int i = 0; i < 8; ++i)
            seed[i] = rd();
    #endif

    std::seed_seq fullSeed{ seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], seed[6], seed[7] };
    return std::mt19937(fullSeed);
}