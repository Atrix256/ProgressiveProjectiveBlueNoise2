#pragma once

#include "rng.h"

template <size_t DIMENSION>
void MitchelsBestCandidateAlgorithm(std::vector< std::array<float, DIMENSION>>& results, size_t desiredItemCount, int candidateMultiplier)
{
    typedef std::array<float, DIMENSION> T;

    static std::mt19937 rng = GetRNG();
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    results.resize(desiredItemCount);

    // for each item we need to fill in
    for (int itemIndex = 0; itemIndex < desiredItemCount; ++itemIndex)
    {
        // calculate how many candidates we want to generate for this item
        int candidateCount = itemIndex * candidateMultiplier + 1;

        T bestCandidate;
        float bestCandidateMinimumDifferenceScore;

        // for each candidate
        for (int candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
        {
            // make a randomized candidate
            T candidate;
            for (size_t i = 0; i < DIMENSION; ++i)
                candidate[i] = dist(rng);

            float minimumDifferenceScore = FLT_MAX;

            // the score of this candidate is the minimum difference from all existing items
            for (int checkItemIndex = 0; checkItemIndex < itemIndex; ++checkItemIndex)
            {
                float distSq = 0.0f;
                for (int i = 0; i < DIMENSION; ++i)
                {
                    float diff = fabsf(results[checkItemIndex][i] - candidate[i]);
                    if (diff > 0.5f)
                        diff = 1.0f - diff;
                    distSq += diff * diff;
                }
                minimumDifferenceScore = std::min(minimumDifferenceScore, distSq);
            }

            // the candidate with the largest minimum distance is the one we want to keep
            if (candidateIndex == 0 || minimumDifferenceScore > bestCandidateMinimumDifferenceScore)
            {
                bestCandidate = candidate;
                bestCandidateMinimumDifferenceScore = minimumDifferenceScore;
            }
        }

        // keep the winning candidate
        results[itemIndex] = bestCandidate;
    }
}