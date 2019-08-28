#pragma once

#include <random>

#include "BN_accel.h"

template <size_t DIMENSION, size_t PARTITIONS, bool EXTRAPENALTY>
void GoodCandidateSubspaceAlgorithmAccell_Rank(std::mt19937& rng, std::vector< std::array<float, DIMENSION>>& results, size_t desiredItemCount, int candidateMultiplier, bool reportProgress)
{
    typedef std::array<float, DIMENSION> T;

    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // map candidate index to score
    struct CandidateScore
    {
        size_t index;
        float score;
    };
    typedef std::vector<CandidateScore> CandidateScores;
    static const size_t c_numScores = (1 << DIMENSION) - 1;  // 2^(dimension)-1

    // setup the acceleration structures
    std::array<GoodCandidateSubspace<DIMENSION, PARTITIONS>, c_numScores> subspaces;
    for (size_t scoreIndex = 0; scoreIndex < c_numScores; ++scoreIndex)
        subspaces[scoreIndex].Init(scoreIndex);

    // make space for the results
    results.resize(desiredItemCount);

    int lastPercent = -1;

    // for each item we need to fill in
    for (int itemIndex = 0; itemIndex < desiredItemCount; ++itemIndex)
    {
        // calculate how many candidates we want to generate for this item
        int candidateCount = itemIndex * candidateMultiplier + 1;

        // generate the candidates
        std::vector<T> candidates;
        candidates.resize(candidateCount);
        for (T& candidate : candidates)
        {
            for (size_t i = 0; i < DIMENSION; ++i)
                candidate[i] = dist(rng);
        }

        // initialize the overall scores
        CandidateScores overallScores;
        overallScores.resize(candidateCount);
        for (int candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
        {
            overallScores[candidateIndex].index = candidateIndex;
            overallScores[candidateIndex].score = 0.0f;
        }

        // allocate space for the individual scores
        CandidateScores scores;
        scores.resize(candidateCount);

        // score the candidates by each measure of scoring
        for (size_t scoreIndex = 0; scoreIndex < c_numScores; ++scoreIndex)
        {
            // get the subspace we are working with
            auto& subspace = subspaces[scoreIndex];

            // for each candidate in this score index...
            for (size_t candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
            {
                // calculate the score of the candidate.
                // the score is the minimum distance to any other points
                scores[candidateIndex].index = candidateIndex;
                scores[candidateIndex].score = subspace.SquaredDistanceToClosestPoint(candidates[candidateIndex]);
            }

            // sort the scores from high to low
            std::sort(
                scores.begin(),
                scores.end(),
                [](const CandidateScore& A, const CandidateScore& B)
                {
                    return A.score > B.score;
                }
            );

            // add the rank of this score a score for each candidate
            for (size_t candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
                overallScores[scores[candidateIndex].index].score += EXTRAPENALTY ? float(candidateIndex) * float(candidateIndex) : float(candidateIndex);
        }

        // sort the overall scores from low to high
        std::sort(
            overallScores.begin(),
            overallScores.end(),
            [](const CandidateScore& A, const CandidateScore& B)
            {
                return A.score < B.score;
            }
        );

        // keep the point that had the lowest summed rank
        results[itemIndex] = candidates[overallScores[0].index];

        // insert this point into the acceleration structures
        for (size_t scoreIndex = 0; scoreIndex < c_numScores; ++scoreIndex)
            subspaces[scoreIndex].Insert(results[itemIndex]);

        // report our percentage done if we should
        if (reportProgress)
        {
            int percent = int(100.0f * float(itemIndex) / float(desiredItemCount));
            if (lastPercent != percent)
            {
                lastPercent = percent;
                printf("\rMaking Points: %i%%", lastPercent);
            }
        }
    }

    if (reportProgress)
        printf("\rMaking Points: 100%%\n");
}