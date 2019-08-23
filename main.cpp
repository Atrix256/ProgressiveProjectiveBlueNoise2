#include <array>
#include <vector>
#include <random>

template <size_t DIMENSION, size_t PARTITIONS>
struct GoodCandidateSubspace
{
    typedef std::array<float, DIMENSION> T;

    void Init(size_t scoreIndex)
    {
        size_t partitionCount = 1;
        int multiplier = 1;
        for (size_t dimensionIndex = 0; dimensionIndex < DIMENSION; ++dimensionIndex)
        {
            axisMask[dimensionIndex] = (scoreIndex & (size_t(1) << dimensionIndex)) ? false : true;

            if (!axisMask[dimensionIndex])
            {
                axisPartitionOffset[dimensionIndex] = 0;
                continue;
            }

            axisPartitionOffset[dimensionIndex] = multiplier;
            partitionCount *= PARTITIONS;
            multiplier *= PARTITIONS;
        }

        partitionedPoints.resize(partitionCount);
        partitionChecked.resize(partitionCount);
    }

    int PartitionForPoint(const T& point) const
    {
        int subspacePartition = 0;
        int multiplier = 1;
        for (size_t dimensionIndex = 0; dimensionIndex < DIMENSION; ++dimensionIndex)
        {
            if (!axisMask[dimensionIndex])
                continue;

            int axisPartition = int(point[dimensionIndex] * float(PARTITIONS));
            axisPartition = std::min(axisPartition, int(PARTITIONS - 1));
            subspacePartition += axisPartition * multiplier;
            multiplier *= PARTITIONS;
        }

        // TODO: use axisPartitionOffset[] instead of duplicating multiplier logic!

        return subspacePartition;
    }

    void PartitionCoordinatesForPoint(const T& point, std::array<int, DIMENSION>& partitionCoordinates) const
    {
        int partition = PartitionForPoint(point);

        for (size_t dimensionIndexPlusOne = DIMENSION; dimensionIndexPlusOne > 0; --dimensionIndexPlusOne)
        {
            size_t dimensionIndex = dimensionIndexPlusOne - 1;

            if (!axisMask[dimensionIndex])
            {
                partitionCoordinates[dimensionIndex] = 0;
                continue;
            }

            partitionCoordinates[dimensionIndex] = partition / axisPartitionOffset[dimensionIndex];
            partition = partition % axisPartitionOffset[dimensionIndex];
        }
    }

    float SquaredDistanceToClosestPointRecursive(const T& point, std::array<int, DIMENSION> partitionCoordinates, int radius, size_t dimensionIndex)
    {
        // if we have run out of dimensions, it's time to search a partition
        if (dimensionIndex == DIMENSION)
        {
            int subspacePartition = 0;
            for (int i = 0; i < DIMENSION; ++i)
            {
                if (!axisMask[i])
                    continue;
                subspacePartition += partitionCoordinates[i] * axisPartitionOffset[i];
            }

            // if we've already checked this partition, nothing to do.
            // otherwise, mark is as checked so it isn't checked again.
            if (partitionChecked[subspacePartition])
                return FLT_MAX;
            partitionChecked[subspacePartition] = true;

            float minDistSq = FLT_MAX;
            for (auto& p : partitionedPoints[subspacePartition])
            {
                float distSq = 0.0f;
                for (size_t i = 0; i < DIMENSION; ++i)
                {
                    if (!axisMask[i])
                        continue;

                    float diff = fabsf(p[i] - point[i]);
                    if (diff > 0.5f)
                        diff = 1.0f - diff;
                    distSq += diff * diff;
                }
                minDistSq = std::min(minDistSq, distSq);
            }

            return minDistSq;
        }

        // if radius 0, or this axis doesn't participate, do a pass through!
        if (radius == 0 || !axisMask[dimensionIndex])
            return SquaredDistanceToClosestPointRecursive(point, partitionCoordinates, radius, dimensionIndex + 1);

        // loop through this axis radius and return the smallest value we've found
        float ret = FLT_MAX;
        std::array<int, DIMENSION> searchPartitionCoordinates = partitionCoordinates;
        for (int axisOffset = -radius; axisOffset <= radius; ++axisOffset)
        {
            searchPartitionCoordinates[dimensionIndex] = (partitionCoordinates[dimensionIndex] + axisOffset + PARTITIONS) % PARTITIONS;
            ret = std::min(ret, SquaredDistanceToClosestPointRecursive(point, searchPartitionCoordinates, radius, dimensionIndex + 1));
        }

        return ret;
    }

    float SquaredDistanceToClosestPoint(const T& point)
    {
        // mark all partitions as having not been checked yet
        std::fill(partitionChecked.begin(), partitionChecked.end(), false);

        // get the partition coordinate this point is in
        std::array<int, DIMENSION> partitionCoordinates;
        PartitionCoordinatesForPoint(point, partitionCoordinates);

        // Loop through increasingly larger rectangular rings until we find a ring that has at least one point.
        // return the distance to the closest point in that ring, and the next ring out.
        // We need to do an extra ring to get the correct answer.
        int maxRadius = int(PARTITIONS / 2);
        bool foundInnerRing = false;
        float minDist = FLT_MAX;
        for (int radius = 0; radius <= maxRadius; ++radius)
        {
            float distance = SquaredDistanceToClosestPointRecursive(point, partitionCoordinates, radius, 0);
            minDist = std::min(minDist, distance);
            if (minDist < FLT_MAX)
            {
                if (foundInnerRing)
                    return minDist;
                else
                    foundInnerRing = true;
            }
        }
        return minDist;
    }

    void Insert(const T& point)
    {
        int subspacePartition = PartitionForPoint(point);
        partitionedPoints[subspacePartition].push_back(point);
    }

    std::vector<std::vector<T>> partitionedPoints;
    std::vector<bool> partitionChecked;
    std::array<bool, DIMENSION> axisMask;
    std::array<int, DIMENSION> axisPartitionOffset;
};


template <size_t DIMENSION, size_t PARTITIONS, bool EXTRAPENALTY>
void GoodCandidateAlgorithmAccell(std::vector< std::array<float, DIMENSION>>& results, size_t desiredItemCount, int candidateMultiplier, bool reportProgress)
{
    typedef std::array<float, DIMENSION> T;

    static std::random_device rd;
    static std::mt19937 rng(rd());
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

int main(int argc, char** argv)
{
    return 0;
}

/*

TODO:

* compare your projective blue noise vs the actual projective blue noise
 * also make that projective blue noise progressive, using the thing from void and cluster algorithm

* subspace projections
* random projections

* Compare vs Eric heitz screen space blue noise, which should essentially be the same as random projections?
 * The Eric heitz screen space blue noise is related because it's optimized against random heaviside functions, so is basically projective blue noise on all axes
 
* also i think this relates to Matt Phar's talk about sample warping and stratified blue noise keeping better low discrepancy under sample warping.


TESTS:
- show samples (2d)
- DFT (2d)


Get this into sample zoo!

*/