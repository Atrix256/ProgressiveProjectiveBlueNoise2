#pragma once

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

    float DistanceToClosestPoint(const T& point)
    {
        return sqrt(SquaredDistanceToClosestPoint(point));
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