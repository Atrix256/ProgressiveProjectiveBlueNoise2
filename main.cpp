#define _CRT_SECURE_NO_WARNINGS

#include <array>
#include <vector>
#include <random>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "BN_Mitchels.h"
#include "BN_progproj.h"
#include "dft.h"

typedef std::array<float, 2> Vec2;

int main(int argc, char** argv)
{
    {
        std::vector<Vec2> points;
        GoodCandidateSubspaceAlgorithmAccell<2, 10, false>(points, 100, 100, true); // TODO: the multiplier is 100. seems high. is that tuned correctly?
    }

    {
        std::vector<Vec2> points;
        MitchelsBestCandidateAlgorithm<2>(points, 100, 1);
    }

    return 0;
}

/*

TODO:

* compare your projective blue noise vs the actual projective blue noise
 * also make that projective blue noise progressive, using the thing from void and cluster algorithm

* maybe compare against both white noise and regular blue noise

* show the extra penalty not working out

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