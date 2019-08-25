#define _CRT_SECURE_NO_WARNINGS

static const size_t c_sampleCount = 1000;
static const size_t c_imageSize = 256;
static const size_t c_radialAverageBucketCount = 64;
static const size_t c_numTestsForAveraging = 10; // TODO: 10 isn't enough!

// Mitchel's best candidate blue noise settings
static const size_t c_mitchelCandidateMultiplier = 1;

// Progressive Projective blue noise settings
static const size_t c_progProjAccelSize = 10;
static const size_t c_progProjCandidateMultiplier = 10; // TODO: need to search for a good value here
// TODO: are the numbers above properly tuned? 10 seems to be as good as 100? but test radial one maybe

// TODO: when true, do the tests a bunch of times (how many? make a constant but set to what?) and average results. show averaged DFT and radial one. make it multi threaded to go as fast as possible. show progress
#define DO_AVERAGE_TEST() false
#define RANDOMIZE_SEEDS() false



#include <array>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "BN_Mitchels.h"
#include "BN_progproj.h"
#include "rng.h"
#include "dft.h"
#include "scoped_timer.h"

typedef std::array<float, 2> Vec2;

std::vector<float> MakeSampleImage(const std::vector<Vec2>& points, size_t imageResolution)
{
    std::vector<float> ret;
    ret.resize(imageResolution*imageResolution, 0.0f);

    for (const Vec2& point : points)
    {
        size_t x = size_t(point[0] * float(imageResolution));
        size_t y = size_t(point[1] * float(imageResolution));
        ret[y*imageResolution + x] = 1.0f;
    }

    return ret;
}

std::vector<uint8_t> ImageFloatToU8(const std::vector<float>& image, size_t imageResolution)
{
    std::vector<uint8_t> ret(imageResolution*imageResolution);
    for (size_t index = 0, count = image.size(); index < count; ++index)
    {
        float valueFloat = powf(image[index], 1.0f / 2.2f);
        ret[index] = uint8_t(valueFloat * 255.0f + 0.5f);
    }
    return ret;
}

void SaveCSV(const char* fileName, const std::vector<float>& spectrum)
{
    FILE* file = fopen(fileName, "w+b");
    for (float f : spectrum)
        fprintf(file, "\"%f\"\n", f);
    fclose(file);
}

int main(int argc, char** argv)
{
    {
        ScopedTimer timer("Progressive Projective Blue Noise");
        std::mt19937 rng = GetRNG();
        std::vector<Vec2> points;
        GoodCandidateSubspaceAlgorithmAccell<2, c_progProjAccelSize, false>(rng, points, c_sampleCount, c_progProjCandidateMultiplier, true);
        std::vector<float> image = MakeSampleImage(points, c_imageSize);
        std::vector<float> imageDFT;
        std::vector<float> radialAveraged;
        DFTPeriodogram(image, imageDFT, c_imageSize, c_sampleCount, radialAveraged, c_radialAverageBucketCount);
        std::vector<uint8_t> imageU8 = ImageFloatToU8(image, c_imageSize);
        std::vector<uint8_t> imageDFTU8 = ImageFloatToU8(imageDFT, c_imageSize);

        stbi_write_png("out/BN_ProgProj.png", int(c_imageSize), int(c_imageSize), 1, imageU8.data(), 0);
        stbi_write_png("out/BN_ProgProj_DFT.png", int(c_imageSize), int(c_imageSize), 1, imageDFTU8.data(), 0);
        SaveCSV("out/BN_ProgProj.csv", radialAveraged);
    }

    {
        ScopedTimer timer("Mitchel's Best Candidate Blue Noise");
        std::mt19937 rng = GetRNG();
        std::vector<Vec2> points;
        MitchelsBestCandidateAlgorithm<2>(rng, points, c_sampleCount, c_mitchelCandidateMultiplier);
        std::vector<float> image = MakeSampleImage(points, c_imageSize);
        std::vector<float> imageDFT;
        std::vector<float> radialAveraged;
        DFTPeriodogram(image, imageDFT, c_imageSize, c_sampleCount, radialAveraged, c_radialAverageBucketCount);
        std::vector<uint8_t> imageU8 = ImageFloatToU8(image, c_imageSize);
        std::vector<uint8_t> imageDFTU8 = ImageFloatToU8(imageDFT, c_imageSize);

        stbi_write_png("out/BN_Mitchels.png", int(c_imageSize), int(c_imageSize), 1, imageU8.data(), 0);
        stbi_write_png("out/BN_Mitchels_DFT.png", int(c_imageSize), int(c_imageSize), 1, imageDFTU8.data(), 0);
        SaveCSV("out/BN_Mitchels.csv", radialAveraged);
    }

    system("pause");
    return 0;
}

/*

TODO:

* make these tests run multithreaded, maybe at least when doing the same test N times, have that threaded.

* use accel structure for regular blue noise too

* make random numbers deterministic by default (make a define to use random instead of fixed seed?)

* compare vs the "extra penalty"
 * does that actually even do anything? might verify and see

* compare your projective blue noise vs the actual projective blue noise
 * also make that projective blue noise progressive, using the thing from void and cluster algorithm

* compare against white noise

* show the extra penalty not working out

* subspace projections you already have
* random projections

* Compare vs Eric heitz screen space blue noise, which should essentially be the same as random projections?
 * The Eric heitz screen space blue noise is related because it's optimized against random heaviside functions, so is basically projective blue noise on all axes
 
* also i think this relates to Matt Phar's talk about sample warping and stratified blue noise keeping better low discrepancy under sample warping.

? do we care about higher dimensions

* also projective blue noise masks w/ modified void and cluster algorithm. maybe this as a second paper / investigation

TESTS:
- show samples (2d)
- DFT (2d)
- radial DFT?

Get this into sample zoo!





Notes:
? were you going to co-author with brandon so it could be the wolfe-mann algorithm? :P
* we are doing periodograms like in the subr16 paper linked below. currently not using squared mag though!


The older progressive projective blue noise repo has some more links that aren't relevant to proj prog but sampling in general & sample zoo

* This specifically has info about calculating the power spectrum (not fourier magnitude!), normalizing it, radial averaging it, and has c++ source code to do so.
 * https://cs.dartmouth.edu/wjarosz/publications/subr16fourier.html

* projective blue noise article: http://resources.mpi-inf.mpg.de/ProjectiveBlueNoise/

*/