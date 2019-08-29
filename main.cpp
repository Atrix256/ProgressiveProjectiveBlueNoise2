#define _CRT_SECURE_NO_WARNINGS

static const size_t c_sampleCount = 1024; // Must be a power of 2, for DFT purposes.
static const size_t c_imageSize = 256;
static const size_t c_radialAverageBucketCount = 64;
static const size_t c_numTestsForAveraging = 100;
static const size_t c_numProjections = 8;  // for 1d projection DFTs.  pi radians times 0/N, 1/N ... (N-1)/N

// Progressive Projective blue noise settings
static const size_t c_progProjAccelSize = 10;

#define DO_AVERAGE_TEST() false
#define DO_SLOW_TESTS() false
#define RANDOMIZE_SEEDS() false


#include <array>
#include <vector>
#include <thread>
#include <atomic>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "BN_Mitchels.h"
#include "BN_progprojRank.h"
#include "BN_progprojMin.h"
#include "rng.h"
#include "dft.h"
#include "scoped_timer.h"

#define NUM_TESTS() (DO_AVERAGE_TEST() ? c_numTestsForAveraging : 1)

static const float c_pi = 3.14159265359f;

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

void SaveCSV(const char* fileName, const std::vector<float>& data)
{
    FILE* file = fopen(fileName, "w+b");
    for (float f : data)
        fprintf(file, "\"%f\"\n", f);
    fclose(file);
}

void SaveCSV(const char* fileName, const std::vector<Vec2>& data)
{
    FILE* file = fopen(fileName, "w+b");
    for (const Vec2& v: data)
        fprintf(file, "%f, %f\n", v[0], v[1]);
    fclose(file);
}

template<size_t NumColumns>
void SaveCSV(const char* fileName, const std::array<std::vector<float>, NumColumns>& data)
{
    FILE* file = fopen(fileName, "w+b");

    for (size_t row = 0; row < data[0].size(); ++row)
    {
        for (size_t column = 0; column < NumColumns; ++column)
            fprintf(file, "\"%f\",", data[column][row]);
        fprintf(file, "\n");
    }

    fclose(file);
}

template <typename T>
void IncrementalAverage(const std::vector<T>& src, std::vector<T>& dest, size_t sampleIndex)
{
    if (sampleIndex == 0)
    {
        dest = src;
        return;
    }

    for (size_t index = 0; index < src.size(); ++index)
        dest[index] = Lerp(dest[index], src[index], 1.0f / float(sampleIndex + 1));
}

template <typename LAMBDA>
void DoTest(const char* label, const char* baseFileName, const LAMBDA& lambda)
{
    ScopedTimer timer(label);

    size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);
    std::vector<std::vector<float>> radialAverageds(NUM_TESTS());
    std::vector<std::vector<uint8_t>> imageDFTU8s(NUM_TESTS());
    std::vector<std::array<std::vector<float>, c_numProjections>> projectionDFTs(NUM_TESTS());

    std::atomic<size_t> nextIndex(0);
    for (size_t threadIndex = 0; threadIndex < threads.size(); ++threadIndex)
    {
        threads[threadIndex] = std::thread(
            [threadIndex, baseFileName, &radialAverageds, &imageDFTU8s, &projectionDFTs, &nextIndex, &lambda]()
            {
                char fileName[1024];

                size_t testIndex = nextIndex.fetch_add(1);
                while (testIndex < NUM_TESTS())
                {
                    // make the points
                    std::mt19937 rng = GetRNG(uint32_t(testIndex));
                    std::vector<Vec2> points;
                    lambda(rng, points);

                    // make an image of the samples
                    std::vector<float> image = MakeSampleImage(points, c_imageSize);

                    // DFT the image and get the radial average as well
                    std::vector<float> imageDFT;
                    std::vector<float>& radialAveraged = radialAverageds[testIndex];
                    DFTPeriodogram(image, imageDFT, c_imageSize, c_sampleCount, radialAveraged, c_radialAverageBucketCount);
                    std::vector<uint8_t> imageU8 = ImageFloatToU8(image, c_imageSize);

                    // convert the DFT to a U8 image
                    std::vector<uint8_t>& imageDFTU8 = imageDFTU8s[testIndex];
                    imageDFTU8 = ImageFloatToU8(imageDFT, c_imageSize);

                    // do projection DFTs
                    std::array<std::vector<float>, c_numProjections>& DFTs = projectionDFTs[testIndex];
                    for (size_t dftIndex = 0; dftIndex < c_numProjections; ++dftIndex)
                    {
                        std::vector<float> projectedValues(points.size());
                        float angle = c_pi * float(dftIndex) / float(c_numProjections);
                        float px = cos(angle);
                        float py = sin(angle);
                        for (size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex)
                        {
                            projectedValues[pointIndex] =
                                points[pointIndex][0] * px +
                                points[pointIndex][1] * py;
                        }

                        DFT1D(projectedValues, DFTs[dftIndex]);
                    }

                    // if this is the first test, write out the "one" images
                    if (testIndex == 0)
                    {
                        sprintf(fileName, "%s_one.png", baseFileName);
                        stbi_write_png(fileName, int(c_imageSize), int(c_imageSize), 1, imageU8.data(), 0);
                        sprintf(fileName, "%s_DFT_one.png", baseFileName);
                        stbi_write_png(fileName, int(c_imageSize), int(c_imageSize), 1, imageDFTU8.data(), 0);
                        sprintf(fileName, "%s_one.csv", baseFileName);
                        SaveCSV(fileName, radialAveraged);
                        sprintf(fileName, "%s_projections_one.csv", baseFileName);
                        SaveCSV(fileName, DFTs);
                        sprintf(fileName, "%s.txt", baseFileName);
                        SaveCSV(fileName, points);
                    }

                    // get next test index to do
                    testIndex = nextIndex.fetch_add(1);
                }
            }
        );
    }
    for (std::thread& t : threads)
        t.join();

    // combine the work of all the threads
    std::vector<float> radialAveraged_avg;
    std::vector<uint8_t> imageDFTU8_avg;
    std::array<std::vector<float>, c_numProjections> DFTs_avg;
    for (size_t index = 0; index < radialAverageds.size(); ++index)
    {
        IncrementalAverage(radialAverageds[index], radialAveraged_avg, index);
        IncrementalAverage(imageDFTU8s[index], imageDFTU8_avg, index);
        for (size_t projIndex = 0; projIndex < c_numProjections; ++projIndex)
            IncrementalAverage(projectionDFTs[index][projIndex], DFTs_avg[projIndex], index);
    }

    // report the averages
    #if DO_AVERAGE_TEST()
        char fileName[1024];
        sprintf(fileName, "%s_DFT_avg.png", baseFileName);
        stbi_write_png(fileName, int(c_imageSize), int(c_imageSize), 1, imageDFTU8_avg.data(), 0);
        sprintf(fileName, "%s_avg.csv", baseFileName);
        SaveCSV(fileName, radialAveraged_avg);
        sprintf(fileName, "%s_projections_avg.csv", baseFileName);
        SaveCSV(fileName, DFTs_avg);
    #endif
}

void DoExpectedDistanceTest()
{
    std::mt19937 rng = GetRNG(0);

    float distance1DAvg = 0.0f;
    float distance2DAvg = 0.0f;

    for (size_t i = 0; i < 10000000; ++i)
    {
        static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        float ax = dist(rng);
        float ay = dist(rng);
        float bx = dist(rng);
        float by = dist(rng);

        float dx = abs(bx - ax);
        float dy = abs(by - ay);

        float distance1D = dx;
        float distance2D = sqrt(dx*dx + dy * dy);

        distance1DAvg = Lerp(distance1DAvg, distance1D, 1.0f / float(i + 1));
        distance2DAvg = Lerp(distance2DAvg, distance2D, 1.0f / float(i + 1));
    }
    printf("1d = %f\n2d = %f\n", distance1DAvg, distance2DAvg);
}

int main(int argc, char** argv)
{
    /*
    DoExpectedDistanceTest();
    system("pause");
    return 0;
    */

    DoTest(
        "Progressive Projective Blue Noise Min",
        "out/BN_ProgProjMin",
        [](std::mt19937& rng, std::vector<Vec2>& points)
        {
            GoodCandidateSubspaceAlgorithmAccell_Min<2, c_progProjAccelSize, false>(rng, points, c_sampleCount, 1, false);
        }
    );

    return 0;

    DoTest(
        "Progressive Projective Blue Noise Rank",
        "out/BN_ProgProjRank",
        [](std::mt19937& rng, std::vector<Vec2>& points)
        {
            GoodCandidateSubspaceAlgorithmAccell_Rank<2, c_progProjAccelSize, false>(rng, points, c_sampleCount, 1, false);
        }
    );

#if DO_SLOW_TESTS()
    DoTest(
        ""Progressive Projective Blue Noise Rank Penalty",
        "out/BN_ProgProjRank_Penalty",
        [](std::mt19937& rng, std::vector<Vec2>& points)
        {
        GoodCandidateSubspaceAlgorithmAccell_Rank<2, c_progProjAccelSize, true>(rng, points, c_sampleCount, 1, false);
        }
    );

    DoTest(
        ""Progressive Projective Blue Noise Rank 5",
        "out/BN_ProgProjRank_5",
        [](std::mt19937& rng, std::vector<Vec2>& points)
        {
        GoodCandidateSubspaceAlgorithmAccell_Rank<2, c_progProjAccelSize, false>(rng, points, c_sampleCount, 5, false);
        }
    );

    DoTest(
        ""Progressive Projective Blue Noise Rank 5 Penalty",
        "out/BN_ProgProjRank_5Penalty",
        [](std::mt19937& rng, std::vector<Vec2>& points)
        {
        GoodCandidateSubspaceAlgorithmAccell_Rank<2, c_progProjAccelSize, true>(rng, points, c_sampleCount, 5, false);
        }
    );

    DoTest(
        ""Progressive Projective Blue Noise Rank 25",
        "out/BN_ProgProjRank_25",
        [](std::mt19937& rng, std::vector<Vec2>& points)
        {
        GoodCandidateSubspaceAlgorithmAccell_Rank<2, c_progProjAccelSize, false>(rng, points, c_sampleCount, 25, false);
        }
    );
#endif

    DoTest(
        "Mitchel's Best Candidate Blue Noise",
        "out/BN_Mitchels",
        [](std::mt19937& rng, std::vector<Vec2>& points)
        {
            MitchelsBestCandidateAlgorithm<2>(rng, points, c_sampleCount, 1);
        }
    );

    DoTest(
        "White Noise",
        "out/White",
        [](std::mt19937& rng, std::vector<Vec2>& points)
        {
            points.resize(c_sampleCount);
            static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (Vec2& v : points)
            {
                v[0] = dist(rng);
                v[1] = dist(rng);
            }
        }
    );

    system("pause");
    return 0;
}

/*

TODO:

* could try normalizing distances 0 to 1, subtracting that from 1 and summing that. It's like rank but not so digital.
* could try making the rank better. try multiplying ranks by a constant like 2?
* could try summing a harmonic mean of distances

* try weighted subspace scores, weighted by subspace sphere packing like in paper
 * might try seeing how raising distance to a power changes things (squared right now!)
 * could like... take harmonic mean of weighted scores or something too.

* try min of subspace scores.  Not all subspace scores should be equal, but it's worth a test
 * it's not any good.  you can see the lines but no blue noise (not missing low frequencies). maybe try with weighted scores?

? maybe the problem is that ranking isn't good enough cause you can do great in many subspaces but bomb it on one, and the results will take that, over things that did slightly ok?
 * maybe compare vs the extra penalty.  actually that wouldn't have changed things.
 * i think you need to sum the distance, but weigh the subspaces differently. maybe by sphere packing amount.

* projective blue noise thing doesn't seem to be working. the projections don't look good at all.
 * probably need to try averaging 100 tests? dunno... a single test should show something shouldn't it?

* in 2d DFT, you aren't using the normalized term! no wonder it wasn't working. check it out. try sRGB correction etc again

* can we provide labels for the projective DFTs somehow?

* remake everything since you bumped it to 1024.

* try 10x, and 5x penalty.
 * I did... cant' seem to get better blue noise characteristics.
 * review your code, make sure it's working correctly

* need to do 1d dft of x and y projections, as well as some number of other projections. Porbably golden ratio angles on 180 degrees?

* use accel structure for regular blue noise too

* compare vs the "extra penalty"
 * does that actually even do anything? might verify and see

* compare your projective blue noise vs the actual projective blue noise
 * also make that projective blue noise progressive, using the thing from void and cluster algorithm

* show the extra penalty not working out

* subspace projections you already have, need to try random projections

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
* Taking minimum of sum of rank isn't good enough. There are times when a point does great on 2 tests, but does much worse on a third (super super close to an existing point) but is accepted. 
 * we would prefer something that does "ok" on all tests, over one that is great on mosts tests and bad on the remaining
? were you going to co-author with brandon so it could be the wolfe-mann algorithm? :P
* we are doing periodograms like in the subr16 paper linked below. currently not using squared mag though!


The older progressive projective blue noise repo has some more links that aren't relevant to proj prog but sampling in general & sample zoo

* This specifically has info about calculating the power spectrum (not fourier magnitude!), normalizing it, radial averaging it, and has c++ source code to do so.
 * https://cs.dartmouth.edu/wjarosz/publications/subr16fourier.html

* projective blue noise article: http://resources.mpi-inf.mpg.de/ProjectiveBlueNoise/

*/