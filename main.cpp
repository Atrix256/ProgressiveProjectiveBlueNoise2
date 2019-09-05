#define _CRT_SECURE_NO_WARNINGS

static const size_t c_sampleCount = 1024; // Must be a power of 2, for DFT purposes.
static const size_t c_imageSize = 256;
static const size_t c_radialAverageBucketCount = 64;
static const size_t c_numTestsForAveraging = 100;
static const size_t c_numProjections = 8;  // for 1d projection DFTs.  pi radians times 0/N, 1/N ... (N-1)/N

// Progressive Projective blue noise settings
static const size_t c_progProjAccelSize = 10;

#define DO_AVERAGE_TEST() true
#define DO_SLOW_TESTS() false
#define RANDOMIZE_SEEDS() false
#define SHOW_ROTATED_PROJECTIONS() false

#include <array>
#include <vector>
#include <thread>
#include <atomic>

#include "BN_Mitchels.h"
#include "BN_progprojRank.h"
#include "BN_progprojMin.h"
#include "rng.h"
#include "dft.h"
#include "scoped_timer.h"
#include "image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define NUM_TESTS() (DO_AVERAGE_TEST() ? c_numTestsForAveraging : 1)

std::vector<float> MakeSampleImage(const std::vector<Vec2>& points, size_t imageResolution)
{
    std::vector<float> ret;
    ret.resize(imageResolution*imageResolution, 0.0f);

    for (const Vec2& point : points)
    {
        size_t x = std::min(size_t(point[0] * float(imageResolution)), imageResolution - 1);
        size_t y = std::min(size_t(point[1] * float(imageResolution)), imageResolution - 1);
        ret[y*imageResolution + x] = 1.0f;
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
    std::vector<std::vector<float>> DFTMagnitudes2d(NUM_TESTS());
    std::vector<std::array<std::vector<float>, c_numProjections>> DFTMagnitudes1d(NUM_TESTS());

    std::atomic<size_t> nextIndex(0);
    for (size_t threadIndex = 0; threadIndex < threads.size(); ++threadIndex)
    {
        threads[threadIndex] = std::thread(
            [threadIndex, baseFileName, &radialAverageds, &DFTMagnitudes2d, &DFTMagnitudes1d, &nextIndex, &lambda]()
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
                    Image imageU8;
                    FloatToImage(image, c_imageSize, c_imageSize, imageU8);

                    // DFT the image
                    DFTPeriodogram(image, DFTMagnitudes2d[testIndex], c_imageSize, c_sampleCount);
                    std::vector<float> imageDFT;
                    NormalizeDFT(DFTMagnitudes2d[testIndex], imageDFT);
                    Image imageDFTU8;
                    FloatToImage(imageDFT, c_imageSize, c_imageSize, imageDFTU8);

                    // get the radial averaged power spectrum
                    std::vector<float>& radialAveraged = radialAverageds[testIndex];
                    RadiallyAveragePowerSpectrum(imageDFT, c_imageSize, radialAveraged, c_radialAverageBucketCount);

                    // TODO: normalize the raidally averaged power spectrum? probably! also make an image??

                    // do projection DFTs
                    std::array<std::vector<float>, c_numProjections> projectedValues;
                    std::array<std::vector<float>, c_numProjections>& projectionDFTsRaw = DFTMagnitudes1d[testIndex];
                    std::array<std::vector<float>, c_numProjections> projectionDFTs;
                    for (size_t projectionIndex = 0; projectionIndex < c_numProjections; ++projectionIndex)
                    {
                        projectedValues[projectionIndex].resize(points.size());
                        float angle = c_pi * float(projectionIndex) / float(c_numProjections);
                        Vec2 projectionAxis = Vec2{ cos(angle), sin(angle) };
                        float length = abs(projectionAxis[0]) + abs(projectionAxis[1]);
                        for (size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex)
                        {
                            // points[pointIndex] is in [0,1].
                            // make point be in [-1, 1]
                            Vec2 point = points[pointIndex] * 2.0f - 1.0f;

                            // do the projection
                            projectedValues[projectionIndex][pointIndex] = Dot(point, projectionAxis);

                            // bring the projected value back into [-1, 1] by scaling
                            projectedValues[projectionIndex][pointIndex] /= length;

                            // put the projected value into [0,1]
                            projectedValues[projectionIndex][pointIndex] = projectedValues[projectionIndex][pointIndex] * 0.5f + 0.5f;
                        }

                        DFT1D(projectedValues[projectionIndex], projectionDFTsRaw[projectionIndex]);
                        NormalizeDFT(projectionDFTsRaw[projectionIndex], projectionDFTs[projectionIndex]);
                    }

                    // if this is the first test, write out the "one" images
                    if (testIndex == 0)
                    {
                        Image separatorH(c_imageSize, 1, 128);

                        // TODO: move this code into a function, rename things, clean up, when working
                        for (size_t projectionIndex = 0; projectionIndex < c_numProjections; ++projectionIndex)
                        {
                            #if SHOW_ROTATED_PROJECTIONS()
                                Image imageRotatedU8;
                            #endif

                            // make image domain images (left side)
                            {
                                Image projectedPointsImage(c_imageSize, 64, 255);
                                DrawCircle(projectedPointsImage, 22, 42, 20, 192);

                                float angle = c_pi * float(projectionIndex) / float(c_numProjections);
                                float px = cos(angle);
                                float py = sin(angle);
                                int x1 = int(22.0f - px * 20.0f);
                                int x2 = int(22.0f + px * 20.0f);
                                int y1 = int(42.0f + py * 20.0f);
                                int y2 = int(42.0f - py * 20.0f);
                                DrawLine(projectedPointsImage, x1, y1, x2, y2, 255);

                                float projectionLength = abs(px) + abs(py);

                                std::array<float, c_imageSize> histogram;
                                std::fill(histogram.begin(), histogram.end(), 0.0f);
                                float maxCount = 0.0f;

                                for (float f : projectedValues[projectionIndex])
                                {
                                    size_t pos = std::min(size_t(f * float(c_imageSize)), c_imageSize - 1);
                                    histogram[pos] += 1.0f;
                                    maxCount = std::max(maxCount, histogram[pos]);
                                }

                                for (float& f : histogram)
                                    f /= maxCount;

                                for (size_t index = 0; index < c_imageSize; ++index)
                                {
                                    if (histogram[index] == 0.0f)
                                        continue;

                                    float pixel = 64.0f - (2.0f + histogram[index] * 60.0f);
                                    DrawPoint(projectedPointsImage, int(index), int(pixel), 0);
                                }

                                #if SHOW_ROTATED_PROJECTIONS()
                                {

                                    std::vector<Vec2> rotatedPoints = points;
                                    for (Vec2& v : rotatedPoints)
                                        v = (Rotate(v - 0.5f, -angle) / projectionLength) + 0.5f;
                                    std::vector<float> imageRotated = MakeSampleImage(rotatedPoints, c_imageSize);
                                    FloatToImage(imageRotated, c_imageSize, c_imageSize, imageRotatedU8);

                                    AppendImageVertical(imageRotatedU8, projectedPointsImage);
                                    AppendImageVertical(imageRotatedU8, separatorH);
                                }
                                #endif

                                // append the image and separator
                                AppendImageVertical(imageU8, projectedPointsImage);
                                AppendImageVertical(imageU8, separatorH);
                            }

                            // make frequency domain images (right side)
                            {
                                Image projectedPointsDFTImage(c_imageSize, 64, 255);

                                std::array<float, c_imageSize> values;
                                std::array<size_t, c_imageSize> valueCount = {};
                                for (size_t index = 0; index < projectionDFTs[projectionIndex].size(); ++index)
                                {
                                    size_t bucket = index * c_imageSize / projectionDFTs[projectionIndex].size();
                                    valueCount[bucket]++;
                                    values[bucket] = Lerp(values[bucket], projectionDFTs[projectionIndex][index], 1.0f / float(valueCount[bucket]));
                                }

                                // get the maximum value
                                float maxValue = 0.0f;
                                for (float f : values)
                                    maxValue = std::max(maxValue, f);

                                for (size_t index = 0; index < c_imageSize; ++index)
                                {
                                    float f = values[index] / maxValue;
                                    float pixel = 64.0f - f * 64.0f;
                                    DrawPoint(projectedPointsDFTImage, int(index), int(pixel), 0);
                                }

                                #if SHOW_ROTATED_PROJECTIONS()
                                {
                                    AppendImageVertical(imageRotatedU8, projectedPointsDFTImage);

                                    sprintf(fileName, "%s_%zu.png", baseFileName, projectionIndex);
                                    SaveImage(fileName, imageRotatedU8);
                                }
                                #endif

                                // TODO: put a mark at 0 hz. it's in the middle

                                // append the image and the separator
                                AppendImageVertical(imageDFTU8, projectedPointsDFTImage);
                                AppendImageVertical(imageDFTU8, separatorH);
                            }
                        }

                        Image seperatorV(1, imageU8.m_height, 128);
                        AppendImageHorizontal(imageU8, seperatorV);
                        AppendImageHorizontal(imageU8, imageDFTU8);
                        sprintf(fileName, "%s_one.png", baseFileName);
                        SaveImage(fileName, imageU8);

                        // TODO: put radial averaged into the image too i think? or make another image.
                        sprintf(fileName, "%s_one.csv", baseFileName);
                        SaveCSV(fileName, radialAveraged);
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

    // report the averages
    #if DO_AVERAGE_TEST()
    {
        // TODO: get radially averaged data in here too!

        // combine the work of all the threads
        std::vector<float> DFTMagnitudes2d_avg = DFTMagnitudes2d[0];
        std::array<std::vector<float>, c_numProjections> DFTMagnitudes1d_avg = DFTMagnitudes1d[0];

        for (size_t testIndex = 1; testIndex < NUM_TESTS(); ++testIndex)
        {
            IncrementalAverage(DFTMagnitudes2d[testIndex], DFTMagnitudes2d_avg, testIndex);

            for (size_t projectionIndex = 0; projectionIndex < c_numProjections; ++projectionIndex)
                IncrementalAverage(DFTMagnitudes1d[testIndex][projectionIndex], DFTMagnitudes1d_avg[projectionIndex], testIndex);
        }

        std::vector<float> DFTMagnitudes2d_avg_normalized;
        NormalizeDFT(DFTMagnitudes2d_avg, DFTMagnitudes2d_avg_normalized);

        std::array<std::vector<float>, c_numProjections> DFTMagnitudes1d_avg_normalized;
        for (size_t projectionIndex = 0; projectionIndex < c_numProjections; ++projectionIndex)
            NormalizeDFT(DFTMagnitudes1d_avg[projectionIndex], DFTMagnitudes1d_avg_normalized[projectionIndex]);

        Image out;
        FloatToImage(DFTMagnitudes2d_avg_normalized, c_imageSize, c_imageSize, out);

        Image separatorH(c_imageSize, 1, 128);
        for (size_t projectionIndex = 0; projectionIndex < c_numProjections; ++projectionIndex)
        {
            // make frequency domain images (right side)
            Image projectedPointsDFTImage(c_imageSize, 64, 255);

            std::array<float, c_imageSize> values;
            std::array<size_t, c_imageSize> valueCount = {};
            for (size_t index = 0; index < DFTMagnitudes1d_avg_normalized[projectionIndex].size(); ++index)
            {
                size_t bucket = index * c_imageSize / DFTMagnitudes1d_avg_normalized[projectionIndex].size();
                valueCount[bucket]++;
                values[bucket] = Lerp(values[bucket], DFTMagnitudes1d_avg_normalized[projectionIndex][index], 1.0f / float(valueCount[bucket]));
            }

            // get the maximum value
            float maxValue = 0.0f;
            for (float f : values)
                maxValue = std::max(maxValue, f);

            for (size_t index = 0; index < c_imageSize; ++index)
            {
                float f = values[index] / maxValue;
                float pixel = 64.0f - f * 64.0f;
                DrawPoint(projectedPointsDFTImage, int(index), int(pixel), 0);
            }

            // TODO: put a mark at 0 hz. it's in the middle

            // append the image and the separator
            AppendImageVertical(out, projectedPointsDFTImage);
            AppendImageVertical(out, separatorH);
        }

        char fileName[1024];
        sprintf(fileName, "%s_DFT_avg.png", baseFileName);
        SaveImage(fileName, out);

        // TODO: radial averages too
    }

    // TODO: make a single image per test that has samples, dft, dft avg

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
        "Progressive Projective Blue Noise Min 5",
        "out/BN_ProgProjMin_5",
        [](std::mt19937& rng, std::vector<Vec2>& points)
        {
            GoodCandidateSubspaceAlgorithmAccell_Min<2, c_progProjAccelSize, false>(rng, points, c_sampleCount, 5, false);
        }
    );

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

* append horizontal the average image to the main image?

* a problem with rotated images is that the corners have less area & points. i wonder how this is handled in the eric heitz paper.

* instead of adding seperator lines in the images, add single pixel width / height images that do that. this means no overlap

* you aren't doing the projection normalization right.
 * it needs to be shrunk from 0.5 so...
 * subtract 0.5 to go from [0,1] to [-0.5,+0.5]. then divide by length / 2 i think.  then add 0.5 again.

* mitchels has some things at "half value" for the first projection.  Are they really having overlap?

* the dft projections may not be correct... or are they? need to think about it & look at the data. maybe look with smaller sample counts?
 * also the regular projections... do they line up? maybe instead of normalizing numerically, should do it analytically, since the first point shouldn't be at zero necesarily etc!

* make a single image per test... left top = points. left below that = projections of points.  right top = dft. right below that = dft of projections

* put projections of points on the point image too. maybe same way the fourier transform goes... the same projections, and put em in images below

 * something else to try: Random center point and random half space. Take the point that has the best discrepancy score? would also need to make sure not too close to existing points. dunno how.

Make 1d fourier transform images instead of csvs. Append them with circle shpwing the projection line

* maybe try dividing 2d distance by sqrt(2)?

* min isn't really min. rename.

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