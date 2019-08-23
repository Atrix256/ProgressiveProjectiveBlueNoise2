#define _CRT_SECURE_NO_WARNINGS

#include <array>
#include <vector>
#include <random>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "BN_Mitchels.h"
#include "BN_progproj.h"
#include "dft.h"

static const size_t c_sampleCount = 1000;
static const size_t c_imageSize = 256;

// Mitchel's best candidate blue noise settings
static const size_t c_mitchelCandidateMultiplier = 1;

// Progressive Projective blue noise settings
static const size_t c_progProjAccelSize = 10;
static const size_t c_progProjCandidateMultiplier = 100;
// TODO: are the numbers above properly tuned?


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

std::vector<uint8_t> ImageFloatToRGBAU8(const std::vector<float>& image, size_t imageResolution)
{
    std::vector<uint8_t> ret(imageResolution*imageResolution);
    for (size_t index = 0, count = image.size(); index < count; ++index)
    {
        float valueFloat = powf(image[index], 1.0f / 2.2f);
        ret[index] = uint8_t(valueFloat * 255.0f + 0.5f);
    }
    return ret;
}

int main(int argc, char** argv)
{
    {
        std::vector<Vec2> points;
        GoodCandidateSubspaceAlgorithmAccell<2, c_progProjAccelSize, false>(points, c_sampleCount, c_progProjCandidateMultiplier, true);
        std::vector<float> image = MakeSampleImage(points, c_imageSize);
        std::vector<float> imageDFT;
        DFT(image, imageDFT, c_imageSize);
        std::vector<uint8_t> imageU8 = ImageFloatToRGBAU8(image, c_imageSize);
        std::vector<uint8_t> imageDFTU8 = ImageFloatToRGBAU8(imageDFT, c_imageSize);

        stbi_write_png("out/BN_ProgProj.png", int(c_imageSize), int(c_imageSize), 1, imageU8.data(), 0);
        stbi_write_png("out/BN_ProgProj_DFT.png", int(c_imageSize), int(c_imageSize), 1, imageDFTU8.data(), 0);
    }

    {
        std::vector<Vec2> points;
        MitchelsBestCandidateAlgorithm<2>(points, c_sampleCount, c_mitchelCandidateMultiplier);
        std::vector<float> image = MakeSampleImage(points, c_imageSize);
        std::vector<float> imageDFT;
        DFT(image, imageDFT, c_imageSize);
        std::vector<uint8_t> imageU8 = ImageFloatToRGBAU8(image, c_imageSize);
        std::vector<uint8_t> imageDFTU8 = ImageFloatToRGBAU8(imageDFT, c_imageSize);

        stbi_write_png("out/BN_Mitchels.png", int(c_imageSize), int(c_imageSize), 1, imageU8.data(), 0);
        stbi_write_png("out/BN_Mitchels_DFT.png", int(c_imageSize), int(c_imageSize), 1, imageDFTU8.data(), 0);
    }

    system("pause");
    return 0;
}

/*

TODO:

* do radial periodogram also. Maybe min, max and average?
* compare vs the "extra penalty"

* maybe do the log coloring of the DFT!

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