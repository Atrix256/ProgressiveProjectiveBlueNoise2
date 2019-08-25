#pragma once

#include "simple_fft/fft_settings.h"
#include "simple_fft/fft.h"

#include <algorithm>
#include <vector>

struct ComplexImage2D
{
    ComplexImage2D(size_t w, size_t h)
    {
        m_width = w;
        m_height = h;
        pixels.resize(w*h, real_type(0.0f));
    }

    size_t m_width;
    size_t m_height;
    std::vector<complex_type> pixels;

    complex_type& operator()(size_t x, size_t y)
    {
        return pixels[y*m_width + x];
    }

    const complex_type& operator()(size_t x, size_t y) const
    {
        return pixels[y*m_width + x];
    }
};

template <typename T>
T Lerp(T A, T B, float t)
{
    return T(float(A) * (1.0f - t) + float(B) * t);
}

void RadiallyAveragePowerSpectrum(const std::vector<float>& imageSrc, size_t imageWidth, std::vector<float>& output, size_t outputCount)
{
    output.clear();
    output.resize(outputCount, 0.0f);

    std::vector<size_t> outputCounts;
    outputCounts.resize(outputCount, 0);

    for (size_t index = 0; index < imageSrc.size(); ++index)
    {
        float x = float(index % imageWidth) - float(imageWidth / 2);
        float y = float(index / imageWidth) - float(imageWidth / 2);

        // ignore DC
        if (x == 0 && y == 0)
            continue;

        float distance = sqrt(x*x + y * y);
        distance /= float(imageWidth / 2);
        distance /= sqrtf(2.0f);

        size_t bucket = size_t(distance * float(outputCount));
        if (bucket >= outputCount)
            bucket = outputCount - 1;

        outputCounts[bucket]++;
        output[bucket] = Lerp(output[bucket], imageSrc[index], 1.0f / float(outputCounts[bucket]));
    }

    // get the maximum value in the output
    float maxValue = output[0];
    for (float f : output)
    {
        if (f > maxValue)
            maxValue = f;
    }

    // normalize the output
    for (float& f : output)
        f /= maxValue;
}

void DFTPeriodogram(const std::vector<float>& imageSrc, std::vector<float>& imageDest, size_t width, size_t numSamples_, std::vector<float>& radialAverage, size_t radialAverageCount)
{
    // convert the source image to complex so it can be DFTd
    ComplexImage2D complexImageIn(width, width);
    for (size_t index = 0, count = width * width; index < count; ++index)
        complexImageIn.pixels[index] = imageSrc[index];

    // DFT the image to get frequency of the samples
    const char* error = nullptr;
    ComplexImage2D complexImageOut(width, width);
    simple_fft::FFT(complexImageIn, complexImageOut, width, width, error);

    // get the magnitudes and max magnitude
    float numSamples = float(numSamples_);
    std::vector<float> magnitudes;
    float maxMag = 0.0f;
    {
        magnitudes.resize(width * width, 0.0f);
        float* dest = magnitudes.data();
        for (size_t y = 0; y < width; ++y)
        {
            size_t srcY = (y + width / 2) % width;
            for (size_t x = 0; x < width; ++x)
            {
                size_t srcX = (x + width / 2) % width;

                const complex_type& c = complexImageOut(srcX, srcY);
                float mag = float(sqrt(c.real()*c.real() + c.imag()*c.imag())) / numSamples;
                maxMag = std::max(mag, maxMag);
                *dest = mag;
                ++dest;
            }
        }
    }

    // make the radially averaged power spectrum
    RadiallyAveragePowerSpectrum(magnitudes, width, radialAverage, radialAverageCount);

    // normalize the magnitudes
    const float c = 1.0f / log(1.0f + maxMag);
    {
        imageDest.resize(width * width);
        const float* src = magnitudes.data();
        float* dest = imageDest.data();
        for (size_t y = 0; y < width; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                float normalized = c * log(1.0f + *src);
                *dest = *src / maxMag;

                ++src;
                ++dest;
            }
        }
    }
}
