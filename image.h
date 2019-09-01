#pragma once

#include <vector>
#include <complex>
#include <stdint.h>
#include <thread>
#include <atomic>
#include <algorithm>

#include "math.h"

#include "stb/stb_image_write.h"

typedef uint8_t uint8;

// -------------------------------------------------------------------------------

struct ImageGrey
{
    ImageGrey(int width = 0, int height = 0, uint8 fill = 255)
    {
        m_width = width;
        m_height = height;
        m_pixels.resize(m_width*m_height);
        std::fill(m_pixels.begin(), m_pixels.end(), fill);
    }

    int m_width;
    int m_height;
    std::vector<uint8> m_pixels;
};

// -------------------------------------------------------------------------------

struct Image
{
    Image(int width=0, int height=0)
    {
        m_width = width;
        m_height = height;
        m_pixels.resize(m_width*m_height * 4); // 4 channels per pixel
        std::fill(m_pixels.begin(), m_pixels.end(), 255);
    }

    int m_width;
    int m_height;
    std::vector<uint8> m_pixels;
};

// -------------------------------------------------------------------------------

struct ImageFloat
{
    ImageFloat(int width = 0, int height = 0)
    {
        m_width = width;
        m_height = height;
        m_pixels.resize(m_width*m_height * 4); // 4 channels per pixel
        std::fill(m_pixels.begin(), m_pixels.end(), 1.0f);
    }

    int m_width;
    int m_height;
    std::vector<float> m_pixels;
};

// -------------------------------------------------------------------------------

inline float Clamp(float x, float min, float max)
{
    if (x <= min)
        return min;
    else if (x >= max)
        return max;
    else
        return x;
}

// -------------------------------------------------------------------------------
inline void SaveImage(const char* fileName, Image& image)
{
    stbi_write_png(fileName, image.m_width, image.m_height, 4, image.m_pixels.data(), 0);
}

// -------------------------------------------------------------------------------
inline void SaveImage(const char* fileName, ImageGrey& image)
{
    stbi_write_png(fileName, image.m_width, image.m_height, 1, image.m_pixels.data(), 0);
}

// -------------------------------------------------------------------------------

inline float SmoothStep(float value, float min, float max)
{
    float x = (value - min) / (max - min);
    x = std::min(x, 1.0f);
    x = std::max(x, 0.0f);

    return 3.0f * x * x - 2.0f * x * x * x;
}

// -------------------------------------------------------------------------------

inline void DrawPoint(ImageGrey& image, int x, int y, uint8 C)
{
    x = std::max(x, 0);
    x = std::min(x, image.m_width-1);
    y = std::max(y, 0);
    y = std::min(y, image.m_height-1);
    image.m_pixels[y * image.m_width + x] = C;
}

inline void DrawAAB(ImageGrey& image, int x1, int y1, int x2, int y2, uint8 C)
{
    for (int y = y1; y <= y2; ++y)
    {
        uint8* pixel = &image.m_pixels[y * image.m_width + x1];
        for (int x = x1; x <= x2; ++x)
        {
            *pixel = C;
            pixel++;
        }
    }
}

inline void DrawLine(ImageGrey& image, int x1, int y1, int x2, int y2, uint8 C)
{
    // pad the AABB of pixels we scan, to account for anti aliasing
    int startX = std::max(std::min(x1, x2) - 4, 0);
    int startY = std::max(std::min(y1, y2) - 4, 0);
    int endX = std::min(std::max(x1, x2) + 4, image.m_width - 1);
    int endY = std::min(std::max(y1, y2) + 4, image.m_height - 1);

    // if (x1,y1) is A and (x2,y2) is B, get a normalized vector from A to B called AB
    float ABX = float(x2 - x1);
    float ABY = float(y2 - y1);
    float ABLen = std::sqrtf(ABX*ABX + ABY * ABY);
    ABX /= ABLen;
    ABY /= ABLen;

    // scan the AABB of our line segment, drawing pixels for the line, as is appropriate
    for (int iy = startY; iy <= endY; ++iy)
    {
        uint8* pixel = &image.m_pixels[(iy * image.m_width + startX)];
        for (int ix = startX; ix <= endX; ++ix)
        {
            // project this current pixel onto the line segment to get the closest point on the line segment to the point
            float ACX = float(ix - x1);
            float ACY = float(iy - y1);
            float lineSegmentT = ACX * ABX + ACY * ABY;
            lineSegmentT = std::min(lineSegmentT, ABLen);
            lineSegmentT = std::max(lineSegmentT, 0.0f);
            float closestX = float(x1) + lineSegmentT * ABX;
            float closestY = float(y1) + lineSegmentT * ABY;

            // calculate the distance from this pixel to the closest point on the line segment
            float distanceX = float(ix) - closestX;
            float distanceY = float(iy) - closestY;
            float distance = std::sqrtf(distanceX*distanceX + distanceY * distanceY);

            // use the distance to figure out how transparent the pixel should be, and apply the color to the pixel
            float alpha = SmoothStep(distance, 2.0f, 0.0f);

            if (alpha > 0.0f)
            {
                pixel[0] = Lerp(pixel[0], C, alpha);
            }

            pixel ++;
        }
    }
}

// -------------------------------------------------------------------------------

inline void ClearImage(Image& image, uint8 R, uint8 G, uint8 B)
{
    uint8* pixel = image.m_pixels.data();
    for (int i = 0, c = image.m_width * image.m_height; i < c; ++i)
    {
        pixel[0] = R;
        pixel[1] = G;
        pixel[2] = B;
        pixel[3] = 255;
        pixel += 4;
    }
}

// -------------------------------------------------------------------------------
inline void AppendImageVertical(Image& result, const Image& top, const Image& bottom)
{
    int width = std::max(top.m_width, bottom.m_width);
    int height = top.m_height + bottom.m_height;
    result = Image(width, height);

    // top image
    {
        const uint8* srcRow = top.m_pixels.data();
        for (int y = 0; y < top.m_height; ++y)
        {
            uint8* destRow = &result.m_pixels[y*width * 4];
            memcpy(destRow, srcRow, top.m_width * 4);
            srcRow += top.m_width * 4;
        }
    }

    // bottom image
    {
        const uint8* srcRow = bottom.m_pixels.data();
        for (int y = 0; y < bottom.m_height; ++y)
        {
            uint8* destRow = &result.m_pixels[(y+top.m_height)*width * 4];
            memcpy(destRow, srcRow, bottom.m_width * 4);
            srcRow += bottom.m_width * 4;
        }
    }
}

// -------------------------------------------------------------------------------
inline void AppendImageVertical(ImageGrey& result, const ImageGrey& top, const ImageGrey& bottom)
{
    int width = std::max(top.m_width, bottom.m_width);
    int height = top.m_height + bottom.m_height;
    result = ImageGrey(width, height);

    // top image
    {
        const uint8* srcRow = top.m_pixels.data();
        for (int y = 0; y < top.m_height; ++y)
        {
            uint8* destRow = &result.m_pixels[y*width];
            memcpy(destRow, srcRow, top.m_width);
            srcRow += top.m_width;
        }
    }

    // bottom image
    {
        const uint8* srcRow = bottom.m_pixels.data();
        for (int y = 0; y < bottom.m_height; ++y)
        {
            uint8* destRow = &result.m_pixels[(y + top.m_height)*width];
            memcpy(destRow, srcRow, bottom.m_width);
            srcRow += bottom.m_width;
        }
    }
}

// -------------------------------------------------------------------------------
inline void AppendImageHorizontal(ImageGrey& result, const ImageGrey& left, const ImageGrey& right)
{
    int width = left.m_width + right.m_width;
    int height = std::max(left.m_height, right.m_height);
    result = ImageGrey(width, height);

    // left image
    {
        const uint8* srcRow = left.m_pixels.data();
        for (int y = 0; y < left.m_height; ++y)
        {
            uint8* destRow = &result.m_pixels[y*width];
            memcpy(destRow, srcRow, left.m_width);
            srcRow += left.m_width;
        }
    }

    // bottom image
    {
        const uint8* srcRow = right.m_pixels.data();
        for (int y = 0; y < right.m_height; ++y)
        {
            uint8* destRow = &result.m_pixels[y * width + left.m_width];
            memcpy(destRow, srcRow, right.m_width);
            srcRow += right.m_width;
        }
    }
}

// -------------------------------------------------------------------------------

inline void DrawCircle(Image& image, int cx, int cy, int radius, uint8 R, uint8 G, uint8 B)
{
    int startX = std::max(cx - radius - 4, 0);
    int startY = std::max(cy - radius - 4, 0);
    int endX = std::min(cx + radius + 4, image.m_width - 1);
    int endY = std::min(cy + radius + 4, image.m_height - 1);

    for (int iy = startY; iy <= endY; ++iy)
    {
        float dy = float(cy - iy);
        uint8* pixel = &image.m_pixels[(iy * image.m_width + startX) * 4];
        for (int ix = startX; ix <= endX; ++ix)
        {
            float dx = float(cx - ix);

            float distance = std::max(std::sqrtf(dx * dx + dy * dy) - float(radius), 0.0f);

            float alpha = SmoothStep(distance, 2.0f, 0.0f);

            if (alpha > 0.0f)
            {
                pixel[0] = Lerp(pixel[0], R, alpha);
                pixel[1] = Lerp(pixel[1], G, alpha);
                pixel[2] = Lerp(pixel[2], B, alpha);
            }

            pixel += 4;
        }
    }
}

// -------------------------------------------------------------------------------

inline void DrawCircle(ImageGrey& image, int cx, int cy, int radius, uint8 C)
{
    int startX = std::max(cx - radius - 4, 0);
    int startY = std::max(cy - radius - 4, 0);
    int endX = std::min(cx + radius + 4, image.m_width - 1);
    int endY = std::min(cy + radius + 4, image.m_height - 1);

    for (int iy = startY; iy <= endY; ++iy)
    {
        float dy = float(cy - iy);
        uint8* pixel = &image.m_pixels[(iy * image.m_width + startX)];
        for (int ix = startX; ix <= endX; ++ix)
        {
            float dx = float(cx - ix);

            float distance = std::max(std::sqrtf(dx * dx + dy * dy) - float(radius), 0.0f);

            float alpha = SmoothStep(distance, 2.0f, 0.0f);

            if (alpha > 0.0f)
            {
                pixel[0] = Lerp(pixel[0], C, alpha);
            }

            pixel ++;
        }
    }
}

// -------------------------------------------------------------------------------

inline void ImageFloatToImage(const ImageFloat& imageFloat, Image& image)
{
    image = Image(imageFloat.m_width, imageFloat.m_height);

    for (size_t pixelIndex = 0; pixelIndex < imageFloat.m_width * imageFloat.m_height; ++pixelIndex)
    {
        for (size_t channelIndex = 0; channelIndex < 3; ++channelIndex)
        {
            float valueLinear = Clamp(imageFloat.m_pixels[pixelIndex * 4 + channelIndex], 0.0f, 1.0f);
            float valuesRGB = powf(valueLinear, 1.0f / 2.2f);
            image.m_pixels[pixelIndex * 4 + channelIndex] = uint8(valuesRGB * 255.0f + 0.5f);
        }

        image.m_pixels[pixelIndex * 4 + 3] = uint8(Clamp(imageFloat.m_pixels[pixelIndex * 4 + 3], 0.0f, 1.0f) * 255.0f + 0.5f);
    }
}

// -------------------------------------------------------------------------------

inline void ImageToImageGrey(const Image& src, ImageGrey& dest)
{
    // Just taking the R channel is fine enough for our needs.

    dest = ImageGrey(src.m_width, src.m_height);

    for (size_t pixelIndex = 0; pixelIndex < src.m_width * src.m_height; ++pixelIndex)
        dest.m_pixels[pixelIndex] = src.m_pixels[pixelIndex * 4];
}

inline void ImageGreyToImage(const ImageGrey& src, Image& dest)
{
    // put it into RGB and make alpha 255. good enough for our needs

    dest = Image(src.m_width, src.m_height);

    for (size_t pixelIndex = 0; pixelIndex < src.m_width * src.m_height; ++pixelIndex)
    {
        dest.m_pixels[pixelIndex * 4 + 0] = src.m_pixels[pixelIndex];
        dest.m_pixels[pixelIndex * 4 + 1] = src.m_pixels[pixelIndex];
        dest.m_pixels[pixelIndex * 4 + 2] = src.m_pixels[pixelIndex];
        dest.m_pixels[pixelIndex * 4 + 3] = 255;
    }
}

inline void FloatToImageGrey(const std::vector<float>& src, int width, int height, ImageGrey& dest)
{
    dest = ImageGrey(width, height);

    for (size_t index = 0, count = dest.m_pixels.size(); index < count; ++index)
    {
        float valueFloat = powf(src[index], 1.0f / 2.2f);
        dest.m_pixels[index] = uint8_t(valueFloat * 255.0f + 0.5f);
    }
}

// -------------------------------------------------------------------------------

inline float MeanSquaredError(const ImageFloat& A, const ImageFloat& B)
{
    float meanSquaredError = 0.0f;
    const float* valueA = A.m_pixels.data();
    const float* valueB = B.m_pixels.data();
    for (size_t index = 1, count = A.m_width*A.m_height * 4; index <= count; ++index)
    {
        float squaredError = (*valueA - *valueB) * (*valueA - *valueB);
        meanSquaredError = Lerp(meanSquaredError, squaredError, 1.0f / float(index));

        ++valueA;
        ++valueB;
    }
    return meanSquaredError;
}
