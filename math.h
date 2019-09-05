#pragma once

static const float c_pi = 3.14159265359f;

template <typename T>
T Lerp(T A, T B, float t)
{
    return T(float(A) * (1.0f - t) + float(B) * t);
}

typedef std::array<float, 2> Vec2;

inline Vec2 operator * (const Vec2& v, float f)
{
    return Vec2{ v[0] * f, v[1] * f };
}

inline Vec2 operator - (const Vec2& v, float f)
{
    return Vec2{ v[0] - f, v[1] - f };
}

inline Vec2 operator + (const Vec2& v, float f)
{
    return Vec2{ v[0] + f, v[1] + f };
}

inline Vec2 operator / (const Vec2& v, float f)
{
    return Vec2{ v[0] / f, v[1] / f };
}

inline float Dot(const Vec2& a, const Vec2& b)
{
    return a[0] * b[0] + a[1] * b[1];
}

inline Vec2 Rotate(const Vec2& v, float theta)
{
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);

    Vec2 ret;
    ret[0] = v[0] * cosTheta - v[1] * sinTheta;
    ret[1] = v[0] * sinTheta + v[1] * cosTheta;
    return ret;
}
