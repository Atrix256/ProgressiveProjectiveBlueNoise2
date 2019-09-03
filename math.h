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

inline float dot(const Vec2& a, const Vec2& b)
{
    return a[0] * b[0] + a[1] * b[1];
}
