#pragma once

static const float c_pi = 3.14159265359f;

template <typename T>
T Lerp(T A, T B, float t)
{
    return T(float(A) * (1.0f - t) + float(B) * t);
}
