#pragma once

#include <cstdint>
#include <functional>
#include <cmath>

namespace holefill {

struct Coord {
    int32_t x = 0;
    int32_t y = 0;

    bool operator<(const Coord& other) const {
        return (x < other.x) || (x == other.x && y < other.y);
    }
};

using WeightFunction = std::function<float(const Coord&, const Coord&)>;

void fill(float* image, const int32_t width, const int32_t height, 
          WeightFunction weightFunc);

void fillApproximate(float* image, const int32_t width, const int32_t height, WeightFunction weightFunc, int windowSize = -1);

} // namespace holefill
