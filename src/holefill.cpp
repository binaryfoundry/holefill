#include <vector>
#include <set>
#include <limits>
#include <cmath>

#include "holefill.h"

namespace holefill {

inline float getPixel(const float* image, int32_t x, int32_t y, int32_t width) {
    return image[y * width + x];
}

std::vector<Coord> findBoundaryPixels(const float* image, const int32_t width, const int32_t height, const std::vector<Coord>& holePixels, const bool use8Connectivity = true) {
    std::vector<Coord> boundaryPixels;
    std::set<Coord> holeSet(holePixels.begin(), holePixels.end());
    std::set<Coord> boundarySet;  // to avoid duplicates

    // Neighbor offsets
    std::vector<Coord> offsets = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    };
    if (use8Connectivity) {
        offsets.insert(offsets.end(), {
            {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
        });
    }

    for (const Coord& p : holePixels) {
        for (const Coord& off : offsets) {
            const int32_t nx = p.x + off.x;
            const int32_t ny = p.y + off.y;

            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                const Coord neighbor{nx, ny};

                if (holeSet.count(neighbor) == 0 && getPixel(image, nx, ny, width) >= 0.0f) {
                    if (boundarySet.insert(neighbor).second) {
                        boundaryPixels.push_back(neighbor);
                    }
                }
            }
        }
    }

    return boundaryPixels;
};

std::vector<Coord> findHolePixels(const float* image, uint32_t width, uint32_t height) {
    std::vector<Coord> holePixels;

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (getPixel(image, x, y, width) < 0.0f) {
                holePixels.emplace_back(x, y);
            }
        }
    }

    return holePixels;
}

void fill(float* image, const int32_t width, const int32_t height, WeightFunction weightFunc) {
    const std::vector<Coord> holePixels = findHolePixels(image, width, height);
    const std::vector<Coord> boundaryPixels = findBoundaryPixels(image, width, height, holePixels);

    for (const auto& u : holePixels) {
        float numerator = 0.0f;
        float denominator = 0.0f;

        for (const auto& v : boundaryPixels) {
            float w = weightFunc(u, v);
            float intensity = getPixel(image, v.x, v.y, width);
            numerator += w * intensity;
            denominator += w;
        }

        if (std::abs(denominator) > std::numeric_limits<float>::epsilon())
            image[u.y * width + u.x] = numerator / denominator;
        else
            image[u.y * width + u.x] = 0.0f; // fallback
    }
}

} // namespace holefill

