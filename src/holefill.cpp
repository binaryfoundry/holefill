#include <cstdint>
#include <vector>
#include <set>

#include "holefill.h"

namespace holefill {

struct Coord {
    int32_t x = 0;
    int32_t y = 0;

    bool operator<(const Coord& other) const {
        return (x < other.x) || (x == other.x && y < other.y);
    }
};

inline float getPixel(const float* image, int32_t y, int32_t x, int32_t cols) {
    return image[y * cols + x];
}

std::vector<Coord> findBoundaryPixels(const float* image, const int32_t rows, const int32_t cols, const std::vector<Coord>& holePixels, const bool use8Connectivity = true) {
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
            const int32_t ny = p.x + off.x;
            const int32_t nx = p.y + off.y;

            if (ny >= 0 && ny < rows && nx >= 0 && nx < cols) {
                const Coord neighbor{ny, nx};

                if (holeSet.count(neighbor) == 0 && getPixel(image, ny, nx, cols) >= 0.0f) {
                    if (boundarySet.insert(neighbor).second) {
                        boundaryPixels.push_back(neighbor);
                    }
                }
            }
        }
    }

    return boundaryPixels;
};

} // namespace holefill

