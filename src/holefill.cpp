#include <vector>
#include <set>
#include <limits>
#include <cmath>

#include "holefill.h"
#include "nanoflann.hpp"

namespace holefill {

inline float getPixel(const float* const image, const int32_t x, const int32_t y, const int32_t width) {
    return image[y * width + x];
}

std::vector<Coord> findBoundaryPixels(const float* const image, const int32_t width, const int32_t height, const std::vector<Coord>& holePixels, const bool use8Connectivity = true) {
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

std::vector<Coord> findHolePixels(const float* const image, const uint32_t width, const uint32_t height) {
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

void fill(float* const image, const int32_t width, const int32_t height, const WeightFunction weightFunc) {
    const std::vector<Coord> holePixels = findHolePixels(image, width, height);
    const std::vector<Coord> boundaryPixels = findBoundaryPixels(image, width, height, holePixels);

    for (const auto& u : holePixels) {
        float numerator = 0.0f;
        float denominator = 0.0f;

        for (const auto& v : boundaryPixels) {
            const float w = weightFunc(u, v);
            const float intensity = getPixel(image, v.x, v.y, width);
            numerator += w * intensity;
            denominator += w;
        }

        image[u.y * width + u.x] = (denominator > std::numeric_limits<float>::epsilon())
            ? numerator / denominator
            : 0.0f;  // Fallback value
    }
}

void fillApproximate(float* const image, const int32_t width, const int32_t height, const WeightFunction weightFunc, const int32_t windowSize) {
    const int32_t halfWindow = windowSize / 2;
    const std::vector<Coord> holePixels = findHolePixels(image, width, height);

    for (const auto& u : holePixels) {
        float numerator = 0.0f;
        float denominator = 0.0f;

        // Loop over fixed-size window
        for (int32_t dy = -halfWindow; dy <= halfWindow; ++dy) {
            for (int32_t dx = -halfWindow; dx <= halfWindow; ++dx) {
                const int32_t nx = u.x + dx;
                const int32_t ny = u.y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    const float value = image[ny * width + nx];
                    if (value >= 0.0f) {  // Valid pixel (not part of hole)
                        const Coord v{nx, ny};
                        const float weight = weightFunc(u, v);
                        numerator += weight * value;
                        denominator += weight;
                    }
                }
            }
        }

        image[u.y * width + u.x] = (denominator > std::numeric_limits<float>::epsilon())
            ? numerator / denominator
            : 0.0f;  // Fallback value
    }
}

// Adaptor for nanoflann
struct CoordCloud {
    std::vector<Coord> points;

    size_t kdtree_get_point_count() const { return points.size(); }

    float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return (dim == 0) ? static_cast<float>(points[idx].x)
                          : static_cast<float>(points[idx].y);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

void fillExactWithSearch(float* const image, const int32_t width, const int32_t height,
                         const WeightFunction weightFunc) {
    const std::vector<Coord> holePixels = findHolePixels(image, width, height);
    const std::vector<Coord> boundaryPixels = findBoundaryPixels(image, width, height, holePixels);

    // Calculate hole center and radius
    float centerX = 0.0f, centerY = 0.0f;
    for (const Coord& p : holePixels) {
        centerX += p.x;
        centerY += p.y;
    }
    centerX /= holePixels.size();
    centerY /= holePixels.size();

    // Calculate maximum distance from center to any hole pixel
    float maxDistSq = 0.0f;
    for (const Coord& p : holePixels) {
        const float dx = p.x - centerX;
        const float dy = p.y - centerY;
        const float distSq = dx * dx + dy * dy;
        maxDistSq = std::max(maxDistSq, distSq);
    }
    const float radius = std::sqrt(maxDistSq) * 1.5f;  // Add 50% margin to ensure we capture all relevant boundary pixels

    CoordCloud cloud;
    cloud.points = boundaryPixels;

    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, CoordCloud>,
        CoordCloud, 2, size_t>;

    KDTree tree(2, cloud, {10});
    tree.buildIndex();

    for (const Coord& u : holePixels) {
        const float queryPt[2] = { static_cast<float>(u.x), static_cast<float>(u.y) };

        std::vector<nanoflann::ResultItem<size_t, float>> matches;
        const float search_radius_sq = radius * radius;
        tree.radiusSearch(queryPt, search_radius_sq, matches);

        float numerator = 0.0f;
        float denominator = 0.0f;

        for (const auto& match : matches) {
            const Coord& v = cloud.points[match.first];
            const float w = weightFunc(u, v);
            const float intensity = image[v.y * width + v.x];
            numerator += w * intensity;
            denominator += w;
        }

        image[u.y * width + u.x] = (denominator > std::numeric_limits<float>::epsilon())
            ? numerator / denominator
            : 0.0f;  // Fallback value
    }
}

} // namespace holefill

