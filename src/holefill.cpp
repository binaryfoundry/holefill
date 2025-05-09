#include <vector>
#include <set>
#include <limits>
#include <cmath>

#include "holefill.h"
#include "nanoflann.hpp"

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

void fillApproximate(float* image, const int32_t width, const int32_t height, WeightFunction weightFunc, int windowSize) {
    const int halfWindow = windowSize / 2;
    const std::vector<Coord> holePixels = findHolePixels(image, width, height);

    for (const auto& u : holePixels) {
        float numerator = 0.0f;
        float denominator = 0.0f;

        // Loop over fixed-size window
        for (int dy = -halfWindow; dy <= halfWindow; ++dy) {
            for (int dx = -halfWindow; dx <= halfWindow; ++dx) {
                int nx = u.x + dx;
                int ny = u.y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float value = image[ny * width + nx];
                    if (value >= 0.0f) {  // Valid pixel (not part of hole)
                        Coord v{nx, ny};
                        float weight = weightFunc(u, v);
                        numerator += weight * value;
                        denominator += weight;
                    }
                }
            }
        }

        if (denominator > std::numeric_limits<float>::epsilon())
            image[u.y * width + u.x] = numerator / denominator;
        else
            image[u.y * width + u.x] = 0.0f; // fallback value
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

void fillExactWithSearch(float* image, int32_t width, int32_t height,
                         WeightFunction weightFunc,
                         float radius) {
    const std::vector<Coord> holePixels = findHolePixels(image, width, height);
    const std::vector<Coord> boundaryPixels = findBoundaryPixels(image, width, height, holePixels);

    CoordCloud cloud;
    cloud.points = boundaryPixels;

    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, CoordCloud>,
        CoordCloud, 2, size_t>;

    KDTree tree(2, cloud, {10});
    tree.buildIndex();

    for (const Coord& u : holePixels) {
        float queryPt[2] = { static_cast<float>(u.x), static_cast<float>(u.y) };

        std::vector<nanoflann::ResultItem<size_t, float>> matches;
        const float search_radius_sq = radius * radius;
        tree.radiusSearch(queryPt, search_radius_sq, matches);

        float numerator = 0.0f;
        float denominator = 0.0f;

        for (const auto& match : matches) {
            const Coord& v = cloud.points[match.first];
            float w = weightFunc(u, v);
            float intensity = image[v.y * width + v.x];
            numerator += w * intensity;
            denominator += w;
        }

        if (denominator > std::numeric_limits<float>::epsilon())
            image[u.y * width + u.x] = numerator / denominator;
        else
            image[u.y * width + u.x] = 0.0f;
    }
}

} // namespace holefill

