#include <vector>
#include <set>
#include <limits>
#include <cmath>
#include <queue>

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

struct HoleInfo {
    float centerX;
    float centerY;
    float radius;
};

HoleInfo calculateHoleInfo(const std::vector<Coord>& holePixels) {
    HoleInfo info;
    info.centerX = 0.0f;
    info.centerY = 0.0f;

    // Calculate center
    for (const Coord& p : holePixels) {
        info.centerX += p.x;
        info.centerY += p.y;
    }
    info.centerX /= holePixels.size();
    info.centerY /= holePixels.size();

    // Calculate maximum distance from center to any hole pixel
    float maxDistSq = 0.0f;
    for (const Coord& p : holePixels) {
        const float dx = p.x - info.centerX;
        const float dy = p.y - info.centerY;
        const float distSq = dx * dx + dy * dy;
        maxDistSq = std::max(maxDistSq, distSq);
    }
    info.radius = std::sqrt(maxDistSq) * 1.5f;  // Add 50% margin to ensure we capture all relevant boundary pixels

    return info;
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

void fillApproximate(float* const image, const int32_t width, const int32_t height) {
    const std::vector<Coord> holePixels = findHolePixels(image, width, height);
    std::vector<std::vector<bool>> isHole(height, std::vector<bool>(width, false));
    std::queue<Coord> toProcess;

    // Initialize hole mask and queue
    for (const auto& p : holePixels) {
        isHole[p.y][p.x] = true;
    }

    // 4-connected neighbor offsets
    const int32_t offsets[8][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
    };

    // First pass: find boundary pixels and add them to queue
    for (const auto& p : holePixels) {
        for (const auto& offset : offsets) {
            int nx = p.x + offset[0];
            int ny = p.y + offset[1];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (!isHole[ny][nx] && image[ny * width + nx] >= 0.0f) {
                    toProcess.push(p);
                    break;
                }
            }
        }
    }

    // Process pixels in order
    while (!toProcess.empty()) {
        const Coord u = toProcess.front();
        toProcess.pop();

        if (!isHole[u.y][u.x]) continue;  // Skip if already processed

        float sum = 0.0f;
        int32_t count = 0;

        // Calculate average of non-hole neighbors
        for (const auto& offset : offsets) {
            const int32_t nx = u.x + offset[0];
            const int32_t ny = u.y + offset[1];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (!isHole[ny][nx] && image[ny * width + nx] >= 0.0f) {
                    sum += image[ny * width + nx];
                    ++count;
                }
            }
        }

        if (count > 0) {
            image[u.y * width + u.x] = sum / count;
            isHole[u.y][u.x] = false;  // Mark as processed

            // Add unprocessed hole neighbors to queue
            for (const auto& offset : offsets) {
                const int32_t nx = u.x + offset[0];
                const int32_t ny = u.y + offset[1];

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    if (isHole[ny][nx]) {
                        toProcess.push({nx, ny});
                    }
                }
            }
        }
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

    const HoleInfo holeInfo = calculateHoleInfo(holePixels);

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
        const float search_radius_sq = holeInfo.radius * holeInfo.radius;
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

