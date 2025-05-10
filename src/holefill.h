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

/**
 * @brief Fills holes in an image using a weighted average of boundary pixels.
 *
 * This function implements the full hole-filling algorithm that considers all boundary pixels
 * in the image when filling each hole. For each hole pixel (pixels with negative values):
 * 1. Looks at every boundary pixel in the entire image
 * 2. Calculates a weight for each boundary pixel based on its distance from the hole pixel
 * 3. Takes a weighted average of all boundary pixels' values, where:
 *    - Closer boundary pixels get higher weights
 *    - Further boundary pixels get lower weights
 *
 * This is the full version that considers all boundary pixels in the image,
 * making it more accurate but slower than the approximate version.
 *
 * @param image Pointer to the image data as a flat array of floats, linear values. Negative values indicate holes.
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @param weightFunc Function that calculates the weight between two pixels based on their coordinates.
 *                   The weight should be higher for closer pixels and lower for distant pixels.
 *
 * @note The image is modified in-place. Hole pixels (negative values) are replaced with
 *       the weighted average of surrounding valid pixels.
 *
 * @see fillApproximate for a faster but less accurate version that uses a fixed window size
 */
void fill(float* image, const int32_t width, const int32_t height, WeightFunction weightFunc);

/**
 * @brief Fills holes in an image using a fast linear-time algorithm that processes pixels from boundary inward.
 *
 * This function implements an efficient hole-filling algorithm that processes pixels in order
 * from the boundary inward. For each hole pixel (pixels with negative values):
 * 1. Processes pixels in order of their distance from the boundary
 * 2. For each pixel, takes the average of its 8-connected non-hole neighbors
 * 3. Once a pixel is filled, its value is used for filling subsequent pixels
 *
 * This version uses a queue-based approach to ensure each pixel is processed exactly once,
 * making it O(h) time complexity where h is the number of hole pixels. It uses 8-connected
 * neighborhood for better quality results.
 *
 * @param image Pointer to the image data as a flat array of floats, linear values. Negative values indicate holes.
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 *
 * @note The image is modified in-place. Hole pixels (negative values) are replaced with
 *       the average of their non-hole neighbors.
 *
 * @see fill for the full version that considers all boundary pixels
 * @see fillExactWithSearch for the KD-tree based version
 */
void fillApproximate(float* image, const int32_t width, const int32_t height);

/**
 * @brief Fills holes in an image using a KD-tree for efficient k-nearest neighbor search.
 *
 * This function implements an exact hole-filling algorithm that uses a KD-tree to efficiently
 * find the k-nearest boundary pixels for each hole pixel. The algorithm works as follows:
 * 1. Identifies all hole pixels (pixels with negative values) and boundary pixels
 * 2. Builds a KD-tree from the boundary pixels for efficient spatial queries
 * 3. For each hole pixel:
 *    - Uses the KD-tree to find its k-nearest boundary pixels, where k is nearestNeighborMax
 *    - Uses the provided weight function to calculate weights between pixels
 *    - Takes a weighted average of the found boundary pixels' values
 *
 * Time Complexity: O(n * log m) where:
 * - n is the number of hole pixels
 * - m is the number of boundary pixels
 * - The KD-tree provides O(log m) nearest neighbor queries
 * In worst case (when m ≈ n), this becomes O(n * log n)
 *
 * Space Complexity: O(m) for storing the KD-tree and boundary pixels
 *
 * This version combines the accuracy of considering the k-nearest boundary pixels with
 * the efficiency of spatial indexing. It's faster than the full version for large images
 * while maintaining good accuracy by focusing on the most relevant boundary pixels.
 *
 * @param image Pointer to the image data as a flat array of floats, linear values. Negative values indicate holes.
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @param weightFunc Function that calculates the weight between two pixels based on their coordinates.
 *                   The weight should be higher for closer pixels and lower for distant pixels.
 * @param nearestNeighborMax Maximum number of nearest boundary pixels to consider for each hole pixel.
 *                          This parameter controls the trade-off between accuracy and performance.
 *                          A larger value will consider more boundary pixels but increase computation time.
 *
 * @note The image is modified in-place. Hole pixels are replaced with the weighted average
 *       of their k-nearest boundary pixels. The algorithm uses nanoflann's KD-tree implementation
 *       for efficient nearest neighbor search.
 *
 * @see fill for the full version that considers all boundary pixels
 * @see fillApproximate for the window-based approximate version
 */
void fillExactWithSearch(float* image, int32_t width, int32_t height,
                         WeightFunction weightFunc, const size_t nearestNeighborMax);

} // namespace holefill
