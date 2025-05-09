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
 * @brief Fills holes in an image using a weighted average of boundary pixels within a fixed window.
 *
 * This function implements an approximate hole-filling algorithm that only considers boundary pixels
 * within a fixed-size window around each hole. For each hole pixel (pixels with negative values):
 * 1. Looks at boundary pixels within a window of specified size around the hole pixel
 * 2. Calculates a weight for each boundary pixel based on its distance from the hole pixel
 * 3. Takes a weighted average of the boundary pixels' values, where:
 *    - Closer boundary pixels get higher weights
 *    - Further boundary pixels get lower weights
 *
 * This is the approximate version that only considers nearby boundary pixels,
 * making it faster but potentially less accurate than the full version.
 *
 * @param image Pointer to the image data as a flat array of floats, linear values. Negative values indicate holes.
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @param weightFunc Function that calculates the weight between two pixels based on their coordinates.
 *                   The weight should be higher for closer pixels and lower for distant pixels.
 * @param windowSize Size of the square window to consider around each hole pixel.
 *                   Must be odd and positive.
 *
 * @note The image is modified in-place. Hole pixels (negative values) are replaced with
 *       the weighted average of surrounding valid pixels within the window.
 *
 * @see fill for the full version that considers all boundary pixels in the image
 */
void fillApproximate(float* image, const int32_t width, const int32_t height, WeightFunction weightFunc, int windowSize);

/**
 * @brief Fills holes in an image using a KD-tree for efficient nearest neighbor search.
 *
 * This function implements an exact hole-filling algorithm that uses a KD-tree to efficiently
 * find boundary pixels within a calculated radius of each hole pixel. For each hole pixel:
 * 1. Uses KD-tree to find all boundary pixels within the calculated radius
 * 2. Uses the provided weight function to calculate weights between pixels
 * 3. Takes a weighted average of the found boundary pixels' values
 *
 * This version combines the accuracy of considering all relevant boundary pixels with
 * the efficiency of spatial indexing, making it faster than the full version for large images
 * while maintaining accuracy. The search radius is automatically calculated based on the
 * size of the hole to ensure all relevant boundary pixels are considered.
 *
 * @param image Pointer to the image data as a flat array of floats. Negative values indicate holes.
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @param weightFunc Function that calculates the weight between two pixels based on their coordinates.
 *                   The weight should be higher for closer pixels and lower for distant pixels.
 *
 * @note The image is modified in-place. Hole pixels are replaced with the weighted average
 *       of boundary pixels found within the calculated radius.
 *
 * @see fill for the full version that considers all boundary pixels
 * @see fillApproximate for the window-based approximate version
 */
void fillExactWithSearch(float* image, int32_t width, int32_t height,
                         WeightFunction weightFunc);

} // namespace holefill
