# Hole Filling Library

A C++ library for filling holes in images using various algorithms with different trade-offs between speed and accuracy.

For answers to the questions in this exercise, see [results/README.md](results/README.md).

## Overview

This library provides three different algorithms for filling holes in images:

1. **Full Fill** (`fill`): Considers all boundary pixels in the image, providing the most accurate results but slower performance.
2. **Approximate Fill** (`fillApproximate`): Uses a fast linear-time algorithm that processes pixels from boundary inward, providing a good balance of speed and quality.
3. **Exact Fill with Search** (`fillExactWithSearch`): Uses a KD-tree for efficient nearest neighbor search, combining accuracy with good performance for large images.

## Features

- Multiple algorithms with different speed/accuracy trade-offs
- Customizable weight functions for distance-based weighting
- Efficient spatial indexing using KD-trees
- Linear time complexity for the approximate version
- In-place image modification

## Requirements

- C++11 or later
- [nanoflann](https://github.com/jlblancoc/nanoflann) for KD-tree implementation

## Further Work

- Much more testing required

## Usage

```cpp
#include "holefill.h"

// Example: Using the approximate fill algorithm
float* image = /* your image data */;
int32_t width = 1024;
int32_t height = 768;

// Fill holes using the fast approximate algorithm
holefill::fillApproximate(image, width, height);

// Example: Using the full fill algorithm with custom weight function
auto weightFunc = [](const holefill::Coord& a, const holefill::Coord& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float distSq = dx * dx + dy * dy;
    return 1.0f / (1.0f + distSq);  // Inverse square distance weight
};
holefill::fill(image, width, height, weightFunc);

// Example: Using the exact fill with KD-tree search
holefill::fillExactWithSearch(image, width, height, weightFunc);
```

## Algorithm Details

### Full Fill
- Time Complexity: O(n * m) where n is number of hole pixels and m is number of boundary pixels
- Space Complexity: O(n + m)
- Best for: Small images or when accuracy is critical

### Approximate Fill
- Time Complexity: O(n) where n is number of hole pixels
- Space Complexity: O(width * height)
- Best for: Large images where speed is important

### Exact Fill with Search
- Time Complexity: O(n * log m) where n is number of hole pixels and m is number of boundary pixels
- Space Complexity: O(n + m)
- Best for: Large images where accuracy is important
- Uses KD-tree for efficient spatial queries of KNN.

## Image Format

The library expects images as flat arrays of floats where:
- Valid pixels have non-negative values
- Hole pixels are indicated by negative values
- The image is stored in row-major order (y * width + x)
