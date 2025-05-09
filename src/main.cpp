#include "holefill.h"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>

#include "stb_image.h"
#include "stb_image_write.h"

float srgbToLinear(const float c) {
    if (c <= 0.04045f)
        return c / 12.92f;
    else
        return powf((c + 0.055f) / 1.055f, 2.4f);
}

float linearToSrgb(const float c) {
    if (c <= 0.0031308f)
        return c * 12.92f;
    else
        return 1.055f * powf(c, 1.0f/2.4f) - 0.055f;
}

// Convert sRGB to grayscale float in [0,1]
float rgbToGrayscaleLinear(const unsigned char r, const unsigned char g, const unsigned char b) {
    const float rf = srgbToLinear(r / 255.0f);
    const float gf = srgbToLinear(g / 255.0f);
    const float bf = srgbToLinear(b / 255.0f);
    return 0.299f * rf + 0.587f * gf + 0.114f * bf;
}

static float defaultWeightFunction(const holefill::Coord& u, const holefill::Coord& v) {
    const float epsilon = 0.01f;
    const float zeta = 3.0f;
    const float dx = static_cast<float>(u.x - v.x);
    const float dy = static_cast<float>(u.y - v.y);
    const float distanceSquared = dx * dx + dy * dy;
    return 1.0f / powf(distanceSquared + epsilon, zeta);
}

// Weight function that takes window size into account
static holefill::WeightFunction createWindowedWeightFunction(int windowSize) {
    return [windowSize](const holefill::Coord& u, const holefill::Coord& v) {
        const float epsilon = 0.01f;
        const float zeta = 3.0f;
        const float dx = static_cast<float>(u.x - v.x);
        const float dy = static_cast<float>(u.y - v.y);
        // Scale the distance by the window size
        const float scaledDistanceSquared = (dx * dx + dy * dy) / (windowSize * windowSize);
        return 1.0f / powf(scaledDistanceSquared + epsilon, zeta);
    };
}

int main(const int argc, const char** const argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image.png> <mask.png> <output.png>\n";
        return 1;
    }

    const char* const imagePath = argv[1];
    const char* const maskPath = argv[2];
    const char* const outputPath = argv[3];

    int width, height, channels;
    const unsigned char* const imageData = stbi_load(imagePath, &width, &height, &channels, 3);  // Force 3 channels
    const unsigned char* const maskData = stbi_load(maskPath, &width, &height, nullptr, 1);      // Force 1 channel

    if (!imageData || !maskData) {
        if (imageData) stbi_image_free(const_cast<unsigned char*>(imageData));
        if (maskData) stbi_image_free(const_cast<unsigned char*>(maskData));
        std::cerr << "Failed to load image or mask.\n";
        return 1;
    }

    // Grayscale float image with hole
    std::vector<float> grayscaleImage(width * height);

    for (int i = 0; i < width * height; ++i) {
        const int idx = i * 3;

        // Convert base image to grayscale
        const float grayscale = rgbToGrayscaleLinear(imageData[idx], imageData[idx + 1], imageData[idx + 2]);

        // Convert mask pixel to grayscale to determine if it's a hole
        const float maskGray = rgbToGrayscaleLinear(maskData[i], maskData[i], maskData[i]);

        // Carve out hole if mask grayscale < 0.5
        grayscaleImage[i] = (maskGray < 0.5f) ? -1.0f : grayscale;
    }

    // Fill the hole using the holefill algorithm
    const int windowSize = 20;  // You can adjust this value
    holefill::fillApproximate(grayscaleImage.data(), width, height, createWindowedWeightFunction(windowSize), windowSize);

    // Save output: convert float image [0,1] to 8-bit grayscale for writing
    std::vector<unsigned char> outputImage(width * height);
    for (int i = 0; i < width * height; ++i) {
        const float linearValue = (grayscaleImage[i] < 0.0f) ? 0.0f : grayscaleImage[i];
        const float srgbValue = linearToSrgb(linearValue);
        const float clamped = std::min(1.0f, std::max(0.0f, srgbValue));
        outputImage[i] = static_cast<unsigned char>(clamped * 255.0f);
    }

    if (!stbi_write_png(outputPath, width, height, 1, outputImage.data(), static_cast<int>(width))) {
        std::cerr << "Failed to write output image.\n";
        return 1;
    }

    std::cout << "Output written to: " << outputPath << std::endl;

    stbi_image_free(const_cast<unsigned char*>(imageData));
    stbi_image_free(const_cast<unsigned char*>(maskData));
    return 0;
}
