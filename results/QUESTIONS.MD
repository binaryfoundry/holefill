# Hole Filling Algorithm Analysis

## Question 1: Complexity Analysis

**If there are boundary pixels and pixels inside the hole, what's the complexity of the algorithm that fills the hole, assuming that the hole and boundary were already found? Try to also express the complexity only in terms of n.**

### Answer:
- **Time Complexity**: O(n * m)
  - For each hole pixel (n), we look at every boundary pixel (m)
  - This is the most straightforward but also the most expensive approach
  - Expressed only in terms of n: O(n²) in worst case (when m ≈ n)

## Question 2: O(n) Approximation Algorithm

**Describe an algorithm that approximates the result in O(n) to a high degree of accuracy.**
*Bonus: implement the suggested algorithm in your library in addition to the algorithm described above.*

### Answer:

#### Complexity Analysis
- **Time Complexity**: O(n) where n is number of hole pixels
  - Each hole pixel is processed exactly once
  - Uses a queue to process pixels in order from boundary inward
  - The number of boundary pixels doesn't affect the complexity
  - This is the most efficient approach in terms of hole pixels

- **Space Complexity**: O(width * height)
- **Best for**: Large images where speed is important
- **Uses**: 8-connected neighborhood for better quality

#### Algorithm Phases

1. **Initialization**:
   - Creates a boolean mask to track hole pixels (O(width * height) space)
   - Marks all hole pixels in the mask
   - Initializes an empty queue for processing

2. **Boundary Detection**:
   - Scans all hole pixels
   - For each hole pixel, checks its 8-connected neighbors
   - If a neighbor is a valid (non-hole) pixel, adds the hole pixel to the processing queue
   - This identifies the boundary layer of hole pixels that can be filled first

3. **Progressive Filling**:
   - Processes pixels in the queue (which ensures we fill from boundary inward)
   - For each pixel:
     - Calculates the average of its 8-connected non-hole neighbors
     - If valid neighbors exist, fills the pixel with this average
     - Marks the pixel as filled in the mask
     - Adds any unfilled hole neighbors to the queue
   - Continues until no more pixels can be filled

#### Key Characteristics
- Each hole pixel is processed exactly once
- Filling proceeds from the boundary inward
- Uses 8-connected neighborhood for better quality results
- Maintains a mask to track filled pixels
- Queue ensures proper order of processing

#### Efficiency Factors
- Processing each pixel exactly once
- Using a queue to maintain the correct processing order
- Avoiding redundant calculations
- Using a boolean mask for O(1) lookups

## Question 3: Exact Solution Algorithm (Bonus)

**Describe and implement an algorithm that finds the exact solution in O(n log n). In this section, feel free to use any algorithmic functionality provided by external libraries as needed.**

The requirement of O(n log n) gives us a clue that a search must be involved. We will do a K-Nearest Neighbors (k-NN) spatial search.

### Answer:
- **Time Complexity**: O(n * log m)
  - For each hole pixel (n), we perform a logarithmic search (log m) of nearest neighbor boundary pixels
  - The KD-tree allows us to find relevant boundary pixels efficiently
  - Expressed only in terms of n: O(n * log n) in worst case (when m ≈ n)
