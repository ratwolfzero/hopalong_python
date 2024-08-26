
# Motivation

- Goal: Efficiently display the Hopalong attractor as an image in an interactive plot window.
- Requirements: Support a very high number of iterations with optimal processing speed and accurate image representation.
- Approach: Implement a straightforward, efficient program with minimal complexity.
  
## The Two-Pass Approach: A Chosen Solution

The two-pass approach involves sequential processing using simple loops, ensuring optimal memory efficiency and stable performance.

### First Pass: Calculating Trajectory Extents

- Purpose: Calculate the spatial extents (minimum and maximum x and y coordinates) of the trajectory.
- Outcome: Determines the extents of the trajectory, which is crucial for accurately  mapping points in the second pass.

### Second Pass: Trajectory Point Calculation and Mapping

- Purpose:
- Compute the sequence of trajectory points.
- Map these points directly to image pixel coordinates.
- Update the corresponding pixels in the image array based on the hit rate of each coordinate.
- Outcome: The image array is populated with an accurate visual representation of the trajectory.

Main Benefit

By separating the extent calculation (first pass) from trajectory point mapping (second pass), this approach allows for efficient sequential processing. Knowing the trajectory extents in advance enables direct and efficient mapping of points to image pixels, optimizing memory usage and maintaining consistent performance.

## Comparison with Single-Pass and Trajectory Points Caching Approaches

### Single-Pass with Caching

- Process: Both calculation and mapping occur within a single loop, with all trajectory points stored in an array to avoid recalculation.
- Memory Usage: Storing all points requires significant memory, potentially leading to overflow or performance issues as iterations increase.
- Performance: Although caching saves recalculation time, the high memory overhead can degrade performance, especially in memory-constrained systems.

### Single-Pass with Direct Mapping

- Process: Trajectory points are calculated and mapped directly to image pixels within a single loop, without caching.
- Efficiency: This method is inefficient and computationally expensive due to the repeated remapping of already processed pixels. This redundancy makes it impractical for large iterations, as it significantly slows down the process and leads to unnecessary computational overhead.

### Single-Pass with Chunked Caching

- Approach: The trajectory is divided into chunks, with each chunk of points computed and temporarily cached before being mapped to image coordinates.
- Mapping Issues: Since the global extents of the trajectory are not known in advance, each chunk must be mapped to pixel coordinates based on incomplete information. This leads to mapping errors and the need to remap pixels as new points are processed.
- Common Issue with Single-Pass Methods: Just like the non-chunked single-pass approach, this method suffers from the inefficiency that already processed pixels need to be remapped. The chunked approach may help manage memory, but it does not resolve the fundamental issue of inaccurate and inefficient pixel mapping.
- Performance and Complexity: While chunking can reduce memory load, it does not improve the efficiency of the mapping process and adds complexity in managing chunks. The remapping issues remain unchanged, making this approach no more efficient than a straightforward single-pass method.

### Two-Pass Chunked Caching

In this variant, the first pass calculates the global extents of the trajectory, and the second pass processes the trajectory in chunks, mapping each chunk accurately to image coordinates based on the global extents. While memory-efficient, this method involves complexity in managing chunks and still faces the inefficiency of repeated remapping within each chunk. Additionally, the overhead of recalculating extents adds to the overall computational cost, making it slightly slower than the standard two-pass approach.

## Advantages of the Two-Pass Approach

- Memory Efficiency: The two-pass approach reduces memory requirements by recalculating trajectory points, eliminating the need for large-scale caching.
- JIT Compatibility: The simple, sequential structure is well-suited for Just-In-Time (JIT) compilation, enhancing execution speed.
- Scalability: As the number of iterations grows, the two-pass approach’s efficiency in memory usage and processing speed becomes more advantageous.

## Disadvantage

- Recalculation: Trajectory points are recalculated in both passes, but this trade-off is preferable to the high memory demands and complexity of alternative methods.

Conclusion

The two-pass approach was chosen for its balance of performance, memory efficiency, and simplicity. Despite the need to recalculate trajectory points, it avoids the pitfalls of high memory consumption, complex implementation, and inefficient mapping found in single-pass approaches, making it the most robust and effective solution for calculating the Hopalong attractor with a high number of iterations.
