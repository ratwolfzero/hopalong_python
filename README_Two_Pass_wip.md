
# Motivation

- Goal: Efficiently display the Hopalong attractor as an image in an interactive plot window.
- Requirements: Support a very high number of iterations with optimal processing speed and accurate image representation.
- Approach: Implement a straightforward, efficient program with minimal complexity.
  
## The Two-Pass Approach: A Chosen Solution

The two-pass approach involves sequential processing using simple loops, ensuring optimal memory efficiency and stable performance.

### First Pass

- Purpose: Calculate the trajectory's spatial boundaries (minimum and maximum x and y coordinates).
- Outcome: Determines the extent of the trajectory, which is crucial for accurately mapping points in the second pass.
  
### Second Pass

- Purpose:
- Compute the sequence of trajectory points.
- Map these points directly to image pixel coordinates.
- Update the corresponding pixels in the image array based on the hit rate of each coordinate.
- Outcome: The image array is populated with an accurate visual representation of the trajectory.
  
### Main Benefit

By separating the extension calculation (first pass) from trajectory point mapping (second pass), the approach allows efficient sequential processing. Knowing the trajectory boundaries in advance enables direct and efficient mapping of points to image pixels, optimizing memory usage and maintaining consistent performance.
Comparison with Single-Pass and Trajectory Points Caching Approaches

## Single-Pass with Caching

- Process: Both calculation and mapping occur within a single loop, with all trajectory points stored in an array to avoid recalculation.
- Memory Usage: Storing all points requires significant memory, which can lead to overflow or performance issues as iterations increase.
- Performance: Although caching saves recalculation time, the high memory overhead can degrade performance, especially in memory-constrained systems.
  
## Single-Pass with Direct Mapping

- Process: Trajectory points are calculated and mapped directly to image pixels within a single loop, without caching.
- Efficiency: This method is inefficient and computationally expensive, as it involves repeated remapping of already processed pixels, making it impractical for large iterations.
  
## Single-Pass with Chunked Caching

Approach: The idea here is to divide the computation into chunks, where each chunk of trajectory points is processed and cached temporarily, then mapped to image coordinates before moving on to the next chunk. This could, in theory, reduce the memory load compared to caching all trajectory points at once.

- Issues with Mapping: Just like in a single-pass direct mapping approach, each chunk would still need to be mapped to image coordinates as soon as it's calculated. This means that mapping inaccuracies or inefficiencies that occur in a direct single-pass approach would still be present. The main issue is that without knowing the full extents of the trajectory (min and max values), accurate mapping cannot be ensured.
- Recalculation of Extents: To address the mapping issue, you would need to recalculate the extents for each chunk or, more realistically, calculate the full trajectory extents first, effectively turning this into a two-pass chunked caching approach.
- Two-Pass Chunked Caching: In this variant, the first pass would calculate the global extents of the trajectory, and the second pass would process the trajectory in chunks, mapping each chunk accurately to image coordinates based on the global extents. This method is memory efficient since it doesn't require storing all trajectory points at once, but it involves the complexity of managing chunks and the overhead of recalculating extents, making it slightly slower than the standard two-pass approach.
- Performance and Complexity: While chunking can help manage memory, the additional complexity in implementation (handling multiple chunks and recalculating extents) and the overhead from function calls at the end of each chunk generally make it less efficient overall. Tt's typically a bit slower than the straightforward two-pass approach.

## Advantages of the Two-Pass Approach

- Memory Efficiency: The two-pass approach eliminates the need for large-scale caching by recalculating trajectory points, thus reducing memory requirements.
- JIT Compatibility: The simple, sequential structure is well-suited for Just-In-Time (JIT) compilation, enhancing execution speed.
- Scalability: As the number of iterations grows, the two-pass approach's efficiency in memory usage and processing speed makes it more advantageous.
Disadvantage
- Recalculation: Trajectory points are recalculated in both passes, but this trade-off is preferable to the high memory demands and complexity of alternative methods.
  
## Conclusion

The two-pass approach was chosen for its balance of performance, memory efficiency, and simplicity. Despite the need to recalculate trajectory points, it avoids the pitfalls of high memory consumption, complex implementation, and inefficient mapping found in single-pass approaches, making it the most robust and effective solution for calculating the Hopalong attractor with a high number of iterations.
