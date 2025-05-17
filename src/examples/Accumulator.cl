/**
 * This kernel adds the value of 'a' to 'accumulator'
 * accumulator[i] += a[i]
 *
 * Programmed by Daniel Williams in 2025
 */

kernel void accumulateKernel(global float* accumulator, global const float* a) {
	int i = get_global_id(0);
	accumulator[i] += a[i];
}