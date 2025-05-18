/**
 * This kernel multiplies the corresponding components of the two given arrays:
 * c[i] = a[i] * b[i]
 *
 * Programmed by Daniel Williams in 2025.
 */

kernel void vectorMultKernel(global const float* a, global const float* b, global float* c) {
	int i = get_global_id(0);
	c[i] = a[i] * b[i];
}