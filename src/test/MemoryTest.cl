
// Sum two arrays in parallel
kernel void sum(global float *arr1, global float *arr2) {
    int i = get_global_id(0);
	arr1[i] += arr2[i];
}

// Multiply two arrays in parallel
kernel void mult(global float *arr1, global float *arr2) {
    int i = get_global_id(0);
	arr1[i] *= arr2[i];
}