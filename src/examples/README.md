This folder contains simple example applications to demonstrate the usage of GPUProgram.

## Descriptions

**VectorAdd.java + VectorAdd.cl**<br>
Performs c[i] = a[i] + b[i] on the GPU with each index in parallel.

**ReuseMemory.java + Accumulator.cl**<br>
Computes the sum of any number of arrays, where each index is processed in parallel.
Memory on the GPU is reused between iterations to accumulate the result, and the result is copied back to the CPU only at the end.