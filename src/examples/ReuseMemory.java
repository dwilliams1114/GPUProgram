package examples;

import java.util.Arrays;
import main.GPUAccess;
import main.GPUProgram;

/**
 * This example adds any number of arrays together iteratively by accumulating the result in the final array.
 * We wait until the end to copy the result off the GPU.
 * In the intermediate iterations, we just re-use the memory that's already allocated on the GPU to accumulate the result.
 *
 * Programmed by Daniel Williams in 2025.
 */

public class ReuseMemory {
	
	public static void main(String[] args) {
		
		// This reads and compiles the VectorAdd.cl file.
		GPUProgram adder = new GPUProgram("accumulateKernel", "src/examples/Accumulator.cl");
		
		// Two input arrays, and one empty output array.
		// The result will be written directly into array 'c'.
		float[] a = {0.5f, 0.5f, 0.4f, 0.6f, 0.0f, 0.7f};
		float[] b = {0.1f, 0.0f, 0.0f, 0.2f, 0.3f, 0.1f};
		float[] c = {0.0f, 0.5f, 0.3f, 0.4f, 0.1f, 0.4f};
		float[] d = {0.2f, 0.0f, 0.6f, 0.7f, 0.9f, 0.3f};
		float[] result = new float[a.length];
		
		// Create a list of references to our original arrays, so we can iterate over them
		float[][] inputs = {a, b, c, d};
		
		// All our arrays are the same size, so we can just use the length of 'a'
		adder.setGlobalWorkGroupSizes(a.length);
		
		// We only need set a reference to the accumulator once
		adder.setArgument(0, result, GPUAccess.READ_WRITE);
		
		// Add each of the inputs to the accumulator
		for (int i = 0; i < inputs.length; i++) {
			
			// Set the second argument to our next array to be added to the result
			adder.setArgument(1, inputs[i], GPUAccess.READ);
			
			// Perform the addition, but don't waste time copying the result back to the CPU yet.
			// We will copy the final result back to memory later.
			adder.executeKernelNoCopyback();
		}
		
		// Copy result back to main memory, so it will be accessible by Java
		adder.copyFromGPU();
		
		// Print result
		System.out.println(Arrays.toString(result));
	}
	
}
