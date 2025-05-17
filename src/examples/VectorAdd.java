package examples;

import java.util.Arrays;
import main.GPUAccess;
import main.GPUProgram;

/*
 * This example adds to arrays, a + b = c, in parallel on the GPU.
 * This example requires "VectorAdd.cl".
 */

public class VectorAdd {
	
	public static void main(String[] args) {
		
		// This reads and compiles the VectorAdd.cl file.
		GPUProgram vecAddProgram = new GPUProgram("vectorAddKernel", "src/examples/VectorAdd.cl");
		
		// Two input arrays, and one empty output array.
		// The result will be written directly into array 'c'.
		float[] a = {0.6f, 0.5f, 0.4f, 0.0f, 0.0f, 0.0f};
		float[] b = {0.0f, 0.0f, 0.0f, 0.2f, 0.3f, 0.4f};
		float[] c = new float[a.length];
		
		// Set the arguments to the kernel function, vectorAddKernel in VectorAdd.cl.
		vecAddProgram.setArgument(0, a, GPUAccess.READ);
		vecAddProgram.setArgument(1, b, GPUAccess.READ);
		vecAddProgram.setArgument(2, c, GPUAccess.WRITE);
		
		// Set the bounds of our work group to the length of our array,
		// or equivalently, the number of parallel threads to use.
		// This limits the value returned by get_global_id(0) in OpenCL to the length of our array.
		// 0 <= get_global_id(0) < a.length
		vecAddProgram.setGlobalWorkGroupSizes(a.length);
		
		// This performs four steps:
		// 1. Copy memory to the GPU
		// 2. Perform the computation
		// 3. Wait for all threads completed
		// 4. Copy result back to main memory (accessible by Java)
		vecAddProgram.executeKernel();
		
		// Print result
		System.out.println(Arrays.toString(c));
	}
	
}
