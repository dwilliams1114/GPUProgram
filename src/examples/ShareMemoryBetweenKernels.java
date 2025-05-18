package examples;

import java.util.Arrays;
import main.GPUAccess;
import main.GPUMem;
import main.GPUProgram;

/**
 * This example shows how to reference the same memory from two different OpenCL kernels.
 * There are of course much more efficient ways to do this, but this example is only
 * meant to demonstrate the memory sharing capability.
 * This example requires "VectorAdd.cl" and "VectorMult.cl"
 * 
 * Programmed by Daniel Williams in 2025.
 * 
 * Steps for memory copies performed by this program:
 *	1. Allocate space for 'result' on the GPU
 *	2. Allocate space for 'a' on the GPU
 *	3. Copy 'a' to the GPU
 *	4. Allocate space for 'b' on the GPU
 *	5. Copy 'b' to the GPU
 *	6. Compute addition (no copies performed)
 *	7. Allocate space for 'c' on the GPU
 *	8. Copy 'c' to the GPU
 *	9. Compute multiplication (no copies performed)
 *	10. Copy 'result' back to the CPU
 */

public class ShareMemoryBetweenKernels {
	
	public static void main(String[] args) {
		
		// Prepare two separate OpenCL programs to add and multiply arrays.
		GPUProgram vecAddProgram = new GPUProgram("vectorAddKernel", "src/examples/VectorAdd.cl");
		GPUProgram vecMultProgram = new GPUProgram("vectorMultKernel", "src/examples/VectorMult.cl");
		
		// We will compute "result = (a + b) * c
		float[] a = {1, 2, 3, 4, 5, 6};
		float[] b = {3, 2, 1, 0, 1, 2};
		float[] c = {2, 1, 2, 1, 2, 3};
		float[] result = new float[a.length];
		
		// Zero out all the memory copy counters
		GPUProgram.resetDebugCounters();
		
		GPUMem resultGPUMem = GPUProgram.allocateMemoryOnGPU(result, GPUAccess.READ_WRITE, true);
		
		// Set arguments for vector addition
		vecAddProgram.setArgument(0, a, GPUAccess.READ);
		vecAddProgram.setArgument(1, b, GPUAccess.READ);
		vecAddProgram.setArgument(2, resultGPUMem);
		vecAddProgram.setGlobalWorkGroupSizes(a.length);
		vecAddProgram.executeKernelNoCopyback(); // Compute the result, but leave it on the GPU for now
		
		// Set arguments for vector multiplication (we are both reading and writing to the 'resultGPUMem' array)
		vecMultProgram.setArgument(0, resultGPUMem);
		vecMultProgram.setArgument(1, c, GPUAccess.READ);
		vecMultProgram.setArgument(2, resultGPUMem);
		vecMultProgram.setGlobalWorkGroupSizes(a.length);
		vecMultProgram.executeKernel(); // Compute result and copy result back to main memory
		
		// Print result
		System.out.println(Arrays.toString(result));
		
		// Print the number of memory copies performed to and from the GPU
		GPUProgram.printDebugCounters();
	}
	
}
