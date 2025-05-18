package test;

import main.GPUAccess;
import main.GPUMem;
import main.GPUProgram;
import main.GPURange;

/**
 * @author Daniel Williams
 *
 * Created on March 30, 2024
 * Last updated on March 31, 2024
 * 
 * Created to test and verify GPU memory manipulation with the GPUProgram library.
 */

public class GPUMemoryTest {
	
	static private GPUProgram vecAddProgram;
	static private GPUProgram vecMultProgram;
	static private int testsFailed = 0;
	
	public static void main(String[] args) {
		
		GPUProgram.initializeGPU();
		
		vecAddProgram = new GPUProgram("sum", "src/gpuTesting/MemoryTest.cl");
		vecMultProgram = new GPUProgram("mult", "src/gpuTesting/MemoryTest.cl");
		
		final int N = 100000000;

		float[] accumulator = new float[N];
		float[] add1 = new float[N];
		float[] mul1 = new float[N];
		
		for (int i = 0; i < N; i++) {
			add1[i] = (float)(Math.random() * 2 - 1);
			mul1[i] = (float)(Math.random() * 2 - 1);
		}
		
		vecAddProgram.setGlobalWorkGroupSizes(N);
		vecMultProgram.setGlobalWorkGroupSizes(N);
		
		//* Test using pointers to memory on the GPU instead of copying back to the CPU on each cycle.
		
		print("CPU:");
		computeResultCPU(accumulator, add1, mul1);
		
		double sum1 = 0;
		for (int i = 0; i < N; i++) {
			sum1 += accumulator[i];
		}
		print("CPU Sum: " + sum1);
		
		accumulator = new float[N];
		print("");

		print("GPU Naive:");
		computeResultGPUWithCopyback(accumulator, add1, mul1);
		
		double sum2 = 0;
		for (int i = 0; i < N; i++) {
			sum2 += accumulator[i];
		}
		print("GPU Sum: " + sum2);
		if (sum1 != sum2) {
			testsFailed++;
		}
		
		accumulator = new float[N];
		print("");

		print("GPU With GPUMem pointers:");
		computeResultGPUWithMemPointers(accumulator, add1, mul1);
		
		double sum3 = 0;
		for (int i = 0; i < N; i++) {
			sum3 += accumulator[i];
		}
		print("GPU Sum: " + sum3);
		print("----------------------\n");
		if (sum1 != sum3) {
			testsFailed++;
		}
		//*/
		
		
		//* Test sub-range array processing
		testSubsetProcessing(accumulator, add1, 0, N);
		testSubsetProcessing(accumulator, add1, 0, N / 10);
		testSubsetProcessing(accumulator, add1, N / 5, N / 4);
		testSubsetProcessing(accumulator, add1, 0, N);
		print("----------------------\n");
		//*/
		
		
		//* Test sub-array range processing with GPUMem pointers
		int startIndex = 10;
		int endIndex = N/4;
		
		float[] accumulator2 = new float[N];
		for (int i = 0; i < N; i++) {
			accumulator2[i] = accumulator[i];
		}
		
		GPUMem accumulatorGpuMem = vecAddProgram.setArgument(0, accumulator2, new GPURange(startIndex, endIndex), GPUAccess.READ_WRITE);
		
		testSubsetWithMemPointers(accumulator, accumulator2, add1, mul1, accumulatorGpuMem, startIndex, endIndex);
		testSubsetWithMemPointers(accumulator, accumulator2, add1, mul1, accumulatorGpuMem, startIndex, endIndex);
		
		long start = System.currentTimeMillis();
		vecMultProgram.copyFromGPU();
		long gpuCopyBackTime = System.currentTimeMillis() - start;
		print("GPU Copy Back Time: " + gpuCopyBackTime + " ms");
		
		double sum4 = 0;
		for (int i = 0; i < N; i++) {
			sum4 += accumulator[i];
		}
		print("CPU Sum: " + sum4);
		
		double sum5 = 0;
		for (int i = 0; i < N; i++) {
			sum5 += accumulator2[i];
		}
		print("GPU Sum: " + sum5);
		if (sum5 != sum4) {
			testsFailed++;
		}
		print("----------------------");
		
		//*/
		
		// Print final result
		if (testsFailed > 0) {
			System.err.println("Failed " + testsFailed + " test" + (testsFailed > 1 ? "s" : "") + "!");
		} else {
			print("PASS");
		}
	}
	
	static void testSubsetWithMemPointers(float[] accumulator1, float[] accumulator2,
				float[] add1, float[] mul1, GPUMem accumulatorMem, int startIndex, int endIndex) {
		
		for (int i = startIndex; i < endIndex; i++) {
			accumulator1[i] += add1[i];
			accumulator1[i] *= mul1[i];
		}
		
		vecAddProgram.setGlobalWorkGroupSizes(endIndex - startIndex);
		vecMultProgram.setGlobalWorkGroupSizes(endIndex - startIndex);
		
		long start = System.currentTimeMillis();
		vecAddProgram.setArgument(0, accumulatorMem);
		vecAddProgram.setArgument(1, add1, new GPURange(startIndex, endIndex), GPUAccess.READ);
		print("GPU Copy To Time:   " + (System.currentTimeMillis() - start) + " ms");
		
		start = System.currentTimeMillis();
		vecAddProgram.executeKernelNoCopyback();
		print("GPU Execution Time: " + (System.currentTimeMillis() - start) + " ms");
		
		start = System.currentTimeMillis();
		vecMultProgram.setArgument(0, accumulatorMem);
		vecMultProgram.setArgument(1, mul1, new GPURange(startIndex, endIndex), GPUAccess.READ);
		print("GPU Copy To Time:   " + (System.currentTimeMillis() - start) + " ms");
		
		start = System.currentTimeMillis();
		vecMultProgram.executeKernelNoCopyback();
		print("GPU Execution Time: " + (System.currentTimeMillis() - start) + " ms");
		
		print("");
	}
	
	static void testSubsetProcessing(float[] accumulator, float[] add1, int startIndex, int endIndex) {
		
		for (int i = 0; i < accumulator.length; i += 9) {
			accumulator[i] = (float)(Math.random() - 0.5);
		}
		
		double sum1 = 0;
		for (int i = startIndex; i < endIndex; i++) {
			sum1 += accumulator[i] + add1[i];
		}
		print("CPU Sum: " + sum1);

		vecAddProgram.setGlobalWorkGroupSizes(endIndex - startIndex);
		
		long start = System.currentTimeMillis();
		vecAddProgram.setArgument(0, accumulator, new GPURange(startIndex, endIndex), GPUAccess.READ_WRITE);
		vecAddProgram.setArgument(1, add1, new GPURange(startIndex, endIndex), GPUAccess.READ);
		long copyToGPUTime = System.currentTimeMillis() - start;
		print("GPU Copy To Time:   " + copyToGPUTime + " ms");
		
		start = System.currentTimeMillis();
		vecAddProgram.executeKernelNoCopyback();
		long gpuExecutionTime = System.currentTimeMillis() - start;
		print("GPU Execution Time: " + gpuExecutionTime + " ms");
		
		start = System.currentTimeMillis();
		vecAddProgram.copyFromGPU();
		long gpuCopyBackTime = System.currentTimeMillis() - start;
		print("GPU Copy Back Time: " + gpuCopyBackTime + " ms");
		
		double sum2 = 0;
		for (int i = startIndex; i < endIndex; i++) {
			sum2 += accumulator[i];
		}
		print("GPU Sum: " + sum2);
		print("");
		if (sum1 != sum2) {
			testsFailed++;
		}
	}
	
	static void computeResultGPUWithMemPointers(float[] accumulator, float[] add1, float[] mul1) {
		vecAddProgram.releaseGPUMemory();
		vecMultProgram.releaseGPUMemory();
		
		long copyToGPUTime = 0;
		long executionTime = 0;
		
		long start = System.currentTimeMillis();
		GPUMem accumulatorGpuMem = vecAddProgram.setArgument(0, accumulator, GPUAccess.READ_WRITE);
		vecMultProgram.setArgument(0, accumulatorGpuMem);
		vecAddProgram.setArgument(1, add1, GPUAccess.READ);
		vecMultProgram.setArgument(1, mul1, GPUAccess.READ);
		copyToGPUTime += System.currentTimeMillis() - start;
		
		for (int k = 0; k < 10; k++) {
			start = System.currentTimeMillis();
			vecAddProgram.executeKernelNoCopyback();
			executionTime += System.currentTimeMillis() - start;
			
			start = System.currentTimeMillis();
			vecMultProgram.executeKernelNoCopyback();
			executionTime += System.currentTimeMillis() - start;
		}

		start = System.currentTimeMillis();
		vecMultProgram.copyFromGPU();
		copyToGPUTime += System.currentTimeMillis() - start;
		
		print("GPU Execute time: " + executionTime + " ms");
		print("Copy to GPU time: " + copyToGPUTime + " ms");
	}
	
	static void computeResultGPUWithCopyback(float[] accumulator, float[] add1, float[] mul1) {
		
		long copyToGPUTime = 0;
		long executionTime = 0;
		
		long start = System.currentTimeMillis();
		vecAddProgram.setArgument(1, add1, GPUAccess.READ);
		vecMultProgram.setArgument(1, mul1, GPUAccess.READ);
		copyToGPUTime += System.currentTimeMillis() - start;
		
		for (int k = 0; k < 10; k++) {
			start = System.currentTimeMillis();
			vecAddProgram.setArgument(0, accumulator, GPUAccess.READ_WRITE);
			copyToGPUTime += System.currentTimeMillis() - start;
			
			start = System.currentTimeMillis();
			vecAddProgram.executeKernel();
			executionTime += System.currentTimeMillis() - start;
			
			start = System.currentTimeMillis();
			vecMultProgram.setArgument(0, accumulator, GPUAccess.READ_WRITE);
			copyToGPUTime += System.currentTimeMillis() - start;

			start = System.currentTimeMillis();
			vecMultProgram.executeKernel();
			executionTime += System.currentTimeMillis() - start;
		}
		
		print("GPU Execute time: " + executionTime + " ms");
		print("Copy to GPU time: " + copyToGPUTime + " ms");
	}
	
	static void computeResultCPU(float[] accumulator, float[] add1, float[] mul1) {
		long cpuStartTime = System.currentTimeMillis();
		for (int k = 0; k < 10; k++) {
			sum(accumulator, add1);
			mult(accumulator, mul1);
		}
		long cpuEndTime = System.currentTimeMillis();
		print("CPU Total Time: " + (cpuEndTime - cpuStartTime) + " ms");
	}
	
	static void sum(float[] arr1, float[] arr2) {
		for (int i = 0; i < arr1.length; i++) {
			arr1[i] += arr2[i];
		}
	}
	
	static void mult(float[] arr1, float[] arr2) {
		for (int i = 0; i < arr1.length; i++) {
			arr1[i] *= arr2[i];
		}
	}
	
	static void print(Object o) {
		System.out.println(o);
	}
}
