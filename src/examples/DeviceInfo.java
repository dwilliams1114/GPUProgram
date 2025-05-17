package examples;

import main.GPUProgram;

/**
 * This example prints useful information about the active GPU.
 * 
 * Programmed by Daniel Williams in 2025
 */

public class DeviceInfo {
	
	public static void main(String[] args) {
		
		System.out.println("Total global memory (bytes): " + GPUProgram.getGlobalMemory());
		System.out.println("Max memory object size (bytes): " + GPUProgram.getMaxMemAllocSize());
		System.out.println("Max work-items per group: " + GPUProgram.getMaxLocalWorkGroupSize());
		
		System.out.println();
		
		// Print general info:
		GPUProgram.printDeviceStatistics();
	}
	
}
