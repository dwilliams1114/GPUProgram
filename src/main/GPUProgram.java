package main;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

/**
 * @author Daniel Williams
 * 
 * Created on February 27, 2019
 * Last update on May 17, 2025
 *
 * This class provides access to OpenCL based general purpose parallel computing using the GPU.
 * 
 * This uses the JOCL (not JogAmp) implementation of Java OpenCL bindings.
 */

enum ArrayType {
	BYTE, INT, FLOAT, LONG, DOUBLE, BUFFERED_IMAGE_INT, BUFFERED_IMAGE_BYTE;
	
	public int getSize() {
		if (this == BYTE) {
			return Sizeof.cl_uchar;
		} else if (this == INT) {
			return Sizeof.cl_int;
		} else if (this == FLOAT) {
			return Sizeof.cl_float;
		} else if (this == LONG) {
			return Sizeof.cl_long;
		} else if (this == DOUBLE) {
			return Sizeof.cl_double;
		} else if (this == BUFFERED_IMAGE_INT) {
			return Sizeof.cl_int;
		} else if (this == BUFFERED_IMAGE_BYTE) {
			return Sizeof.cl_uchar;
		}
		new Exception("\nUnknown ArrayType: " + this).printStackTrace();
		System.exit(1);
		return 0;
	}
}

public class GPUProgram {
	
	private static cl_command_queue commandQueue;
	private static cl_context context;
	private static cl_device_id device;
	private cl_program program;
	private cl_kernel kernel;
	
	// For debugging purposes
	public static int copyToGPUCounter = 0;	// How many times we copied from CPU to GPU
	public static int copyToCPUCounter = 0;	// How many times we copied from GPU to CPU
	public static int allocCounter = 0;		// How many times we allocated memory on the GPU
	public static int copyCounter = 0;		// How many times we copied memory between places on the GPU
	
	// Whether the GPU has already been initialized
	private static boolean initialized = false;
	
	// Work size for each dimension
	private long[] globalWorkSize = null;
	
	// Local work size for each dimension
	private long[] localWorkSize = null;
	
	private int maxArrayArgIndex = 0;
	private int[] arrayArgumentNum; 
	private GPUMem[] arrayGPUPointers;
	
	/** Step 1: Call this first to initialize the GPU.
	 * Calling this multiple times is okay.
	 */
	@SuppressWarnings("deprecation")
	public synchronized static void initializeGPU() {
		
		// Don't initialize twice
		if (initialized) {
			return;
		}
		initialized = true;
		
		//final long deviceType = CL.CL_DEVICE_TYPE_DEFAULT;
		//final long deviceType = CL.CL_DEVICE_TYPE_CPU;
		final long deviceType = CL.CL_DEVICE_TYPE_GPU | CL.CL_DEVICE_TYPE_ACCELERATOR;
		final int deviceIndex = 0;
		
		// Enable exceptions and subsequently omit explicit error checks
		CL.setExceptionsEnabled(true);
		
		// Obtain the number of platforms
		int numPlatformsArray[] = new int[1];
		CL.clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];
		
		// Obtain a platform ID
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		CL.clGetPlatformIDs(platforms.length, platforms, null);
		
		final int platformIndex = 0;
		cl_platform_id platform = platforms[platformIndex];
		
		// Initialize the context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);
		
		// Obtain the number of devices for the platform
		int numDevicesArray[] = new int[1];
		CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];
		
		// Obtain a device ID 
		cl_device_id devices[] = new cl_device_id[numDevices];
		CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		device = devices[deviceIndex];
		
		// Create a context for the selected device
		context = CL.clCreateContext(contextProperties, 1, new cl_device_id[] { device },
				null, null, null);
		
		//print(getDeviceString(device, CL.CL_DEVICE_NAME));
		//print(getDeviceString(device, CL.CL_DEVICE_VENDOR));
		//print(getDeviceString(device, CL.CL_DRIVER_VERSION));
		//print(getDeviceInt(device, CL.CL_DEVICE_MAX_COMPUTE_UNITS));
		//print(getDeviceInt(device, CL.CL_DEVICE_MAX_CLOCK_FREQUENCY));
		
		// Create a command-queue for the selected device
		try {
			commandQueue = CL.clCreateCommandQueueWithProperties(context, device, null, null);
		} catch (Exception e) { // This is for older systems (OpenCL 1.2)
			commandQueue = CL.clCreateCommandQueue(context, device, 0, null);
		}
	}
	
	/** Step 2a: (Overload) This is called to create the kernel from the shader that will be repeatedly executed.
	 * @param kernelName Name of the function to execute in the shader program source.
	 * @param filePath Path to the file to compile.
	 */
	public GPUProgram(String kernelName, String filePath) {
		this(kernelName, filePath, null);
	}
	
	/** Step 2b: This is called to create the kernel from the shader that will be repeatedly executed.
	 * @param kernelName Name of the function to execute in the shader program source.
	 * @param filePath Path to the file to compile.
	 * @param includePath Path to a directory containing other files #include'd in the source.
	 */
	public GPUProgram(String kernelName, String filePath, String includePath) {
		initializeGPU();
		try {
			// Load the lines of code for the OpenCL kernel
			final BufferedReader br = new BufferedReader(
					new InputStreamReader(new FileInputStream(filePath)));
			String sourceCode = "";
			String line = null;
			while (true) {
				line = br.readLine();
				if (line == null) {
					break;
				}
				sourceCode += line + "\n";
			}
			br.close();
			
			// Create the program
			program = CL.clCreateProgramWithSource(context, 1, new String[] {sourceCode}, null, null);
			
			String opts = "-Werror -cl-mad-enable -cl-fast-relaxed-math -cl-unsafe-math-optimizations";
			
			// Include any other files if needed
			if (includePath != null && !includePath.trim().isEmpty()) {
				opts += " -I " + includePath;
			}
			
			// Build the program
			CL.clBuildProgram(program, 0, null, opts, null, null);
			
			/* Possible optimization parameters:
			-cl-strict-aliasing
			-cl-mad-enable
			-cl-no-signed-zeros
			
			-cl-finite-math-only
			-cl-fast-relaxed-math
			-w
			-Werror
			*/
			
			// Create the kernel
			kernel = CL.clCreateKernel(program, kernelName, null);
		} catch (Exception e) {
			System.err.println("Building: " + filePath);
			e.printStackTrace();
			dispose();
			System.exit(1);
		}

		arrayArgumentNum = new int[30]; 
		arrayGPUPointers = new GPUMem[30];
		
		// Set all of the write indices to -1
		for (int i = 0; i < arrayArgumentNum.length; i++) {
			arrayArgumentNum[i] = -1;
		}
	}
	
	/** Step 3a: Set the sizes of the global work groups.
	 * @param workSizes (Variadic) This is effectively the array size(s) and dimension(s) that the GPU will iterate over.
	 */
	public void setGlobalWorkGroupSizes(long ... workSizes) {
		globalWorkSize = workSizes;
	}
	
	/** Step 3b (optional): Set the sizes of the local work groups.
	 * @param workSizes (Variadic) The global work groups are broken up into these local work groups.
	 * The global work group sizes must be divisible by these sizes.
	 */
	public void setLocalWorkGroupSizes(long ... workSizes) {
		localWorkSize = workSizes;
	}
	
	/** Step 3c (optional): Automatically calculate the local work group sizes based on
	 * the global work group sizes.
	 */
	public void updateAutoLocalWorkGroupSizes() {
		if (globalWorkSize == null) {
			error("Must assign global work group sizes before computing local work group sizes");
		}
		
		localWorkSize = new long[globalWorkSize.length];
		for (int i = 0; i < globalWorkSize.length; i++) {
			// Try to guess the best local work group size
			if (globalWorkSize[i] % 7 == 0) {
				localWorkSize[i] = 7;
			} else if (globalWorkSize[i] % 5 == 0) {
				localWorkSize[i] = 5;
			} else if (globalWorkSize[i] % 4 == 0) {
				localWorkSize[i] = 4;
			} else if (globalWorkSize[i] % 3 == 0) {
				localWorkSize[i] = 3;
			} else if (globalWorkSize[i] % 2 == 0) {
				localWorkSize[i] = 2;
			} else {
				localWorkSize[i] = 1;
			}
		}
	}
	
	/** Step 4a: Set the arguments for the given kernel.
	 * Arguments only need to be set if they have changed!  They will persist in the GPU otherwise.
	 * @param argNum The index of the parameter into the function in the OpenCL kernel to execute (starting at 0).
	 * @param arg The array to copy to the GPU.
	 * @param accessType GPUAccess.WRITE, GPUAccess.READ, or GPUAccess.READ_WRITE.
	 * If set to "WRITE" or "READ_WRITE", then this argument will be automatically copied back to the CPU after
	 * calling executeKernel().
	 * @return GPUMem pointer to the memory allocated on the GPU.
	 */
	public GPUMem setArgument(int argNum, Object arg, GPUAccess accessType) {
		return setArgument(argNum, arg, null, accessType);
	}
	
	/** Step 4b: Set the arguments for the given kernel.
	 * Arguments only need to be set if they have changed!  They will persist in the GPU otherwise.
	 * @param argNum The index of the parameter into the function in the OpenCL kernel to execute (starting at 0).
	 * @param arg The array to copy to the GPU.
	 * @param dataRange The subset of the array to copy to the GPU.
	 * @param accessType GPUAccess.WRITE, GPUAccess.READ, or GPUAccess.READ_WRITE.
	 * If set to "WRITE" or "READ_WRITE", then this argument will be automatically copied back to the CPU after
	 * calling executeKernel().
	 * @return GPUMem pointer to the memory allocated on the GPU.
	 */
	public GPUMem setArgument(int argNum, Object arg, GPURange dataRange, GPUAccess accessType) {
		
		if (argNum < 0) {
			error("Kernel argNum must be positive");
		}
		
		if (!initialized) {
			error("GPU not initialized");
		}
		
		if (arg == null) {
			error("Argument is null");
		}
		
		if (arg instanceof Float)  {
			CL.clSetKernelArg(kernel, argNum, Sizeof.cl_float, Pointer.to(new float[] { (float)arg }));
			return null;
			
		} else if (arg instanceof Integer) {
			CL.clSetKernelArg(kernel, argNum, Sizeof.cl_int, Pointer.to(new int[] { (int)arg }));
			return null;
			
		} else if (arg instanceof Long) {
			CL.clSetKernelArg(kernel, argNum, Sizeof.cl_long, Pointer.to(new long[] { (long)arg }));
			return null;
		
		} else {
			
			String argTypeName = "";
			ArrayType type = null;
			long typeSize = 0;
			long originalArrayLength = -1;
			Pointer dataPointer = null;
			
			maxArrayArgIndex = Math.max(maxArrayArgIndex, argNum);
			
			// Determine which type of object this is
			if (arg instanceof float[]) {
				float[] array = (float[])arg;
				argTypeName = "float[]";
				type = ArrayType.FLOAT;
				typeSize = Sizeof.cl_float;
				dataPointer = Pointer.to(array);
				originalArrayLength = array.length;
			} else if (arg instanceof byte[]) {
				byte[] array = (byte[])arg;
				argTypeName = "byte[]";
				type = ArrayType.BYTE;
				typeSize = Sizeof.cl_uchar;
				dataPointer = Pointer.to(array);
				originalArrayLength = array.length;
			} else if (arg instanceof int[]) {
				int[] array = (int[])arg;
				argTypeName = "int[]";
				type = ArrayType.INT;
				typeSize = Sizeof.cl_int;
				dataPointer = Pointer.to(array);
				originalArrayLength = array.length;
			} else if (arg instanceof BufferedImage) {
				
				BufferedImage image = (BufferedImage)arg;
				
				if (image.getType() == BufferedImage.TYPE_INT_RGB ||
					image.getType() == BufferedImage.TYPE_INT_BGR ||
					image.getType() == BufferedImage.TYPE_INT_ARGB_PRE ||
					image.getType() == BufferedImage.TYPE_INT_ARGB) {
					
					final DataBufferInt dataBuffer = (DataBufferInt) ((BufferedImage)arg).getRaster().getDataBuffer();
					final int[] array = dataBuffer.getData();
					type = ArrayType.BUFFERED_IMAGE_INT;
					typeSize = Sizeof.cl_int;
					dataPointer = Pointer.to(array);
					originalArrayLength = array.length;
					
				} else if (image.getType() == BufferedImage.TYPE_3BYTE_BGR ||
						   image.getType() == BufferedImage.TYPE_4BYTE_ABGR ||
						   image.getType() == BufferedImage.TYPE_4BYTE_ABGR_PRE ||
						   image.getType() == BufferedImage.TYPE_BYTE_GRAY) {
					
					final DataBufferByte dataBuffer = (DataBufferByte) ((BufferedImage)arg).getRaster().getDataBuffer();
					final byte[] array = dataBuffer.getData();
					type = ArrayType.BUFFERED_IMAGE_BYTE;
					typeSize = Sizeof.cl_uchar;
					dataPointer = Pointer.to(array);
					originalArrayLength = array.length;
					
				} else {
					error("BufferedImage must be an INT or BYTE type.");
				}
				
				argTypeName = "BufferedImage";
				
			} else if (arg == null) {
				error("Argument is null");
			} else {
				error("Invalid argument type: " + arg.getClass().getSimpleName());
			}
			
			// Disallow zero-length arrays
			if (originalArrayLength <= 0) {
				error("Cannot send zero-length arrays to GPU");
			}
			
			// If we weren't given a range, then assume the full array length
			if (dataRange == null) {
				dataRange = new GPURange(0, originalArrayLength);
			}
			
			cl_mem mem = null;
			
			// Find the pointer to memory if it already exists
			if (arrayGPUPointers[argNum] != null) {
				
				if (arrayGPUPointers[argNum].type != type) {
					error("Argument is a " + argTypeName + ", but GPUMem points to a " + arrayGPUPointers[argNum].type);
				}

				mem = arrayGPUPointers[argNum].mem;
				if (mem != null) {
					
					// If we need to expand memory to fit this new array, then deallocate the old one.
					if (dataRange.size > arrayGPUPointers[argNum].maxAllocatedSize) {
						//CL.clReleaseMemObject(mem); // TODO releasing this memory object makes the whole program go nuts when multithreading. Causes memory corruption.
						// TODO there are apparently issues with clReleaseMemObject. It probably doesn't play well with Java's garbage collector.
						// TODO This could be the source of a memory leak, so be careful!
						// TODO Is this issue now fixed since I fill allocated memory with zeros?
						mem = null; // Force reallocation below
					}
					
					// Protect against changing the access type of memory that already exists
					if (arrayGPUPointers[argNum].accessType != accessType) {
						error("Access type changed!");
					}
				}
			}
			
			// Check whether the memory is in range
			if (dataRange.end > originalArrayLength) {
				error("GPURange " + dataRange + " overruns array of length " + originalArrayLength);
			}
			
			// If we don't already have a GPU buffer, then allocate one on the GPU.
			if (mem == null) {
				mem = CL.clCreateBuffer(context, accessType.value, dataRange.size * typeSize, null, null);
				allocCounter++;
			}
			
			// Copy this array to the GPU
			if (accessType == GPUAccess.READ || accessType == GPUAccess.READ_WRITE) {
				// This step takes a long time and is heavily affected by GPU memory clock at the moment.
				CL.clEnqueueWriteBuffer(commandQueue, mem, true, 0, dataRange.size * typeSize,
						dataPointer.withByteOffset(dataRange.start * typeSize), 0, null, null);
				copyToGPUCounter++;
			}
			
			CL.clSetKernelArg(kernel, argNum, Sizeof.cl_mem, Pointer.to(mem));
			
			// Keep track of the array for later if needed
			arrayArgumentNum[argNum] = argNum;
			if (arrayGPUPointers[argNum] == null) {
				arrayGPUPointers[argNum] = new GPUMem(mem, dataPointer, type, dataRange, accessType);
			} else {
				arrayGPUPointers[argNum].mem = mem;
				arrayGPUPointers[argNum].accessType = accessType;
				arrayGPUPointers[argNum].arrayRange = dataRange;
				arrayGPUPointers[argNum].maxAllocatedSize = Math.max(arrayGPUPointers[argNum].maxAllocatedSize, dataRange.size);
				arrayGPUPointers[argNum].pointer = dataPointer;
			}
			
			return arrayGPUPointers[argNum];
		}
	}
	
	/** Step 4d: Set the arguments for the given kernel with a pointer to memory already on the GPU.
	 * Arguments only need to be set if they have changed!  They will persist in the GPU otherwise.
	 * @param argNum The index of the parameter into the function in the OpenCL kernel to execute (starting at 0).
	 * @param gpuMemPointer A GPUMem reference to the memory on the GPU to use for this argument.
	 * The GPURange and access-type is automatically retrieved from the GPUMem.
	 */
	public void setArgument(int argNum, GPUMem gpuMemPointer) {
		
		if (argNum < 0) {
			error("Kernel argNum must be positive");
		}
		
		if (!initialized) {
			error("GPU not initialized");
		}
		
		if (gpuMemPointer == null) {
			error("GPUMem must not be null");
		}
		
		maxArrayArgIndex = Math.max(maxArrayArgIndex, argNum);
		
		// Don't do error checking if this GPUMem was deallocated anyway.
		if (arrayGPUPointers[argNum] != null && arrayGPUPointers[argNum].mem != null) {
			
			// Check if the pointer to the cl_mem changed
			if (arrayGPUPointers[argNum].mem != gpuMemPointer.mem) {
				error("Duplicate allocated GPU memory: " + arrayGPUPointers[argNum].mem + " != " + gpuMemPointer.mem + ".\n" +
						"Did you mean to dispose the old memory first?");
			}
	
			// Check if the size changed
			if (arrayGPUPointers[argNum].arrayRange.size != gpuMemPointer.arrayRange.size) {
				error("Wrong size! " + arrayGPUPointers[argNum].arrayRange.size + " != " + gpuMemPointer.arrayRange.size);
			}
		}
		
		CL.clSetKernelArg(kernel, argNum, Sizeof.cl_mem, Pointer.to(gpuMemPointer.mem));
		
		// Keep track of the array for later if needed
		arrayArgumentNum[argNum] = argNum;
		arrayGPUPointers[argNum] = gpuMemPointer;
	}
	
	/** Step 5a: Process the data on the GPU and copy the results back.
	 * If this is used, then you can skip executeKernelNoCopyback() and copyFromGPU().
	 */
	public void executeKernel() {
		
		// Execute the kernel
		executeKernelNoCopyback();
		
		// Copy all the results back from the GPU
		copyFromGPU();
	}
	
	/** Step 5b: Process the data on the GPU *without* automatically copying the results back.
	 * To automatically copy back memory, use executeKernel() instead.
	 * To copy memory back to the CPU manually, use copyFromGPU().
	 */
	public void executeKernelNoCopyback() {
		
		if (globalWorkSize == null) {
			error("Global work size not set. (Use setGlobalWorkGroupSizes)");
		}
		
		if (localWorkSize != null) {
			
			// Check if the work sizes are compatible
			if (localWorkSize.length != globalWorkSize.length) {
				error("Dimension of local work group must equal dimension of global work group!");
			}
			
			// Check for correct local work sizes
			for (int i = 0; i < localWorkSize.length; i++) {
				if (globalWorkSize[i] % localWorkSize[i] != 0) {
					error("Global work-group size must be divisible by local work-group size: " +
							globalWorkSize[i] + " % " + localWorkSize[i] + " != 0");
				}
			}
		}
		
		// This does the actual processing
		CL.clEnqueueNDRangeKernel(commandQueue, kernel, globalWorkSize.length,
				null, globalWorkSize, localWorkSize, 0, null, null);
		
		// Wait for the computation to finish
		CL.clFinish(commandQueue);
	}

	/** Step 5c: Copy the data back from the GPU.
	 * This is done automatically by executeKernel().
	 * Don't need this if you called executeKernel().
	 * This is a blocking call.
	 */
	public void copyFromGPU() {
		
		// Use this array to deduplicate copies.
		int deduplicatorIndex = 0;
		GPUMem[] referencesAlreadyCopied = null;
		
		for (int i = 0; i < maxArrayArgIndex + 1; i++) {
			if (arrayGPUPointers[i] != null) {
				// If we might have written to this array on the GPU, then copy it back to the CPU.
				if (arrayGPUPointers[i].accessType == GPUAccess.WRITE ||
					arrayGPUPointers[i].accessType == GPUAccess.READ_WRITE) {
					
					// Check that we didn't already copy this one previously
					boolean alreadyCopied = false;
					if (referencesAlreadyCopied != null) {
						for (int j = 0; j < referencesAlreadyCopied.length; j++) {
							if (referencesAlreadyCopied[j] == null || referencesAlreadyCopied[j] == arrayGPUPointers[i]) {
								alreadyCopied = true;
								break;
							}
						}
					}
					
					// If we haven't already copied this GPUMem to the CPU, then copy it over now.
					if (!alreadyCopied) {
						long argTypeSize = arrayGPUPointers[i].type.getSize();
						CL.clEnqueueReadBuffer(commandQueue, arrayGPUPointers[i].mem, true, 0,
								argTypeSize * arrayGPUPointers[i].arrayRange.size,
								arrayGPUPointers[i].pointer.withByteOffset(arrayGPUPointers[i].arrayRange.start * argTypeSize),
								0, null, null);
						copyToCPUCounter++;
					}
					
					// Record this in the list of GPUMems that were already copied to the CPU.
					if (referencesAlreadyCopied == null) {
						referencesAlreadyCopied = new GPUMem[maxArrayArgIndex + 1];
					}
					referencesAlreadyCopied[deduplicatorIndex] = arrayGPUPointers[i];
					deduplicatorIndex++;
				}
			}
		}
	}
	
	/** Step 6: Free all memory.
	 * Call this to release all resources that were used for this instance.
	 */
	public void dispose() {
		
		// Finish all operations on the command queue
		if (commandQueue != null) {
			CL.clFlush(commandQueue);
			CL.clFinish(commandQueue);
		}
		
		// Release all of the arguments
		releaseGPUMemory();
		
		// Release the main things
		if (kernel != null) {
			CL.clReleaseKernel(kernel);
		}
		if (program != null) {
			CL.clReleaseProgram(program);
		}
		
		// Clear the data for Java
		arrayGPUPointers = null;
		program = null;
		kernel = null;
	}
	
	/** Deallocate all memory objects on the GPU. Arguments can now be reused.
	 */
	public void releaseGPUMemory() {
		
		// Release all of the arguments
		if (arrayGPUPointers != null) {
			for (int i = 0; i < arrayGPUPointers.length; i++) {
				if (arrayGPUPointers[i] != null) {
					arrayGPUPointers[i].dispose();
				}
				arrayGPUPointers[i] = null;
			}
		}
	}
	
	/** Reserve blank memory on the GPU, and return a GPUMem pointer to that memory.
	 * @param arr The array associated with this GPUMem object, whose length is used for allocation size.
	 * @param accessType GPUAccess.WRITE, GPUAccess.READ, or GPUAccess.READ_WRITE
	 * @param fillWithZeros Whether to fill the newly allocated memory with zeros or not.
	 * @return GPUMem pointer to the new memory allocated on the GPU.
	 */
	public static GPUMem allocateMemoryOnGPU(Object arr, GPUAccess accessType, boolean fillWithZeros) {
		
		if (arr == null) {
			error2("Argument is null");
		}
		
		ArrayType type = null;
		long numElements = -1;
		long typeSize = 0;
		Pointer arrayPointer = null;
		if (arr instanceof float[]) {
			type = ArrayType.FLOAT;
			numElements = ((float[])arr).length;
			typeSize = Sizeof.cl_float;
			arrayPointer = Pointer.to((float[])arr);
		} else if (arr instanceof byte[]) {
			type = ArrayType.BYTE;
			numElements = ((byte[])arr).length;
			typeSize = Sizeof.cl_uchar;
			arrayPointer = Pointer.to((byte[])arr);
		} else if (arr instanceof int[]) {
			type = ArrayType.INT;
			numElements = ((int[])arr).length;
			typeSize = Sizeof.cl_int;
			arrayPointer = Pointer.to((int[])arr);
		} else {
			error2("Unimplemented array type.");
		}
		
		cl_mem mem = CL.clCreateBuffer(context, accessType.value, numElements * typeSize, null, null);
		allocCounter++;
		
		if (fillWithZeros) {
			CL.clEnqueueFillBuffer(commandQueue, mem, Pointer.to(new byte[] {0}), 1, 0, numElements * typeSize, 0, null, null);
			//CL.clFlush(commandQueue);
			//CL.clFinish(commandQueue); // TODO are these necessary?
		}
		
		return new GPUMem(mem, arrayPointer, type, new GPURange(0, numElements), accessType);
	}
	
	/** Copy an array to the GPU and return a GPU pointer.
	 * If existingMem is specified, then it will copy over that GPU memory.
	 * @param existingMem Copy the array associated with this GPUMem to the existing allocated location on the GPU.
	 * @return GPUMem pointer to the array on the GPU.
	 */
	public static GPUMem copyArrayToGPU(GPUMem existingMem) {
		return copyArrayToGPU(null, existingMem, null, 0, null);
	}
	
	/** Copy an array to the GPU and return a GPU pointer.
	 * If existingMem is specified, then it will copy over that GPU memory.
	 * @param arr The array to copy to the GPU.
	 * @param accessType GPUAccess.WRITE, GPUAccess.READ, or GPUAccess.READ_WRITE
	 * @return GPUMem pointer to the array on the GPU.
	 */
	public static GPUMem copyArrayToGPU(Object arr, GPUAccess accessType) {
		return copyArrayToGPU(arr, null, null, 0, accessType);
	}
	
	/** Copy an array to the GPU and return a GPU pointer.
	 * If existingMem is specified, then it will copy over that GPU memory.
	 * @param arr The array to copy to the GPU.
	 * @param existingMem (Optional) The existing memory to overwrite on the GPU.
	 * @return GPUMem pointer to the array on the GPU.
	 */
	public static GPUMem copyArrayToGPU(Object arr, GPUMem existingMem) {
		return copyArrayToGPU(arr, existingMem, null, 0, null);
	}
	
	/** Copy an array to the GPU and return a GPU pointer.
	 * If existingMem is specified, then it will copy over that GPU memory.
	 * @param arr The array to copy to the GPU.
	 * @param existingMem (Optional) The existing memory to overwrite on the GPU.
	 * @param destOffset The offset (in sizeof(type)) in the GPU memory to copy to.
	 * @return GPUMem pointer to the array on the GPU.
	 */
	public static GPUMem copyArrayToGPU(Object arr, GPUMem existingMem, int destOffset) {
		return copyArrayToGPU(arr, existingMem, null, destOffset, null);
	}
	
	/** Copy an array to the GPU and return a GPU pointer.
	 * If existingMem is specified, then it will copy over that GPU memory.
	 * @param arr The array to copy to the GPU.
	 * @param existingMem (Optional) The existing memory to overwrite on the GPU.
	 * @param sourceRange (Optional) The subset of the 'arr' to be copied to the GPU.
	 * @param destOffset The offset (in sizeof(type)) in the GPU memory to copy to.
	 * @param accessType GPUAccess.WRITE, GPUAccess.READ, or GPUAccess.READ_WRITE
	 * @return GPUMem pointer to the array on the GPU.
	 */
	public static GPUMem copyArrayToGPU(Object arr, GPUMem existingMem, GPURange sourceRange, int destOffset, GPUAccess accessType) {
		
		if (accessType == GPUAccess.WRITE) {
			error2("Cannot copy write-only memory to GPU!\n" +
					"(It can only be written to by the GPU.)");
		}
		
		if (destOffset < 0) {
			error2("Destination offset cannot be negative.");
		}

		ArrayType type = null;
		String argTypeName = null;
		long originalArrayLength = -1;
		long typeSize = 0;
		Pointer dataPointer = null;
		
		// If we weren't given an array, then get it from the GPUMem
		if (arr == null) {
			if (existingMem != null) {
				type = existingMem.type;
				argTypeName = "<unknown>";
				originalArrayLength = existingMem.arrayRange.size; // Just a guess
				typeSize = existingMem.type.getSize();
				dataPointer = existingMem.pointer;
			} else {
				error2("GPUMem and array arguments can't both be null");
			}
		} else {
			if (arr instanceof float[]) {
				type = ArrayType.FLOAT;
				argTypeName = "float[]";
				originalArrayLength = ((float[])arr).length;
				typeSize = Sizeof.cl_float;
				dataPointer = Pointer.to((float[])arr);
			} else if (arr instanceof byte[]) {
				type = ArrayType.BYTE;
				argTypeName = "byte[]";
				originalArrayLength = ((byte[])arr).length;
				typeSize = Sizeof.cl_uchar;
				dataPointer = Pointer.to((byte[])arr);
			} else if (arr instanceof int[]) {
				type = ArrayType.INT;
				argTypeName = "int[]";
				originalArrayLength = ((int[])arr).length;
				typeSize = Sizeof.cl_int;
				dataPointer = Pointer.to((int[])arr);
			} else {
				error2("Unsupported array type.");
			}
		}
		
		// If we don't have an access type, then get it from the GPUMem
		if (existingMem != null && accessType == null) {
			accessType = existingMem.accessType;
		}
		
		// Invalid cases...
		if (existingMem != null) {
			
			if (existingMem.arrayRange == null || existingMem.accessType == null) {
				error2("Attempted to access deallocated GPUMem object.");
			}
			
			if (existingMem.type != type) {
				error2("Argument is a " + argTypeName + ", but GPUMem points to a " + existingMem.type);
			}
			
			if (existingMem.accessType != accessType) {
				error2("Access type mismatch.");
			}
		}
		
		// If we don't have a GPURange, then create one
		if (sourceRange == null) {
			//if (existingMem != null) {
			//	sourceRange = existingMem.arrayRange;
			//} else {
				sourceRange = new GPURange(0, originalArrayLength);
			//}
		}
		
		// Check whether the memory is in range
		if (sourceRange.end > originalArrayLength) {
			error2("GPURange " + sourceRange + " overruns array of length " + originalArrayLength);
		}
		
		// If we don't already have a GPU buffer, then allocate one on the GPU.
		if (existingMem == null) {
			cl_mem mem = CL.clCreateBuffer(context, accessType.value, sourceRange.size * typeSize, null, null);
			existingMem = new GPUMem(mem, dataPointer, type, sourceRange, accessType);
			allocCounter++;
		}
		
		// Check if we overrun the allocated size
		if (sourceRange.size + destOffset > existingMem.maxAllocatedSize) {
			error2("Range [" + destOffset + ", " + (destOffset + sourceRange.size) +
					") does not fit inside GPUMem of max-length " + existingMem.maxAllocatedSize + ".");
		}
		
		// This step takes a long time and is heavily affected by GPU memory clock at the moment.
		CL.clEnqueueWriteBuffer(commandQueue, existingMem.mem, true, destOffset * typeSize, sourceRange.size * typeSize,
						dataPointer.withByteOffset(sourceRange.start * typeSize), 0, null, null);
		copyToGPUCounter++;
		
		existingMem.arrayRange = sourceRange;
		existingMem.pointer = dataPointer;
		
		return existingMem;
	}
	
	/** Copy from the GPU to the CPU.
	 * @param source A GPUMem pointer who's memory on the GPU will be copied back to the associated array on the CPU.
	 */
	public static void copyArrayToCPU(GPUMem source) {
		copyArrayToCPU_helper(source, source.pointer, source.arrayRange.size, source.type.getSize());
	}
	
	/** Copy from the GPU to the CPU.
	 * @param source A GPUMem pointer to memory on the GPU to copy from.
	 * @param destArray A Java array to copy the data into.
	 */
	public static void copyArrayToCPU(GPUMem source, Object destArray) {
		
		if (destArray == null) {
			error2("Array to copy to must not be null.");
		}
		
		long numElements = -1;
		long typeSize = 0;
		Pointer dataPointer = null;
		if (destArray instanceof float[]) {
			numElements = ((float[])destArray).length;
			typeSize = Sizeof.cl_float;
			dataPointer = Pointer.to((float[])destArray);
		} else if (destArray instanceof byte[]) {
			numElements = ((byte[])destArray).length;
			typeSize = Sizeof.cl_uchar;
			dataPointer = Pointer.to((byte[])destArray);
		} else if (destArray instanceof int[]) {
			numElements = ((int[])destArray).length;
			typeSize = Sizeof.cl_int;
			dataPointer = Pointer.to((int[])destArray);
		} else {
			error2("Unsupported array type.");
		}
		
		copyArrayToCPU_helper(source, dataPointer, numElements, typeSize);
	}
	
	// Local helper function for the above two
	private static void copyArrayToCPU_helper(GPUMem source, Pointer dataPointer, final long numElements, final long typeSize) {
		
		if (source == null) {
			error2("Source GPUMem is null.");
		}
		
		if (dataPointer == null) {
			error2("Pointer to CPU memory is null");
		}
		
		if (source.arrayRange == null || source.accessType == null) {
			error2("Attempted to access deallocated GPUMem object.");
		}
		
		// Size checks
		if (source.arrayRange.size > numElements) {
			error2("Cannot copy " + source.arrayRange.size +
					" elements from GPU to array of length "+ numElements + ".");
		}
		if (source.arrayRange.end > numElements) {
			error2("GPU array overruns end of CPU array: " + source.arrayRange +
					" into [0," + numElements + ").");
		}
		
		// Copy data from the GPU to main memory
		CL.clEnqueueReadBuffer(commandQueue, source.mem, true, 0,
				typeSize * source.arrayRange.size,
				dataPointer.withByteOffset(source.arrayRange.start * typeSize), 0, null, null);
		copyToCPUCounter++;
	}
	
	/** Copy between two buffers on the GPU
	 * @param source A GPUMem pointer to the memory on the GPU to copy from.
	 * @param dest A GPUMem pointer to the memory on the GPU to be overwritten.
	 */
	public static void copyGPUMem(GPUMem source, GPUMem dest) {
		copyGPUMem(source, dest, null, null);
	}
	
	/** Copy between two buffers on the GPU
	 * @param source A GPUMem pointer to the memory on the GPU to copy from.
	 * @param dest A GPUMem pointer to the memory on the GPU to be overwritten.
	 * @param sourceRange A GPURange indicating the subset of the source memory to copy from.
	 * @param destRange A GPURange indicating the subset of the destination memory to copy to.
	 * (Must match the length of the source).
	 */
	public static void copyGPUMem(GPUMem source, GPUMem dest, GPURange sourceRange, GPURange destRange) {
		
		if (dest == null) {
			error2("Destination GPUMem is null!");
		}
		
		if (sourceRange == null) {
			sourceRange = source.arrayRange;
		} else {
			if (sourceRange.end > source.maxAllocatedSize) {
				error2("Range " + sourceRange + " out of bounds for source " + source.arrayRange);
			}
		}
		
		if (destRange == null) {
			destRange = dest.arrayRange;
		} else {
			if (destRange.end > dest.maxAllocatedSize) {
				error2("Range " + destRange + " out of bounds for source " + dest.arrayRange);
			}
		}
		
		if (sourceRange.size != destRange.size) {
			error2("GPUMem of size " + sourceRange.size + " doesn't match destination size " + destRange.size);
		}
		
		if (source.type != dest.type) {
			error2("Cannot copy between " + source.type + " and " + dest.type);
		}

		long typeSize = 1;
		if (source.type == ArrayType.BUFFERED_IMAGE_INT) {
			typeSize = Sizeof.cl_int;
		} else if (source.type == ArrayType.BUFFERED_IMAGE_BYTE) {
			typeSize = Sizeof.cl_uchar;
		} else if (source.type == ArrayType.BYTE) {
			typeSize = Sizeof.cl_uchar;
		} else if (source.type == ArrayType.DOUBLE) {
			typeSize = Sizeof.cl_double;
		} else if (source.type == ArrayType.FLOAT) {
			typeSize = Sizeof.cl_float;
		} else if (source.type == ArrayType.INT) {
			typeSize = Sizeof.cl_int;
		} else if (source.type == ArrayType.LONG) {
			typeSize = Sizeof.cl_long;
		} else {
			error2("Unimplemented ArrayType: " + source.type);
		}
		
		CL.clEnqueueCopyBuffer(commandQueue, source.mem, dest.mem,
				sourceRange.start * typeSize, destRange.start * typeSize, sourceRange.size * typeSize, 0, null, null);
		copyCounter++;
	}
	
	/** Return the number of bytes of global memory in this GPU.
	 * @return bytes
	 */
	public static long getGlobalMemory() {
		GPUProgram.initializeGPU();
		
		long[] value = {0};
		CL.clGetDeviceInfo(device, CL.CL_DEVICE_GLOBAL_MEM_SIZE,
				Sizeof.cl_ulong, Pointer.to(value), new long[] {value.length});
		return value[0];
	}
	
	/** Returns the maximum size of a single memory object on the GPU.
	 * @return bytes
	 */
	public static long getMaxMemAllocSize() {
		GPUProgram.initializeGPU();
		
		long[] value = {0};
		CL.clGetDeviceInfo(device, CL.CL_DEVICE_MAX_MEM_ALLOC_SIZE,
				Sizeof.cl_ulong, Pointer.to(value), new long[] {value.length});
		return value[0];
	}
	
	/** Return the maximum size of a local work group
	 */
	public static int getMaxLocalWorkGroupSize() {
		return (int)getDeviceInfoInt(device, CL.CL_DEVICE_MAX_WORK_GROUP_SIZE);
	}
	
	/** Display metrics for the GPU.
	 */
	public static void printDeviceStatistics() {
		GPUProgram.initializeGPU();
		
		print("Device Name: " + getDeviceInfoString(device, CL.CL_DEVICE_VENDOR) + " " + getDeviceInfoString(device, CL.CL_DEVICE_NAME));
		print("Parallel Compute Units: " + getDeviceInfoInt(device, CL.CL_DEVICE_MAX_COMPUTE_UNITS));
		print("Local Memory Size: " + getDeviceInfoInt(device, CL.CL_DEVICE_LOCAL_MEM_SIZE));
		print("Global Memory Size: " + getDeviceInfoInt(device, CL.CL_DEVICE_GLOBAL_MEM_SIZE));
		print("Max Allocated Memory Size: " + getDeviceInfoInt(device, CL.CL_DEVICE_MAX_MEM_ALLOC_SIZE));
		print("Max Local Work Group Size: " + getDeviceInfoInt(device, CL.CL_DEVICE_MAX_WORK_GROUP_SIZE));
		print("Max Local Work Group Size per Dimension: " + getDeviceInfoArray(device, CL.CL_DEVICE_MAX_WORK_ITEM_SIZES));
		print("Max Work Dimensions: " + getDeviceInfoInt(device, CL.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
	}
	
	/** Return whether this kernel argument has already been set.
	 * @param argNum Argument number to check.
	 * @return true if the argument has already been set for the kernel.
	 */
	public boolean isArgumentSet(int argNum) {
		return arrayGPUPointers[argNum] != null;
	}
	
	/** Return whether the GPU has been initialized.
	 * @return true if initializeGPU() was already called.
	 */
	public static boolean isInitialized() {
		return initialized;
	}
	
	/** Reset the debug counters to zero.
	 */
	public static void resetDebugCounters() {
		copyToGPUCounter = 0;
		copyToCPUCounter = 0;
		allocCounter = 0;
		copyCounter = 0;
	}
	
	/** Print out the current values of all debug counters.
	 */
	public static void printDebugCounters() {
		print("Copies to GPU: " + copyToGPUCounter);
		print("Copies to CPU: " + copyToCPUCounter);
		print("Allocs on GPU: " + allocCounter);
		print("Copies on GPU: " + copyCounter);
	}
	
	// Returns the value of the device info parameter with the given name
	private static String getDeviceInfoString(cl_device_id device, int paramName) {
		// Obtain the length of the string that will be queried
		long size[] = new long[1];
		CL.clGetDeviceInfo(device, paramName, 0, null, size);
		
		// Create a buffer of the appropriate size and fill it with the info
		byte buffer[] = new byte[(int) size[0]];
		CL.clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);
		
		// Create a string from the buffer (excluding the trailing \0 byte)
		return new String(buffer, 0, buffer.length - 1);
	}
	
	// Get some integer value from the device properties
	private static long getDeviceInfoInt(cl_device_id device, int paramName) {
		long[] value = {0};
		CL.clGetDeviceInfo(device, paramName, Sizeof.cl_ulong, Pointer.to(value), new long[] {value.length});
		return value[0];
	}
	
	// Get some array value from the device properties
	private static String getDeviceInfoArray(cl_device_id device, int paramName) {
		long[] values = new long[3];
		long[] size = {0};
		CL.clGetDeviceInfo(device, paramName, Sizeof.cl_ulong * values.length, Pointer.to(values), size);
		int parameterCount = Math.min(values.length, (int)size[0]);
		
		String s = "[";
		for (int i = 0; i < parameterCount; i++) {
			s += values[i];
			if (i != parameterCount - 1) {
				s += " ";
			}
		}
		return s + "]";
	}
	
	// Conveniently print an error
	private void error(String s) {
		new Exception(s).printStackTrace();
		dispose();
		System.exit(1);
	}
	
	// Conveniently print an error
	private static void error2(String s) {
		new Exception(s).printStackTrace();
		System.exit(1);
	}
	
	private static void print(Object o) {
		System.out.println(o);
	}
}
