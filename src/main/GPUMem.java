package main;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_mem;

// GPUMem keeps track of a pointer to memory in the GPU, and the type of memory.
// This goes with GPUProgram.

public class GPUMem {
	protected final ArrayType type;	// Type of the original array.
	protected cl_mem mem;			// Pointer to memory on the GPU.
	protected GPURange arrayRange;	// Elements in the original array to copy into from the GPU. (May be smaller than maxAllocatedSize.)
	protected GPUAccess accessType;		// Read, Write, Read-Write.
	protected long maxAllocatedSize;	// Size of the cl_mem on the GPU (in elements, not bytes)
	protected Pointer pointer;		// Pointer to the Java array to read or write to.
	
	protected GPUMem(cl_mem mem, Pointer arrayPointer, ArrayType type, GPURange arrayRange, GPUAccess accessType) {
		this.mem = mem;
		this.type = type;
		this.arrayRange = arrayRange;
		this.accessType = accessType;
		this.maxAllocatedSize = arrayRange.size;
		this.pointer = arrayPointer;
	}
	
	// Change the range over the original array that we will copy to or from.
	public void setRange(GPURange newRange) {
		if (newRange.size > maxAllocatedSize) {
			new Exception("New size " + newRange + " exceeds maximum allocated, "
						+ maxAllocatedSize + ".").printStackTrace();
			System.exit(1);
		}
		
		this.arrayRange = newRange;
	}
	
	// Release all GPU memory and make sure this object cannot be used again.
	public void dispose() {
		if (mem != null) {
			CL.clReleaseMemObject(mem);
			mem = null;
		}
		arrayRange = null;
		pointer = null;
		accessType = null;
		maxAllocatedSize = -1;
	}
	
	// Return the access type for this GPU memory.
	public GPUAccess getAccessType() {
		return accessType;
	}
}
