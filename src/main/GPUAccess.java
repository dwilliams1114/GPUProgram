package main;

import org.jocl.CL;

/**
 * Enum representing read-only, write-only, and read-write access on the GPU.
 * When we speak of reading or writing, we are referring to the GPU doing the reading or writing, not the CPU.
 */
public enum GPUAccess {
	READ(CL.CL_MEM_READ_ONLY),
	WRITE(CL.CL_MEM_WRITE_ONLY),
	READ_WRITE(CL.CL_MEM_READ_WRITE);
	
	protected final long value;
	
	GPUAccess(long accessType) {
		this.value = accessType;
	}
}
