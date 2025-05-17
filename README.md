**GPUProgram** is a Java library for easily running code on the GPU using OpenCL.
It runs on top of the JOCL library, which is included in the lib folder.

Although not as flexible as directly using the OpenCL API, it handles the vast majority
of common cases while allowing cleaner, more readable code.

# Features

- Big performance boost compared to pure Java
- Perform automatic CPU-to-GPU-to-CPU memory copies in simple cases
- Perform operations on BufferedImages directly
- Cross platform (Mac, Windows, and probably Linux)
- Can be deployed in production (i.e. can be bundled into a portable format that "just works" when you double click it)
- Automatic memory cleanup
- Single GPU support only (currently)

# Data type support

- Fully supports GPU manipulation of:
    - `float[]`
	- `byte[]`
	- `int[]`
- And partially supports manipulating:
    - `double[]`
	- `long[]`
	- `BufferedImage` of type `TYPE_INT_RGB`, `TYPE_INT_BGR`, and `TYPE_INT_ARGB`
- And the following primitives:
    - `float`
	- `int`
	- `long`
- The library will give you an error if you use an unsupported type.

# Example

This example implements parallel vector addition: a + b = c.

VectorAdd.java:
```java
GPUProgram vecAddProgram = new GPUProgram("vectorAddKernel", "src/examples/VectorAdd.cl");

float[] a = {0.6f, 0.5f, 0.4f, 0.0f, 0.0f, 0.0f};
float[] b = {0.0f, 0.0f, 0.0f, 0.2f, 0.3f, 0.4f};
float[] c = new float[a.length];

vecAddProgram.setArgument(0, a, GPUAccess.READ);
vecAddProgram.setArgument(1, b, GPUAccess.READ);
vecAddProgram.setArgument(2, c, GPUAccess.WRITE);
vecAddProgram.setGlobalWorkGroupSizes(a.length);
vecAddProgram.executeKernel();

System.out.println(Arrays.toString(c));
```

VectorAdd.cl:
```opencl
kernel void vectorAddKernel(global const float* a, global const float* b, global float* c) {
	int i = get_global_id(0);
	c[i] = a[i] + b[i];
}
```

# Code Standard

1. No lambda expressions
1. 100% imperative code
1. Java 8+ compatability
1. Use tabs or else
1. `abstract`, `interface`, `extends`, and `implements` are banned

# Contributing

1. Pull repo
1. Make a branch
1. Commit to branch
1. Make pull request into master
1. I will either be accept or reject
1. If I take too long, email me.

OR

1. Create an issue
1. Email me if I don't respond or resolve it quickly

# Troubleshooting

1. Error: `The import org.jocl cannot be resolved`
    - If using Eclipse, create a new User Library and include the .jar files in the 'lib' folder.
1. Error: `UnsatisfiedLinkError: Error while loading native library...`
    - If using Eclipse, edit the libraries in your build path to include the 'lib' folder.
	- Set the "Native library location" to 'lib'
1. Error: `CL_INVALID_COMMAND_QUEUE`
	- This could mean a lot of things:
	- SEGFAULT on the GPU, usually caused by iterating off the end of an array in OpenCL
	- Some argument to the kernel is incorrect
	- You have a global or local work group set too large for your GPU
1. Error: `Global work size not set`
    - You need to call the `GPUProgram.setGlobalWorkGroupSizes` function to decide how many threads to invoke in parallel
1. Error: `CL_INVALID_KERNEL`
    - You may be trying to use a GPUProgram which has already been disposed using the `.dispose()` function
1. Error: `CL_BUILD_PROGRAM_FAILURE`
    - Syntax error in the source OpenCL file being loaded into the GPUProgram class
1. Error: `java.lang.NoClassDefFoundError: org/jocl/NativePointerObject`
	- If running from a JAR file, make sure that it was built with JOCL-0.2.0RC.jar and has that jar in an accessible location
	- Make sure that the .dll files (in the 'lib' folder) are in the same directory that the command is being run from