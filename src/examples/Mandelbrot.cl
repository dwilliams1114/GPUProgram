/**
 * This kernel renders the Mandelbrot Set onto the given BufferedImage data buffer for the given BufferedImage type.
 * 
 * Programmed by Daniel Williams in 2025.
 */

// Types of buffered images in Java:
#define BUFFERED_IMAGE_TYPE_INT_RGB			0x1
#define BUFFERED_IMAGE_TYPE_INT_ARGB		0x2
#define BUFFERED_IMAGE_TYPE_INT_ARGB_PRE	0x3
#define BUFFERED_IMAGE_TYPE_INT_BGR			0x4
#define BUFFERED_IMAGE_TYPE_3BYTE_BGR		0x5
#define BUFFERED_IMAGE_TYPE_4BYTE_ABGR		0x6
#define BUFFERED_IMAGE_TYPE_4BYTE_ABGR_PRE	0x7
#define BUFFERED_IMAGE_TYPE_BYTE_GRAY		0xA

// Given some number of iterations required to escape the Mandelbrot, return the corresponding color.
// This computes a bunch of color curves to make the image look smooth and pleasing.
uint getColor(int iterations, int maxIterations) {

	float brightness = sin(M_PI_2_F * iterations / maxIterations);
	
	float red = pow(brightness, 1.4f);
	float blue = pow(brightness, 0.6f);
	float green = pow(brightness, 1.5f);
	
	uint r = (uint)clamp(red   * 255, 0.0f, 255.0f);
	uint g = (uint)clamp(green * 255, 0.0f, 255.0f);
	uint b = (uint)clamp(blue  * 255, 0.0f, 255.0f);
	uint a = 0xFF; // Alpha should be opaque
	
	return (a << 24) | (r << 16) | (g << 8) | b;
}

// Extract the red channel from a bit-wise color
uchar getRed(uint color) {
	return (color >> 16) & 0xFF;
}

// Extract the green channel from a bit-wise color
uchar getGreen(uint color) {
	return (color >> 8) & 0xFF;
}

// Extract the blue channel from a bit-wise color
uchar getBlue(uint color) {
	return color & 0xFF;
}

// Extract the alpha channel from a bit-wise color
uchar getAlpha(uint color) {
	return (color >> 24) & 0xFF;
}

// Swap the red and blue channels
uint swapRedAndBlue(uint color) {
	return (0xFF << 24) | ((color & 0xFF0000) >> 16) | (color & 0xFF00) | ((color & 0xFF) << 16);
}

// Main kernel, called from Java:
kernel void renderMandelbrot(
    global uchar *byteOutputImage,
    const int width, const int height,
    const float minX, const float minY,
    const float maxX, const float maxY,
    const int maxIterations,
	const int imageType)
{
	// Get which pixel we need to compute in the image
    uint pixelX = get_global_id(0);
    uint pixelY = get_global_id(1);
	
    float r = minX + pixelX * (maxX - minX) / width;
    float i = minY + pixelY * (maxY - minY) / height;

    float x = 0;
    float y = 0;
	
	// Iterate the imaginary Mandelbrot equation until divergence, or until max iterations
    int iteration = 0;
    while (iteration < maxIterations) {
		float xx = x * x;
		float yy = y * y;
		y = 2 * x * y + i;
		x = xx - yy + r;
		if (xx + yy > 4) {
			break;
		}
		iteration++;
    }
	
	// Convert the number of iterations into a color.
	// i.e., color the image based on distance to the chaotic edge.
	uint color = getColor(iteration, maxIterations);
	
	// Handle each type of BufferedImage in its own way
	if (imageType == BUFFERED_IMAGE_TYPE_INT_RGB ||
		imageType == BUFFERED_IMAGE_TYPE_INT_ARGB ||
		imageType == BUFFERED_IMAGE_TYPE_INT_ARGB_PRE) {
		
		uint *intOutputImage = (uint *)byteOutputImage;
		intOutputImage[pixelY * width + pixelX] = color;
		
	} else if (imageType == BUFFERED_IMAGE_TYPE_INT_BGR) {
		
		uint *intOutputImage = (uint *)byteOutputImage;
		intOutputImage[pixelY * width + pixelX] = swapRedAndBlue(color);
		
	} else if (imageType == BUFFERED_IMAGE_TYPE_3BYTE_BGR) {
		
		uint pixelIndex = (pixelY * width + pixelX) * 3;
		byteOutputImage[pixelIndex + 0] = getBlue(color);
		byteOutputImage[pixelIndex + 1] = getRed(color);
		byteOutputImage[pixelIndex + 2] = getGreen(color);
		
	} else if (imageType == BUFFERED_IMAGE_TYPE_4BYTE_ABGR ||
			   imageType == BUFFERED_IMAGE_TYPE_4BYTE_ABGR_PRE) {
		
		uint pixelIndex = (pixelY * width + pixelX) * 4;
		byteOutputImage[pixelIndex + 0] = getAlpha(color);
		byteOutputImage[pixelIndex + 1] = getBlue(color);
		byteOutputImage[pixelIndex + 2] = getRed(color);
		byteOutputImage[pixelIndex + 3] = getGreen(color);
		
	} else if (imageType == BUFFERED_IMAGE_TYPE_BYTE_GRAY) {
		
		uint pixelIndex = pixelY * width + pixelX;
		byteOutputImage[pixelIndex] = getRed(color);
		
	}
}