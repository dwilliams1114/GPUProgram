package examples;

import java.awt.image.BufferedImage;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import main.GPUAccess;
import main.GPUProgram;

/**
 * This example renders the Mandelbrot Set onto a BufferedImage.
 * Each pixel is processed in parallel on the GPU.
 * It works for almost any BufferedImage type.
 * This program requires "Mandelbrot.cl".
 * 
 * Programmed by Daniel Williams in 2025.
 */

public class Mandelbrot {
	
	public static void main(String[] args) {
		
		final int WIDTH = 1000;
		final int HEIGHT = 800;
		final float MIN_X = -1.9f;
		final float MAX_X = 0.6f;
		final float MIN_Y = -1.0f;
		final float MAX_Y = 1.0f;
		final int MAX_ITERATIONS = 80;
		final int BUFFERED_IMAGE_TYPE = BufferedImage.TYPE_3BYTE_BGR;
		final BufferedImage image = new BufferedImage(WIDTH, HEIGHT, BUFFERED_IMAGE_TYPE);
		
		// Load the OpenCL program
		GPUProgram renderMandelbrot = new GPUProgram("renderMandelbrot", "src/examples/Mandelbrot.cl");
		
		// Set the iteration bounds and process each pixel in a separate thread
		renderMandelbrot.setGlobalWorkGroupSizes(WIDTH, HEIGHT);
		
		// Set the arguments to the 'renderMandelbrot' function in Mandelbrot.cl
		renderMandelbrot.setArgument(0, image, GPUAccess.WRITE);
		renderMandelbrot.setArgument(1, WIDTH, GPUAccess.READ);
		renderMandelbrot.setArgument(2, HEIGHT, GPUAccess.READ);
		renderMandelbrot.setArgument(3, MIN_X, GPUAccess.READ);
		renderMandelbrot.setArgument(4, MIN_Y, GPUAccess.READ);
		renderMandelbrot.setArgument(5, MAX_X, GPUAccess.READ);
		renderMandelbrot.setArgument(6, MAX_Y, GPUAccess.READ);
		renderMandelbrot.setArgument(7, MAX_ITERATIONS, GPUAccess.READ);
		renderMandelbrot.setArgument(8, BUFFERED_IMAGE_TYPE, GPUAccess.READ);
		
		// Time the execution
		long startTime = System.nanoTime();
		renderMandelbrot.executeKernel();
		long deltaTime = System.nanoTime() - startTime;
		
		System.out.println("Rendered in " + deltaTime / 1e6d + " milliseconds");
		
		// Display the image in a window
		displayImage(image);
	}
	
	// Display the given image in a window
	private static void displayImage(BufferedImage image) {
		JFrame window = new JFrame("Mandelbrot");
		JLabel renderLabel = new JLabel(new ImageIcon(image));
		window.add(renderLabel);
		window.setResizable(false);
		window.pack();
		window.setLocationRelativeTo(null);
		window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		window.setVisible(true);
	}
	
}
