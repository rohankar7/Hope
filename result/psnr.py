import cv2
import numpy as np

def calculate_psnr(original, reconstructed):
    # Convert images to float type for precision in calculations
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    # Compute Mean Squared Error (MSE) between the two images
    mse = np.mean((original - reconstructed) ** 2)
    # If MSE is zero, the images are identical; set PSNR to infinity
    if mse == 0:
        return float('inf')
    # Calculate PSNR using the correct formula
    PIXEL_MAX = 255.0  # Assuming 8-bit images
    psnr = 20 * np.log10((PIXEL_MAX) / np.sqrt(mse))
    return psnr

def main(): 
	# Load the original and reconstructed images
    original_image = cv2.imread('path_to_original_image.png')
    reconstructed_image = cv2.imread('path_to_reconstructed_image.png', 1)
    # Calculate PSNR
    psnr_value = calculate_psnr(original_image, reconstructed_image)
    print("PSNR value:", psnr_value)
	
if __name__ == "__main__": 
	main()