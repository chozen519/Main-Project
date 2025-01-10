import cv2
import numpy as np
import os

def reduce_brightness_with_min_filter(image):
    """
    Reduce bright object intensity for each pixel by taking the minimum
    value in a 5x5 neighborhood of the pixel.
    
    Parameters:
        image (numpy.ndarray): Input RGB image as a numpy array.

    Returns:
        numpy.ndarray: Image with reduced bright object intensity.
    """
    kernel_size = 5  # 5x5 neighborhood
    reduced_brightness_image = cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8))
    return reduced_brightness_image

def quad_segmentation(image, gamma_threshold=0.2):
    """
    Perform quad-segmentation and choose the sub-block with the highest intensity.
    
    Parameters:
        image (numpy.ndarray): Input image as a numpy array.
        gamma_threshold (float): Segmentation ratio threshold.
    
    Returns:
        numpy.ndarray: Sub-block with the highest intensity after segmentation.
    """
    h, w, _ = image.shape
    size_ratio = h * w  # Original size

    while (h * w) / size_ratio > gamma_threshold:
        # Current sub-block dimensions
        h_sub, w_sub = h, w
        
        # Calculate the midpoints for segmentation
        h_mid, w_mid = h_sub // 2, w_sub // 2

        # Divide the image into 4 sub-blocks
        sub_blocks = [
            image[:h_mid, :w_mid],   # Top-left
            image[:h_mid, w_mid:],  # Top-right
            image[h_mid:, :w_mid],  # Bottom-left
            image[h_mid:, w_mid:]   # Bottom-right
        ]

        # Calculate the average brightness for each sub-block
        avg_brightness = [np.mean(block) for block in sub_blocks]

        # Choose the sub-block with the highest average brightness
        max_brightness_idx = np.argmax(avg_brightness)
        image = sub_blocks[max_brightness_idx]
        h, w = image.shape[:2]  # Update the dimensions of the new input

    return image

def calculate_pixel_value(image):
    """
    Calculate the required pixel value I_dash(r,g,b) = min sum((I_dash(x) - 1)^2).
    
    Parameters:
        image (numpy.ndarray): Input image as a numpy array.

    Returns:
        numpy.ndarray: Pixel values after calculation.
    """
    # Convert image to float for calculations
    image = image.astype(np.float32) / 255.0
    
    # Reference point in RGB space
    reference = np.array([1.0, 1.0, 1.0])

    # Calculate the Euclidean distance from each pixel to the reference point
    distances = np.sqrt(np.sum((image - reference) ** 2, axis=-1))

    # Find the pixel with the minimum distance
    min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)

    return image[min_distance_idx]

def calculate_atmospheric_light(pixel_value):
    """
    Obtain the atmospheric light A(r, g, b) = I_dash(r, g, b).
    
    Parameters:
        pixel_value (numpy.ndarray): Pixel value obtained from calculate_pixel_value.
    
    Returns:
        numpy.ndarray: Atmospheric light value.
    """
    return pixel_value

if __name__ == "__main__":
    input_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/Preprocessed"
    output_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/Atmospheric"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, "all_atmospheric_lights.txt")

    with open(output_file, "w") as f:  # Use "w" to overwrite if necessary
        for image_name in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)

            if image is not None:
                reduced_image = reduce_brightness_with_min_filter(image)
                segmented_image = quad_segmentation(reduced_image)
                required_pixel_value = calculate_pixel_value(segmented_image)
                atmospheric_light = calculate_atmospheric_light(required_pixel_value)
                print("Atmospheric Light (R, G, B):", atmospheric_light)
                f.write(f"Image: {image_name} - Atmospheric Light (R, G, B): {atmospheric_light.tolist()}\n")
            else:
                print(f"Error: Could not load image {image_name}. Skipping...")

    print(f"All atmospheric lights saved to {output_file}")

