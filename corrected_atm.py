import cv2
import numpy as np
import os

def reduce_brightness_with_min_filter(image):
    kernel_size = 5  # 5x5 neighborhood
    reduced_brightness_image = cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8))
    return reduced_brightness_image

def quad_segmentation(image, gamma_threshold=0.2):
    h, w, _ = image.shape
    size_ratio = h * w

    while (h * w) / size_ratio > gamma_threshold:
        h_sub, w_sub = h, w
        h_mid, w_mid = h_sub // 2, w_sub // 2

        sub_blocks = [
            image[:h_mid, :w_mid],   # Top-left
            image[:h_mid, w_mid:],   # Top-right
            image[h_mid:, :w_mid],   # Bottom-left
            image[h_mid:, w_mid:]    # Bottom-right
        ]

        avg_brightness = [np.mean(block) for block in sub_blocks]
        max_brightness_idx = np.argmax(avg_brightness)
        image = sub_blocks[max_brightness_idx]
        h, w = image.shape[:2]

    return image

def calculate_pixel_value(image):
    image = image.astype(np.float32) / 255.0
    reference = np.array([1.0, 1.0, 1.0])
    distances = np.sqrt(np.sum((image - reference) ** 2, axis=-1))
    min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return image[min_distance_idx]

def calculate_atmospheric_light(pixel_value):
    return pixel_value

if __name__ == "__main__":
    input_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/Preprocessed"
    output_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/Atmospheric"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        if image is not None:
            reduced_image = reduce_brightness_with_min_filter(image)
            segmented_image = quad_segmentation(reduced_image)
            required_pixel_value = calculate_pixel_value(segmented_image)
            atmospheric_light = calculate_atmospheric_light(required_pixel_value)
            print("Atmospheric Light (R, G, B):", atmospheric_light)

            # Create a filename for the atmospheric light text file
            light_filename = os.path.splitext(image_name)[0] + "_light.txt"
            light_file_path = os.path.join(output_folder, light_filename)

            # Save the atmospheric light values to a separate text file
            with open(light_file_path, "w") as light_file:
                light_file.write(f"{atmospheric_light[0]} {atmospheric_light[1]} {atmospheric_light[2]}\n")
        else:
            print(f"Error: Could not load image {image_name}. Skipping...")

    print("Atmospheric light values saved for each image.")
