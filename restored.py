import cv2
import numpy as np
import os

def read_atmospheric_light_from_file(filename, light_folder):
    """
    Read the atmospheric light value for RGB channels from the corresponding text file.

    Parameters:
        filename: The name of the image file.
        light_folder: Folder containing atmospheric light text files.

    Returns:
        A: A 3-element array representing the atmospheric light values for RGB.
    """
    # Construct the atmospheric light filename, assuming the convention "01_outdoor_hazy_light.txt"
    light_filename = os.path.splitext(filename)[0] + "_light.txt"  # Include .txt extension
    light_path = os.path.join(light_folder, light_filename)

    try:
        with open(light_path, 'r') as f:
            # Read the atmospheric light values (R, G, B) from the first line
            R, G, B = map(float, f.readline().strip().split())
        return np.array([R, G, B])  # Return as a NumPy array for easy processing
    except FileNotFoundError:
        print(f"Error: Atmospheric light file not found for {filename} at path {light_path}")
        return None
    except ValueError:
        print(f"Error: Incorrect format in atmospheric light file for {filename}")
        return None


def atmospheric_scattering_model(hazy_image, refined_transmission_map, A):
    """
    Apply the atmospheric scattering model to recover the scene radiance (clear image).

    Parameters:
        hazy_image: The input hazy image (I(x)) in float32 format.
        refined_transmission_map: The refined transmission map (t(x)).
        A: Estimated atmospheric light (A) as an array of RGB values.

    Returns:
        clear_image: The recovered clear image (J(x)).
    """
    # Ensure transmission map is not zero to avoid division by zero errors
    refined_transmission_map = np.clip(refined_transmission_map, 1e-6, 1.0)

    # Apply the atmospheric scattering model equation for each channel (RGB)
    clear_image = np.zeros_like(hazy_image)
    for i in range(3):  # Loop over each channel (R, G, B)
        clear_image[..., i] = (hazy_image[..., i] - A[i] * (1 - refined_transmission_map)) / refined_transmission_map

    return np.clip(clear_image, 0.0, 1.0)  # Ensure the result is within [0, 1] range

def main(input_folder, depth_output_folder, transmission_output_folder, light_folder, output_folder):
    """
    Process hazy images, use pre-calculated atmospheric light, and apply atmospheric scattering model.

    Parameters:
        input_folder: Folder containing hazy images.
        depth_output_folder: Folder to save refined depth maps.
        transmission_output_folder: Folder to save refined transmission maps.
        light_folder: Folder containing atmospheric light text files.
        output_folder: Folder to save the recovered clear images.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        print(f"Processing {filename}...")

        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):

            image_path = os.path.join(input_folder, filename)
            hazy_image = cv2.imread(image_path).astype(np.float32) / 255.0

            if hazy_image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Load the refined transmission map
            transmission_map_path = os.path.join(transmission_output_folder, f"{os.path.splitext(filename)[0]}_transmission.jpg")
            refined_transmission_map = cv2.imread(transmission_map_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

            if refined_transmission_map is None:
                print(f"Failed to load transmission map: {transmission_map_path}")
                continue

            # Read the pre-calculated atmospheric light from the file (RGB values)
            A = read_atmospheric_light_from_file(filename, light_folder)
            if A is None:
                print(f"Skipping image {filename} due to missing or invalid atmospheric light value.")
                continue

            # Apply the atmospheric scattering model to recover the clear image
            clear_image = atmospheric_scattering_model(hazy_image, refined_transmission_map, A)

            # Save the recovered clear image
            clear_image_output = (clear_image * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_clear.jpg"), clear_image_output)

            print(f"Processed and saved recovered clear image for {filename}")

# Input and output folders
input_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/Preprocessed"
depth_output_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/initial_depth_map"
transmission_output_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/refined_transmission"
light_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/Atmospheric"  # Folder with light values
output_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/recovered_clear_images"

main(input_folder, depth_output_folder, transmission_output_folder, light_folder, output_folder)
