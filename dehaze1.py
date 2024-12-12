import cv2
import numpy as np
import os

def calculate_detection_index(image):
    """Calculate the detection index to identify unbalanced color channels."""
    mean_r = np.mean(image[:, :, 2])  # Red channel
    mean_g = np.mean(image[:, :, 1])  # Green channel
    mean_b = np.mean(image[:, :, 0])  # Blue channel
    detection_index = abs(mean_r - mean_g) + abs(mean_g - mean_b) + abs(mean_b - mean_r)
    return detection_index

def color_balance(image):
    """Apply color balancing to the image."""
    image = image.astype(np.float32)
    channels = cv2.split(image)
    mean = [np.mean(channel) for channel in channels]
    std_dev = [np.std(channel) for channel in channels]
    
    balanced_channels = [
        (channel - mean_channel) / std_channel * std_dev[1] + mean[1]
        for channel, mean_channel, std_channel in zip(channels, mean, std_dev)
    ]
    
    balanced_image = cv2.merge(balanced_channels)
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)
    return balanced_image

def process_images(input_folder, output_folder, threshold=0.15):
    """Process images in the folder, applying color balance if detection index exceeds the threshold."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image = cv2.imread(file_path)
            
            if image is not None:
                # Calculate detection index
                detection_index = calculate_detection_index(image)
                print(f"Image: {filename}, Detection Index: {detection_index:.2f}")
                
                if detection_index > threshold:
                    # Apply color balancing
                    balanced_image = color_balance(image)
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, balanced_image)
                    print(f"Processed and saved: {output_path}")
                else:
                    print(f"No changes needed for: {filename}")
            else:
                print(f"Failed to load image: {file_path}")

# Define your input and output folders
input_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/InputImages"
output_folder = "C:/Users/green/Desktop/Main project/dataset/IHAZE/Preprocessed1"

# Process images
process_images(input_folder, output_folder)
