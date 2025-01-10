import cv2
import numpy as np
import os

def gradient_domain_guided_filter(I, p, r, eps):
    try:
        """
        Gradient Domain Guided Filter.

        Parameters:
            I: Guide image (grayscale).
            p: Input depth map or image to refine.
            r: Radius of the filter.
            eps: Regularization parameter.

        Returns:
            q: Refined depth map or image.
        """
        mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
        mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))

        var_I = mean_II - mean_I * mean_I
        cov_Ip = mean_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

        grad_I_x = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
        grad_I_y = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)
        grad_p_x = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
        grad_p_y = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)

        grad_guidance = grad_I_x * grad_p_x + grad_I_y * grad_p_y
        refined_a = mean_a + 0.1 * grad_guidance  # Weighted gradient refinement

        q = refined_a * I + mean_b
        return q
    except Exception as e:
        print(f"Error in gradient_domain_guided_filter: {e}")
        return None

def estimate_initial_depth(image_hsv):
    try:
        """Estimate initial depth map using HSV value channel inversion."""
        return 1 - image_hsv[:, :, 2]
    except Exception as e:
        print(f"Error in estimate_initial_depth: {e}")
        return None

def apply_minimum_filter(depth_map, filter_size):
    try:
        """Apply minimum filter for local refinement."""
        return cv2.erode(depth_map, np.ones((filter_size, filter_size), np.uint8))
    except Exception as e:
        print(f"Error in apply_minimum_filter: {e}")
        return None

def weighted_fusion(depth_map_small, depth_map_large, alpha_small=0.6, alpha_large=0.4):
    try:
        """Weighted fusion of depth maps with balance factors."""
        return alpha_small * depth_map_small + alpha_large * depth_map_large
    except Exception as e:
        print(f"Error in weighted_fusion: {e}")
        return None

def calculate_transmission_map(depth_map, beta=1.2):
    try:
        """Estimate transmission map from depth map."""
        return np.exp(-beta * depth_map)
    except Exception as e:
        print(f"Error in calculate_transmission_map: {e}")
        return None

def estimate_initial_transmission_map(image_hsv):
    try:
        """
        Estimate initial transmission map using weighted fusion of small and large depth maps.

        Parameters:
            image_hsv: Input image in HSV format.

        Returns:
            Initial transmission map.
        """
        initial_depth = estimate_initial_depth(image_hsv)
        if initial_depth is None:
            return None

        depth_map_small = apply_minimum_filter(initial_depth, filter_size=3)
        depth_map_large = apply_minimum_filter(initial_depth, filter_size=15)
        if depth_map_small is None or depth_map_large is None:
            return None

        fused_depth_map = weighted_fusion(depth_map_small, depth_map_large)
        if fused_depth_map is None:
            return None

        return calculate_transmission_map(fused_depth_map)
    except Exception as e:
        print(f"Error in estimate_initial_transmission_map: {e}")
        return None

def main(input_folder, depth_output_folder, transmission_output_folder):
    try:
        """
        Process images to refine depth maps and estimate transmission maps.

        Parameters:
            input_folder: Folder containing hazy images.
            depth_output_folder: Folder to save refined depth maps.
            transmission_output_folder: Folder to save initial transmission maps.
        """
        os.makedirs(depth_output_folder, exist_ok=True)
        os.makedirs(transmission_output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            print(f"Processing {filename}...")
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path).astype(np.float32) / 255.0

                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                image_hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

                initial_transmission_map = estimate_initial_transmission_map(image_hsv)
                if initial_transmission_map is None:
                    print(f"Failed to estimate initial transmission map for {filename}")
                    continue

                grayscale_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

                refined_depth_map = gradient_domain_guided_filter(grayscale_image, initial_transmission_map, r=15, eps=1e-3)
                if refined_depth_map is None:
                    print(f"Failed to refine depth map for {filename}")
                    continue

                refined_depth_output = (refined_depth_map * 255).clip(0, 255).astype(np.uint8)
                transmission_output = (initial_transmission_map * 255).clip(0, 255).astype(np.uint8)

                cv2.imwrite(os.path.join(depth_output_folder, f"{os.path.splitext(filename)[0]}_depth.jpg"), refined_depth_output)
                cv2.imwrite(os.path.join(transmission_output_folder, f"{os.path.splitext(filename)[0]}_transmission.jpg"), transmission_output)

                print(f"Processed and saved results for {filename}")
    except Exception as e:
        print(f"Error in main function: {e}")

# Input and output folders
input_folder = "C:/Users/green/Desktop/Main project/dataset/OHAZE/Preprocessed"
depth_output_folder = "C:/Users/green/Desktop/Main project/dataset/OHAZE/initial_depth_map"
transmission_output_folder = "C:/Users/green/Desktop/Main project/dataset/OHAZE/refined_transmission"

main(input_folder, depth_output_folder, transmission_output_folder)