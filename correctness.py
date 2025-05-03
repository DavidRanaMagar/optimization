import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse


# Install required packages
# pip install opencv-python numpy matplotlib tqdm

# To run
# python3 correctness.py
def compare_image_directories(cpu_dir, gpu_dir, output_dir=None, threshold=30, save_threshold=1.0):
    """
    Compare all images between CPU and GPU output directories

    Args:
        cpu_dir: Path to CPU output directory
        gpu_dir: Path to GPU output directory
        output_dir: Where to save difference visualizations (None to skip saving)
        threshold: Pixel difference threshold (0-255)
        save_threshold: Only save images with difference > this percentage
    """
    results = {
        'identical_count': 0,
        'different_count': 0,
        'max_difference': 0,
        'min_difference': 100,
        'avg_difference': 0,
        'details': {}
    }

    # Get list of image files
    cpu_files = sorted([f for f in os.listdir(cpu_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gpu_files = sorted([f for f in os.listdir(gpu_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not cpu_files or not gpu_files:
        raise ValueError("No images found in one or both directories")

    if len(cpu_files) != len(gpu_files):
        print(f"Warning: Different number of images (CPU: {len(cpu_files)}, GPU: {len(gpu_files)})")

    # Create output directory if specified and needed
    differences = []

    # Compare each pair of images
    for filename in tqdm(cpu_files, desc="Comparing images"):
        if filename not in gpu_files:
            print(f"Skipping {filename} - not found in GPU directory")
            continue

        cpu_path = os.path.join(cpu_dir, filename)
        gpu_path = os.path.join(gpu_dir, filename)

        try:
            identical, percent_diff, diff_img = compare_images(
                cpu_path, gpu_path, None, threshold  # Don't save initially
            )

            # Only prepare to save if difference exceeds threshold
            output_path = None
            if output_dir and percent_diff > save_threshold:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, f"diff_{filename}")
                # Re-run comparison to save the diff image
                identical, percent_diff, diff_img = compare_images(
                    cpu_path, gpu_path, output_path, threshold
                )

            results['details'][filename] = {
                'identical': identical,
                'difference_percent': percent_diff,
                'diff_image_path': output_path if output_path else None
            }

            differences.append(percent_diff)

            if identical:
                results['identical_count'] += 1
            else:
                results['different_count'] += 1

        except Exception as e:
            print(f"Error comparing {filename}: {str(e)}")
            results['details'][filename] = {'error': str(e)}

    # Calculate summary statistics
    if differences:
        results['max_difference'] = max(differences)
        results['min_difference'] = min(differences)
        results['avg_difference'] = sum(differences) / len(differences)

    return results

def compare_images(img1_path, img2_path, output_diff_path=None, threshold=30):
    """Compare two images and return difference metrics"""
    # Read images
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        raise ValueError("Could not read one or both images")

    # Convert to grayscale if they're color images
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Check if dimensions match
    if img1.shape != img2.shape:
        # Resize the larger image to match the smaller one
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        new_h = min(h1, h2)
        new_w = min(w1, w2)
        img1 = cv2.resize(img1, (new_w, new_h))
        img2 = cv2.resize(img2, (new_w, new_h))

    # Compute absolute difference
    diff = cv2.absdiff(img1, img2)

    # Threshold the difference to get a binary mask of differences
    _, threshold_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Calculate difference percentage
    total_pixels = diff.size
    changed_pixels = np.count_nonzero(threshold_diff)
    difference_percentage = (changed_pixels / total_pixels) * 100

    # Create visualization if requested
    if output_diff_path:
        if len(img1.shape) == 2:  # If grayscale, convert to BGR for colored output
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Create a red mask for differences
        diff_mask = cv2.cvtColor(threshold_diff, cv2.COLOR_GRAY2BGR)
        diff_mask[:, :, 0:2] = 0  # Zero out green and blue channels (keep only red)

        # Combine images with differences highlighted
        combined = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        combined = cv2.addWeighted(combined, 0.7, diff_mask, 0.3, 0)
        cv2.imwrite(output_diff_path, combined)

    # Determine if images are identical (with small tolerance)
    are_identical = difference_percentage < 0  # 0% difference threshold

    return are_identical, difference_percentage, diff

def print_comparison_summary(results):
    """Print a formatted summary of the comparison results"""
    print("\n=== Comparison Summary ===")
    print(f"Identical images: {results['identical_count']}")
    print(f"Different images: {results['different_count']}")
    print(f"Maximum difference: {results['max_difference']:.2f}%")
    print(f"Minimum difference: {results['min_difference']:.2f}%")
    print(f"Average difference: {results['avg_difference']:.2f}%")

    # Print details for significantly different images
    print("\nSignificantly different images (>0% difference):")
    for filename, data in results['details'].items():
        if 'difference_percent' in data and data['difference_percent'] > 1:
            diff_info = f" (saved to {data['diff_image_path']}" if data['diff_image_path'] else ""
            print(f"- {filename}: {data['difference_percent']:.2f}% difference{diff_info}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare CPU and GPU output images')
    parser.add_argument('--cpu_dir', default='data/cpu_output', help='CPU output directory')
    parser.add_argument('--gpu_dir', default='data/gpu_output', help='GPU output directory')
    parser.add_argument('--output_dir', default='data/differences', help='Directory to save difference images')
    parser.add_argument('--threshold', type=int, default=30, help='Pixel difference threshold (0-255)')
    parser.add_argument('--save_threshold', type=float, default=1.0,
                        help='Only save images with difference > this percentage')
    args = parser.parse_args()

    # Compare all images
    results = compare_image_directories(
        args.cpu_dir,
        args.gpu_dir,
        args.output_dir,
        args.threshold,
        args.save_threshold
    )

    # Print summary
    print_comparison_summary(results)

    # Optionally visualize some differences
    if results['different_count'] > 0:
        sample_diff = next((k for k,v in results['details'].items()
                            if 'difference_percent' in v and v['difference_percent'] > 0), None)

        if sample_diff and results['details'][sample_diff]['diff_image_path']:
            print("\nVisualizing sample difference...")
            diff_img = cv2.imread(results['details'][sample_diff]['diff_image_path'])

            plt.figure(figsize=(10, 5))
            plt.imshow(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Sample Difference Visualization\n({sample_diff}, {results['details'][sample_diff]['difference_percent']:.2f}% different)")
            plt.axis('off')
            plt.show()