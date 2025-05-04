import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import cv2

def compare_image_directories(cpu_dir, gpu_dir, output_dir=None, threshold=30, save_threshold=1.0):
    """
    Compare all images between CPU and GPU output directories
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
    cpu_files = sorted([f for f in os.listdir(cpu_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gpu_files = sorted([f for f in os.listdir(gpu_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"\n--- Comparing CPU ▶ {cpu_dir}\n       vs GPU ▶ {gpu_dir} ---")
    print("CPU contents:", os.listdir(cpu_dir))
    print("PNG/JPG found in CPU:", cpu_files)
    print("PNG/JPG found in GPU:", gpu_files)

    if not cpu_files or not gpu_files:
        raise ValueError("No images found in one or both directories")

    if len(cpu_files) != len(gpu_files):
        print(f"Warning: Different number of images "
              f"(CPU: {len(cpu_files)}, GPU: {len(gpu_files)})")

    differences = []

    for filename in tqdm(cpu_files, desc="Comparing images"):
        if filename not in gpu_files:
            print(f"Skipping {filename} — not found in GPU directory")
            continue

        cpu_path = os.path.join(cpu_dir, filename)
        gpu_path = os.path.join(gpu_dir, filename)

        try:
            identical, percent_diff, _ = compare_images(
                cpu_path, gpu_path, None, threshold
            )

            # Nếu vượt qua save_threshold thì lưu ảnh diff
            diff_path = None
            if output_dir and percent_diff > save_threshold:
                os.makedirs(output_dir, exist_ok=True)
                diff_path = os.path.join(output_dir, f"diff_{filename}")
                identical, percent_diff, _ = compare_images(
                    cpu_path, gpu_path, diff_path, threshold
                )

            results['details'][filename] = {
                'identical': identical,
                'difference_percent': percent_diff,
                'diff_image_path': diff_path
            }
            differences.append(percent_diff)

            if identical:
                results['identical_count'] += 1
            else:
                results['different_count'] += 1

        except Exception as e:
            print(f"Error comparing {filename}: {e}")
            results['details'][filename] = {'error': str(e)}

    if differences:
        results['max_difference'] = max(differences)
        results['min_difference'] = min(differences)
        results['avg_difference'] = sum(differences) / len(differences)

    return results

def compare_images(img1_path, img2_path, output_diff_path=None, threshold=30):
    """Compare two images and return (are_identical, percent_diff, raw_diff)"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
    if img1 is None or img2 is None:
        raise ValueError("Could not read one or both images")

    # Grayscale
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize nếu kích thước khác
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

    diff = cv2.absdiff(img1, img2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    total = diff.size
    changed = np.count_nonzero(thresh)
    pct = (changed / total) * 100

    if output_diff_path:
        # convert to BGR for mask overlay
        base1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        base2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        mask[:, :, :2] = 0  # chỉ giữ channel đỏ
        combo = cv2.addWeighted(base1, 0.5, base2, 0.5, 0)
        combo = cv2.addWeighted(combo, 0.7, mask, 0.3, 0)
        cv2.imwrite(output_diff_path, combo)

    # Sửa lại logic: identical khi %dif == 0
    return (pct == 0), pct, diff

def print_comparison_summary(results):
    print("\n=== Comparison Summary ===")
    print(f"Identical images: {results['identical_count']}")
    print(f"Different images: {results['different_count']}")
    print(f"Max difference: {results['max_difference']:.2f}%")
    print(f"Min difference: {results['min_difference']:.2f}%")
    print(f"Avg difference: {results['avg_difference']:.2f}%")
    print("\nDetails (>% difference):")
    for fn, d in results['details'].items():
        if d.get('difference_percent', 0) > 1:
            loc = d.get('diff_image_path') or ''
            print(f"- {fn}: {d['difference_percent']:.2f}%  {loc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare CPU vs multiple GPU output images'
    )
    parser.add_argument(
        '--cpu_dir',
        default=r'D:\cosc4397\optimization\data\cpu_output',
        help='CPU output directory'
    )
    parser.add_argument(
        '--gpu_dirs',
        nargs='+',
        default=[
            # r'D:\cosc4397\optimization\data\gpu_naive_output'
             r'D:\cosc4397\optimization\data\gpu_optimal_output'
        ],
        help='List of GPU output directories'
    )
    parser.add_argument(
        '--output_dir',
        default='data/differences_all',
        help='Base directory to save all diff images'
    )
    parser.add_argument(
        '--threshold', type=int, default=30,
        help='Pixel difference threshold (0-255)'
    )
    parser.add_argument(
        '--save_threshold', type=float, default=1.0,
        help='Only save diffs > this %'
    )
    args = parser.parse_args()

    for gpu_dir in args.gpu_dirs:
        # tạo thư mục con cho mỗi GPU folder
        gpu_name = os.path.basename(gpu_dir.rstrip(r'\/'))
        out_subdir = os.path.join(args.output_dir, gpu_name)
        res = compare_image_directories(
            args.cpu_dir,
            gpu_dir,
            output_dir=out_subdir,
            threshold=args.threshold,
            save_threshold=args.save_threshold
        )
        print(f"\n=== CPU vs {gpu_name} ===")
        print_comparison_summary(res)

        # nếu bạn muốn show ảnh sample dùng matplotlib, bỏ comment block sau
        if res['different_count'] > 0:
            sample = next((k for k,v in res['details'].items()
                           if v.get('difference_percent',0)>0), None)
            if sample:
                img = cv2.imread(res['details'][sample]['diff_image_path'])
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f"{sample} ({res['details'][sample]['difference_percent']:.2f}%)")
                plt.axis('off')
                plt.show()
