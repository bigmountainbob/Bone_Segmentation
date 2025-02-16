import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import glob
from skimage import io, filters, img_as_ubyte
from scipy.ndimage import gaussian_filter

path_slice = glob.glob(r'C:\Users\kfran\PycharmProjects\Bone_Segmentation\Bone_Seg\Slices\*.tiff')
path_mask = glob.glob(r'C:\Users\kfran\PycharmProjects\Bone_Segmentation\Bone_Seg\Masks\*.tiff')

for i in range(len(path_slice)):
    # Load the grayscale image
    image = io.imread(path_slice[i], as_gray=True)
    image = img_as_ubyte(image)  # Convert to 8-bit format

    mask = io.imread(path_mask[i], as_gray=True)
    mask = img_as_ubyte(mask) # Convert to 8-bit


    # # Create a circular mask
    # mask = np.zeros_like(image, dtype=np.uint8)
    # cy, cx = image.shape[0] // 2, image.shape[1] // 2
    # radius = 500  # Adjust as needed
    # y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    # mask_area = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    # mask[mask_area] = 255  # White circular mask

    # 🔹 Step 1: Smooth the mask to prevent edge artifacts
    # mask_blurred = gaussian_filter(mask.astype(float), sigma=50) / 255.0  # Normalize to [0,1]

    # Extract the region inside the mask
    masked_image = image * mask  # Apply the mask
    nonzero_masked_pixels = masked_image[mask > 0]  # Extract nonzero pixels

    # mask_blurred = mask

    # 🔹 Step 2: Apply Multiple Thresholding Techniques
    threshold_methods = {
        "Otsu": filters.threshold_otsu(nonzero_masked_pixels),
        "Sauvola": filters.threshold_sauvola(masked_image, window_size=25),
        "Mean": filters.threshold_mean(nonzero_masked_pixels),
        "Yen": filters.threshold_yen(nonzero_masked_pixels),
        "Li": filters.threshold_li(nonzero_masked_pixels),
        "Triangle": filters.threshold_triangle(nonzero_masked_pixels),
        "ISODATA": filters.threshold_isodata(nonzero_masked_pixels),
        "Niblack": filters.threshold_niblack(masked_image, window_size=25, k=-0.2),
        "Phansalkar": filters.threshold_sauvola(masked_image, window_size=15, k=0.25, r=0.5),  # Phansalkar variation
        "Bernsen": filters.threshold_niblack(masked_image, window_size=35, k=0.0),  # Bernsen variation
        "Entropy": filters.threshold_li(nonzero_masked_pixels)  # Alternative entropy-based
    }

    # 🔹 Handle `threshold_minimum()` error safely
    try:
        threshold_methods["Minimum"] = filters.threshold_minimum(nonzero_masked_pixels)
    except RuntimeError:
        print("⚠️ Warning: `threshold_minimum()` failed! Using Otsu instead.")
        threshold_methods["Minimum"] = threshold_methods["Otsu"]  # Fallback to Otsu

    # Apply thresholds
    thresholded_images = {name: (masked_image > value) * 255 for name, value in threshold_methods.items()}
    thresholded_images["Sauvola"] = (masked_image > threshold_methods["Sauvola"]) * 255
    thresholded_images["Niblack"] = (masked_image > threshold_methods["Niblack"]) * 255
    thresholded_images["Phansalkar"] = (masked_image > threshold_methods["Phansalkar"]) * 255
    thresholded_images["Bernsen"] = (masked_image > threshold_methods["Bernsen"]) * 255

    # 🔹 Step 3: Smoothly Blend the Thresholded Images Back
    # def blend_threshold(original, thresholded, mask_blurred):
    #     return (original * (1 - mask_blurred) + thresholded * mask_blurred).astype(np.uint8)

    # blended_results = {name: blend_threshold(image, thres, mask_blurred) for name, thres in thresholded_images.items()}

    # 🔹 Step 4: Display Results
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))
    axes = axes.ravel()

    # Show original and mask
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    # axes[2].imshow(mask, cmap='gray')
    # axes[2].set_title("Smoothed Mask")

    # Add thresholded images to grid
    for i, (name, img) in enumerate(thresholded_images.items()):
        axes[i + 2].imshow(img, cmap='gray')
        axes[i + 2].set_title(name)

    plt.tight_layout()
    plt.show()

