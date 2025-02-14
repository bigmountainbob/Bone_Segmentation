import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, img_as_ubyte
from scipy.ndimage import gaussian_filter

# Load the grayscale image
image = io.imread(r"C:\Users\kfran\PycharmProjects\Bone_Segmentation\Bone_Seg\Images\1924_6_slices1.tiff", as_gray=True)
image = img_as_ubyte(image)  # Convert to 8-bit format

# 🔹 Step 1: Threshold the image to create a binary mask
threshold_value = filters.threshold_otsu(image)
binary_mask = image > threshold_value  # Create a binary mask based on the threshold

# 🔹 Step 2: Label connected components
labeled_mask, num_labels = measure.label(binary_mask, connectivity=2, return_num=True)

# 🔹 Step 3: Find the largest and second-largest connected components
props = measure.regionprops(labeled_mask)

# Sort regions by area (largest to smallest) and select the second-largest
sorted_regions = sorted(props, key=lambda x: x.area, reverse=True)

# Ensure there are at least two regions
if len(sorted_regions) > 1:
    second_largest_region = sorted_regions[1]  # Second largest region
else:
    print("⚠️ Only one region found. Using the only available region.")
    second_largest_region = sorted_regions[0]  # Use the largest if only one region exists

# Create a new mask for the second-largest region
second_largest_mask = np.zeros_like(binary_mask)
second_largest_mask[labeled_mask == second_largest_region.label] = 255  # Keep the second-largest object

# 🔹 Step 4: Apply the second-largest mask to the image (focus on the second-largest object)
masked_image = image * second_largest_mask

# 🔹 Step 5: Smooth the mask to prevent edge artifacts
mask_blurred = gaussian_filter(second_largest_mask.astype(float), sigma=10) / 255.0  # Normalize to [0,1]

# 🔹 Step 6: Apply Multiple Thresholding Techniques
nonzero_masked_pixels = masked_image[second_largest_mask > 0]  # Only pixels inside the second-largest mask

threshold_methods = {
    "Otsu": filters.threshold_otsu(nonzero_masked_pixels),
    "Sauvola": filters.threshold_sauvola(masked_image, window_size=25),
    "Mean": filters.threshold_mean(nonzero_masked_pixels),
    "Yen": filters.threshold_yen(nonzero_masked_pixels),
    "Li": filters.threshold_li(nonzero_masked_pixels),
    "Triangle": filters.threshold_triangle(nonzero_masked_pixels),
    "ISODATA": filters.threshold_isodata(nonzero_masked_pixels),
    "Niblack": filters.threshold_niblack(masked_image, window_size=25, k=-0.2),
    "Phansalkar": filters.threshold_sauvola(masked_image, window_size=15, k=0.25, r=0.5),
    "Bernsen": filters.threshold_niblack(masked_image, window_size=35, k=0.0),
    "Entropy": filters.threshold_li(nonzero_masked_pixels)
}

# Apply thresholds
thresholded_images = {name: (masked_image > value) * 255 for name, value in threshold_methods.items()}
thresholded_images["Sauvola"] = (masked_image > threshold_methods["Sauvola"]) * 255
thresholded_images["Niblack"] = (masked_image > threshold_methods["Niblack"]) * 255
thresholded_images["Phansalkar"] = (masked_image > threshold_methods["Phansalkar"]) * 255
thresholded_images["Bernsen"] = (masked_image > threshold_methods["Bernsen"]) * 255

# 🔹 Step 7: Smoothly Blend the Thresholded Images Back
def blend_threshold(original, thresholded, mask_blurred):
    return (original * (1 - mask_blurred) + thresholded * mask_blurred).astype(np.uint8)

blended_results = {name: blend_threshold(image, thres, mask_blurred) for name, thres in thresholded_images.items()}

# 🔹 Step 8: Display Results
fig, axes = plt.subplots(4, 4, figsize=(15, 12))
axes = axes.ravel()

# Show original and mask
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[1].imshow(binary_mask, cmap='gray')
axes[1].set_title("Binary Mask (Otsu)")
axes[2].imshow(second_largest_mask, cmap='gray')
axes[2].set_title("Second Largest Object Mask")

# Add thresholded images to grid
for i, (name, img) in enumerate(blended_results.items()):
    axes[i + 3].imshow(img, cmap='gray')
    axes[i + 3].set_title(name)

plt.tight_layout()
plt.show()
