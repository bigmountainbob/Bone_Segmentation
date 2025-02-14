import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r"C:\Users\kfran\PycharmProjects\Bone_Segmentation\Bone_Seg\Images\1924_6_slices1.tiff", cv2.IMREAD_GRAYSCALE)

# Create a mask (example: exclude a central circular region)
mask = np.zeros_like(image, dtype=np.uint8)
cv2.circle(mask, (image.shape[1]//2, image.shape[0]//2), 500, 255, -1)  # White circle as mask

kernel = np.ones((5,5), np.uint8)
mask_dilated = cv2.dilate(mask, kernel, iterations=1)

# Apply thresholding only to unmasked regions
masked_image = cv2.bitwise_and(image, image, mask=mask_dilated)

plt.imshow(masked_image)
plt.show()

plt.hist(masked_image)
plt.show()

# Apply different thresholding techniques
_, binary_thresh = cv2.threshold(masked_image, 127, 255, cv2.THRESH_BINARY)
_, binary_inv_thresh = cv2.threshold(masked_image, 127, 255, cv2.THRESH_BINARY_INV)
_, trunc_thresh = cv2.threshold(masked_image, 127, 255, cv2.THRESH_TRUNC)
_, tozero_thresh = cv2.threshold(masked_image, 127, 255, cv2.THRESH_TOZERO)
_, tozero_inv_thresh = cv2.threshold(masked_image, 127, 255, cv2.THRESH_TOZERO_INV)
_, otsu_thresh = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive_thresh = cv2.adaptiveThreshold(masked_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Restore the masked region to the output
binary_thresh = cv2.bitwise_or(binary_thresh, mask)
binary_inv_thresh = cv2.bitwise_or(binary_inv_thresh, mask)
trunc_thresh = cv2.bitwise_or(trunc_thresh, mask)
tozero_thresh = cv2.bitwise_or(tozero_thresh, mask)
tozero_inv_thresh = cv2.bitwise_or(tozero_inv_thresh, mask)
otsu_thresh = cv2.bitwise_or(otsu_thresh, mask)
adaptive_thresh = cv2.bitwise_or(adaptive_thresh, mask)

# Display results
cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Binary Thresholding", binary_thresh)
cv2.imshow("Binary Inverse Thresholding", binary_inv_thresh)
cv2.imshow("Truncation Thresholding", trunc_thresh)
cv2.imshow("ToZero Thresholding", tozero_thresh)
cv2.imshow("ToZero Inverse Thresholding", tozero_inv_thresh)
cv2.imshow("Otsu Thresholding", otsu_thresh)
cv2.imshow("Adaptive Thresholding", adaptive_thresh)

cv2.waitKey(10000)
cv2.destroyAllWindows()
