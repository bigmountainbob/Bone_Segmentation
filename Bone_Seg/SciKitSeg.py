import skimage as ski
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy import ndimage as ndi
import glob
import cv2
from skimage.filters import try_all_threshold


path_slice = glob.glob(r'C:\Users\kfran\PycharmProjects\Bone_Segmentation\Bone_Seg\Slices\*.tiff')
path_mask = glob.glob(r'C:\Users\kfran\PycharmProjects\Bone_Segmentation\Bone_Seg\Masks\*.tiff')

def lowest_nonzero(matrix):
    matrix = np.array(matrix)  # Convert to numpy array for easy processing
    nonzero_elements = matrix[matrix > 0]  # Filter out nonzero elements
    return np.min(nonzero_elements) if nonzero_elements.size > 0 else None  # Return min or None if empty


for i in range(len(path_slice)):
    img1 = cv2.imread(path_slice[i])
    mask = plt.imread(path_mask[i])
    mask = mask / 255
    masked_image = cv2.bitwise_and(img1,img1, mask=cv2.bitwise_not(mask))
    tr_1 = np.array(img1)

    ar_1 = mask*tr_1
    cf = lowest_nonzero(ar_1)
    ar_1 -= cf
    ar_1[ar_1 < 0] = 0

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

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## Greyscale Image Thresholding

    # Create histogram of intensity 0 = black
    # fig1 = plt.hist(ar_1)
    # fig = plt.imshow(ar_1)
    # plt.colorbar()
    # plt.show()
    # plt.clf

    # markers = np.zeros_like(ar_1)
    # markers[ar_1 < 33300] = 1  # Define dark threshold
    # markers[ar_1 > 35590] = 2  # Define light threshold
    # mask_th = np.zeros_like(markers)
    # mask_th[markers == 1] = 1
    # plt.imshow(mask_th, cmap='grey')
    # plt.colorbar()
    # plt.savefig('Threshold_' + str(i) + '.png')
    # plt.clf()
    # # plt.show()
    #
    # fig, ax = try_all_threshold(ar_1,figsize=(15,10),verbose=True)
    # plt.show()