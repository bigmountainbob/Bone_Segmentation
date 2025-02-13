import skimage as ski
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy import ndimage as ndi
import glob

path_slice = glob.glob(r'C:\Users\kfran\PycharmProjects\Bone_Segmentation\Bone_Seg\Slices\*.tiff')
path_mask = glob.glob(r'C:\Users\kfran\PycharmProjects\Bone_Segmentation\Bone_Seg\Masks\*.tiff')


def lowest_nonzero(matrix):
    matrix = np.array(matrix)  # Convert to numpy array for easy processing
    nonzero_elements = matrix[matrix > 0]  # Filter out nonzero elements
    return np.min(nonzero_elements) if nonzero_elements.size > 0 else None  # Return min or None if empty


for i in range(len(path_slice)):
    img1 = plt.imread(path_slice[i])
    mask = plt.imread(path_mask[i])
    ar_1 = np.array(img1)
    # Create histogram of intensity 0 = black
    # fig1 = plt.hist(ar_1)
    # fig2 = plt.imshow(ar_1)
    # Apply Sobel filter with mask
    el_map = ski.filters.sobel(ar_1, mask=mask)
    can_map = ski.feature.canny(ar_1,mask = mask)
    # Create markers to distinguish from background
    markers = np.zeros_like(ar_1)
    # Edit values to remove background
    markers[ar_1 < 33300] = 1  # Define dark threshold
    markers[ar_1 > 35590] = 2  # Define light threshold

    # segmented_bone = ski.segmentation.watershed(el_map,markers) # Lose detail with this
    # segmented_bone = ndi.binary_fill_holes(markers -1)

    mask = np.zeros_like(markers)
    mask[markers == 1] = 1

    plt.imshow(mask, cmap='grey')
    plt.colorbar()
    plt.savefig('Sobel_' + str(i) + '.png')
    plt.clf()
    # plt.show()

