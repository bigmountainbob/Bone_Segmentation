import skimage as ski
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy import ndimage as ndi

# Read in 16 bit tiff image and mask
img1 = plt.imread(r'C:\Users\kfran\PycharmProjects\Vertebrae_Segmentation\Vert_Seg\Images\1924_6_slices1.tiff')
mask1 = plt.imread(r'C:\Users\kfran\PycharmProjects\Vertebrae_Segmentation\Vert_Seg\Images\1924_6_slices_mask1.tiff')
# Convert to numpyarray
ar_1 = np.array(img1)
# Create histogram of intensity 0 = black
# fig1 = plt.hist(ar_1)
# plt.show()
# fig2 = plt.imshow(ar_1)
# Apply Sobel filter with mask
el_map = ski.filters.sobel(ar_1, mask=mask1)
# Create markers to distinguish from background
markers = np.zeros_like(ar_1)
# Edit values to remove background
markers[ar_1 < 33300] = 1 # Define dark threshold
markers[ar_1 > 35590] = 2 # Define light threshold

# segmented_bone = ski.segmentation.watershed(el_map,markers) # Lose detail with this
# segmented_bone = ndi.binary_fill_holes(markers -1)

mask = np.zeros_like(markers)
mask[markers == 1 ] = 1

plt.imshow(mask)
plt.colorbar()
plt.show()


