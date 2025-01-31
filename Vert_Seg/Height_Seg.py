import skimage as ski
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

# Read in tiff image
img1 = plt.imread(r'C:\Users\kfran\PycharmProjects\Scikit\Vert_Seg\Images\1924_6_slices1.tiff')
mask1 = plt.imread(r'C:\Users\kfran\PycharmProjects\Scikit\Vert_Seg\Images\1924_6_slices_mask1.tiff')
ar_1 = np.array(img1)
# fig1 = plt.hist(ar_1)
# fig2 = plt.imshow(ar_1)
el_map = ski.filters.sobel(ar_1, mask=mask1)
markers = np.zeros_like(ar_1)
markers[ar_1 < 33300] = 1
markers[ar_1 > 35590] = 2
plt.imshow(markers)
plt.colorbar()
plt.show()


