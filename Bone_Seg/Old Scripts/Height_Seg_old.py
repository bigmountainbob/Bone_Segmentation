import skimage as ski
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

img1 = plt.imread(r'C:\Users\kfran\PycharmProjects\Scikit\Vert_Seg\Images\1924_6_slices1.tiff',)
implot = plt.imshow(img1,cmap="gray")
# plt.colorbar()
# plt.show()

el_1 = ski.filters.sobel(img1)
el_plot = plt.imshow(el_1,cmap="gray")
# plt.colorbar()
# plt.show()
ar_1 = np.array(img1)
print(ar_1)

hist, hist_centers = ski.exposure.histogram(img1)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(img1, cmap=plt.cm.gray)
axes[0].set_axis_off()
axes[1].plot(hist_centers, hist, lw=2)
plt.show()

# markers = np.zeros_like(img1)
# markers[img1 < 125] = 1
# markers[img1 > 138] = 2
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(markers, cmap=plt.cm.nipy_spectral)
# ax.set_title('markers')
# ax.set_axis_off()

# plt.show()