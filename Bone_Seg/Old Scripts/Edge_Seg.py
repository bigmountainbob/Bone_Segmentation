import skimage as ski
import matplotlib.pyplot as plt
import scipy as sp

img1 = plt.imread(r'C:\Users\kfran\OneDrive - University of Colorado Colorado Springs\Documents\School\Lab\PyCharm\Vert_Seg\Scripts\Images\1924_6_slices1.tiff',)
implot = plt.imshow(img1,cmap="gray")
plt.colorbar()
plt.show()

edges = ski.feature.canny(img1 / 100.)
fill_vert = sp.ndimage.binary_fill_holes(edges)

implot = plt.imshow(edges,cmap="gray")
plt.colorbar()
plt.show()

