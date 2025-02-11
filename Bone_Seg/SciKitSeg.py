import skimage as ski
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy import ndimage as ndi
import glob

from skimage.filters import try_all_threshold

path_slice = glob.glob(r'C:\Users\kfran\PycharmProjects\Bone_Segmentation\Bone_Seg\Slices\*.tiff')
path_mask = glob.glob(r'C:\Users\kfran\PycharmProjects\Bone_Segmentation\Bone_Seg\Masks\*.tiff')


for i in range(len(path_slice)):
    img1 = plt.imread(path_slice[i])
    mask = plt.imread(path_mask[i])
    ar_1 = np.array(img1)
    mask = mask/255
    ar_1 = mask*ar_1
    # corfac = min(ar_1)

    ## Greyscale Image Thresholding

    # Create histogram of intensity 0 = black
    # fig1 = plt.hist(ar_1)
    # fig = plt.imshow(ar_1)
    # plt.show()
    # plt.clf

    markers = np.zeros_like(ar_1)
    markers[ar_1 < 33300] = 1  # Define dark threshold
    markers[ar_1 > 35590] = 2  # Define light threshold
    mask_th = np.zeros_like(markers)
    mask_th[markers == 1] = 1
    plt.imshow(mask_th, cmap='grey')
    plt.colorbar()
    plt.savefig('Threshold_' + str(i) + '.png')
    plt.clf()
    # plt.show()

    fig, ax = try_all_threshold(ar_1,figsize=(15,10),verbose=True)
    plt.show()