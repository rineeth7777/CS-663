import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.stats import multivariate_normal
from scipy.spatial import distance
from PIL import Image
from math import exp


patch_size = 9
window_size = 25
h = 1


def getPatch(img,y,x):
	patch = np.empty((9,9))
	patch[:] = np.NaN

	shape = img.shape
	# print(shape)
	# print(x,y)

	leftPad = max(0,4-x)
	rightPad = max(0,x-shape[1]+5)
	topPad = max(0,4-y)
	bottomPad = max(0,y-shape[0]+5)

	left = max(0,x-4)
	right = min(shape[1]-1,x+4)
	top = max(0,y-4)
	bottom = min(shape[0]-1,y+4)

	# print(top,bottom,left,right)

	patch[topPad:8-bottomPad,leftPad:8-rightPad] = np.copy(img[top:bottom,left:right])
	return patch



path = "../data/grassNoisy.mat"

dic = loadmat(path)
img = dic["imgCorrupt"]


rv = multivariate_normal(mean=[4,4],cov=7.5)
Gaussian_mask = []

for i in range(9):
	Gaussian_mask.append([])
	for j in range(9):
		Gaussian_mask[i].append(100*rv.pdf([i,j]))

maxvalue = Gaussian_mask[4][4]
Gaussian_mask = [Gaussian_mask[i][j]/maxvalue for j in range(9) for i in range(9)]
Gaussian_mask = np.array(Gaussian_mask)

plt.imshow(Gaussian_mask, cmap = 'gray')
plt.savefig('Gaussian_mask.png',bbox_inches='tight')

y = np.zeros((img.shape[0],img.shape[1]))


patch_list = np.empty((img.shape[0],img.shape[1],9,9))

print("generating patches")
for i in range(img.shape[0]):
	for j in range(img.shape[1]):

		patch_list[i][j][:][:] = getPatch(img,i,j)
		patch_list[i][j][4][4] = np.NaN

for i in range(img.shape[0]):
	for j in range(img.shape[1]):

		patch_center = np.copy(patch_list[i,j])
		patch_center = patch_center.reshape(1,81)

		window_patches = np.empty((25,25,9,9))
		# window_patches[max(0,12-i):24-max(0,i-img.shape[0]+13),max(0,12-j):24-max(0,j-img.shape[1]+13)] = np.copy(patch_list[max(0,i-12):min(img.shape[0],i+12),max(0,j-12):min(img.shape[1],j+12)])
		window_patches = np.copy(patch_list[max(0,i-12):min(img.shape[0],i+12),max(0,j-12):min(img.shape[1],j+12)])

		window_patches = window_patches - patch_center
		window_patches = window_patches**2
		window_patches = window_patches*Gaussian_mask
		distances = np.nansum(window_patches,1)
		distances = distances*(-1)
		distances = distances/(2*(h**2))
		weights = 2.718**distances

		weights = weights[~np.isnan(weights)]
		weights = weights/np.nansum(weights)

		img_window = np.copy(img[max(0,i-12):min(img.shape[0],i+12),max(0,j-12):min(img.shape[1],j+12)])
		img_window = img_window.flatten()

		y[i][j] = np.dot(weights,img_window)


plt.imshow(y, cmap='gray')
plt.show()

plt.imshow(y, cmap='gray')
plt.colorbar()
plt.savefig('grass_modified_1')
