import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import img_as_float

# img = cv2.imread("rec_image/C1.bmp", cv2.IMREAD_GRAYSCALE)
# thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
# plt.imsave("thresh.png", thresh, cmap='gray')

# TODO
import json
para = json.load(open('sim_data/calibration/camera_params.json'))
print(para)