from skimage.morphology import skeletonize
import cv2
import numpy as np


mask = cv2.imread("sim_data/mask/mask_cam2.png", cv2.IMREAD_GRAYSCALE)
skeleton = skeletonize(mask > 1)
cv2.imwrite("skeleton.png", skeleton.astype(np.uint8)*255)