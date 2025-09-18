import numpy as np
from collections import defaultdict, deque
from skimage.morphology import skeletonize
import cv2

import numpy as np
from skimage.morphology import skeletonize
from collections import defaultdict, deque

mask = cv2.imread("sim_data/mask/maskthick_cam1.png", cv2.IMREAD_GRAYSCALE)
skeleton = skeletonize(mask > 10).astype(np.uint8)
# cv2.imwrite("mask.png", skeleton * 255)
points = np.argwhere(skeleton > 0)

end_point_conv = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
degree_conv = end_point_conv 