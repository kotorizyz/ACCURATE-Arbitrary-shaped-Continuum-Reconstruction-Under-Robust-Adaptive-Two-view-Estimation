import pydicom
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from pydicom.tag import Tag
import cv2
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.morphology import skeletonize

file = 'ctdataset/SE5/IM1'
dcm = pydicom.dcmread(file)

num_frames = dcm.NumberOfFrames
num_mats = len(dcm[0x0021100b].value)

angle1 = np.array([dcm[0x00191001].value]*num_frames) + dcm[0x00191197].value
angle2 = np.array([dcm[0x00191002].value]*num_frames) + dcm[0x00191198].value
angle3 = np.array([dcm[0x00191003].value]*num_frames) + dcm[0x00191199].value

P_list = []
for item in dcm[0x0021100b]:
    vals = list(map(float, item[0x0021100c].value))
    P = np.array(vals).reshape(3,4)
    P_list.append(P)

P_list = P_list[3:294]
imgs = dcm.pixel_array