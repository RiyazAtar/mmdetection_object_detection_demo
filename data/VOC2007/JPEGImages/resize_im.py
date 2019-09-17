import cv2
import os
import numpy as np


import glob
img_list = os.listdir('./')

for img in img_list:
	im = cv2.imread(img)
	img_name = os.path.splitext(os.path.basename(img))

	im  = cv2.resize(im,(800,600))
	cv2.imwrite(img_name[0]+img_name[1],im)