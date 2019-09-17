import cv2
import os
import numpy as np


import glob
img_list = glob.glob('/media/guruleben/datasetsRiyaj/mmdetection_object_detection_demo/data/VOC2007/JPEGImages/*')

for img in img_list:
	im = cv2.imread(img)
	img_name = os.path.splitext(os.path.basename(img))
	print(im.shape,img_name)
	im  = cv2.resize(im,(800,800))
	cv2.imwrite('/media/guruleben/datasetsRiyaj/mmdetection_object_detection_demo/data/VOC2007/JPEGImages/'+img_name[0]+img_name[1],im)