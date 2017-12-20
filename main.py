# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
from scipy import stats
import pandas as pd
import mlpnetwork


imageSet = ['road.png','astronaut.png']

for img in imageSet:
	# load the image and convert it to a floating point data type
	image = img_as_float(io.imread(img))
	segments = slic(image, n_segments = 300, sigma = 5)
	datax=[]
	datay=[]
	img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
	(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# cv2.imshow('image',img)
	r1=np.unique(segments)
	for i in r1:
		indx,indy = np.where(segments==i)
		imgVal=image[indx,indy]
		meanval1=np.mean(imgVal,axis=0)
		imgVal1=im_bw[indx,indy]
		mode1,mf=stats.mode(imgVal1, axis=None)
		datax.append(meanval1)
		datay.append(mode1)

mlpnetwork.mlpnetwork(datax,datay)