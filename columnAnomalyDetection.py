import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2

p = '/home/osama/Documents/datasets/textile/dataset/training_set/1'
imgName = 'C3R1EDTF.TIF'

img = imageio.imread(p + '//' + imgName)
img = cv2.fastNlMeansDenoising(np.uint8(img), 10, 9, 3)
img = np.absolute(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=7))
ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
imgSliceList = np.ndarray((128,512))
for i in range(128):
    imgSlice = img[:, i*6 : 6*(i+1)]
    avgSlice = np.mean(imgSlice, axis=1)
    imgSliceList[i,:] = avgSlice

plt.subplot(211)
plt.imshow(img, cmap='gray')
plt.subplot(212)
plt.imshow(imgSliceList, cmap='gray')