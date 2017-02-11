from os import listdir
from skimage.color import rgb2gray

import cv2

dir = listdir('figures3')
for file in dir:
    image = cv2.imread('figures3/' + str(file))
    image = rgb2gray(image)
    image = image * 255
    justFigure = cv2.resize(image, (30, 30), interpolation=cv2.INTER_AREA)

    cv2.imwrite('figures4/' + str(file), justFigure)