from os import listdir

import cv2

dir = listdir('scandinavianPNG')
index = 1
for filename in dir:
    image = cv2.imread('scandinavianPNG/img'+str(index)+'.png')

    print filename

    ratio = float(image.shape[1]) / image.shape[0]
    image = cv2.resize(image, (int(386 * ratio), 386), interpolation=cv2.INTER_AREA)

    cv2.imwrite('smaller_images/img'+str(index)+'.png', image)
    index += 1