from skimage.filter import threshold_adaptive,threshold_otsu

from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from skimage.color import rgb2grey


from skimage.io import imread
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import closing
from skimage.morphology import dilation, square
from skimage.morphology import erosion
from skimage.morphology import opening
from scipy import ndimage



import cv2

import numpy as np

# for i in range(5856, 5891):
#     im = Image.open('scandinavianImg/DSC0'+str(i)+'.JPG')
#     broj = i-5855
#     im.save('scandinavianPNG/img'+str(broj)+'.png')


def field(img):

    # img = erosion(img, selem=square(20))

    labeled_img = label(thresh)
    regions = regionprops(labeled_img)

    for region in regions:
        bbox = region.bbox
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]

        print 'H: ' + str(h)
        print 'V: ' + str(w)
        print 'bbox[0]: ' + str(bbox[0])
        print 'bbox[2]: ' + str(bbox[2])
        print 'bbox[1]: ' + str(bbox[1])
        print 'bbox[3]: ' + str(bbox[3])

        if np.logical_and(h > img.shape[1]/2, w > img.shape[0]/2):
            figure = img[bbox[0]: bbox[2], bbox[1]: bbox[3]]
            figure = dilation(figure, selem=square(20))
            plt.imshow(figure, 'gray')
            plt.show()


filename = 'scandinavianPNG/img7.png'
img = cv2.imread(filename)
imgClear = cv2.imread(filename)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # edges = cv2.Canny(gray,50,150,apertureSize = 3)
# edges = threshold_adaptive(gray, 801, offset=50)
# edges = np.invert(edges)
# edges = (edges * 255).astype('uint8')
# lines = cv2.HoughLines(edges,1,np.pi/360,1000)
#
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#
# plt.imshow(img, 'gray')
# plt.show()

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,100,255,0)
thresh = threshold_adaptive(img, 801, offset=50)
thresh = rgb2gray(thresh)
thresh = thresh < 0.5
print thresh
#thresh = np.invert(thresh)
thresh = dilation(thresh, selem=square(10))


contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

#cv2.drawContours(img, contours, -1, (0,255,0), 3)

vertical = []
horizontal = []

for cnt in contours:
    vertical.append(cnt[0][0][1])
    horizontal.append(cnt[0][0][0])


top = np.argmin(vertical)
right = np.argmax(horizontal)
down = np.argmax(vertical)
left = np.argmin(horizontal)

print len(vertical)
print len(horizontal)

topCnt = contours[top]
rightCnt = contours[right]
downCnt = contours[down]
leftCnt = contours[left]

print top
print right
print down
print left

cv2.drawContours(img, [topCnt], 0, (0,255,0), 10) # ceo okvir
# cv2.drawContours(img, [rightCnt], 0, (0,255,255), 50) # jedna tacka
# cv2.drawContours(img, [downCnt], 0, (0,0,255), 50) # jedna tacka
# cv2.drawContours(img, [leftCnt], 0, (255,0,0), 50) # jedna tacka


plt.imshow(img)
plt.show()

okvir = np.logical_and(np.logical_and(img[...,0] == 0, img[...,1] == 255), img[...,2] == 0)
okvir = dilation(okvir, selem=square(10))

labeled_img = label(okvir)
regions = regionprops(labeled_img)

h_old = 0
w_old = 0
verticalMiddle = 0
horizontalMiddle = 0

for region in regions:
    bbox = region.bbox
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]

    if np.logical_and(h>h_old, w>w_old):
        border = okvir[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        verticalMiddle = (bbox[0] + bbox[2] ) / 2
        horizontalMiddle = (bbox[1] + bbox[3]) / 2

        h_old = h
        w_old = w

furthest1I = 0
furthest1J = 0

furthest2I = 0
furthest2J = 0

furthest3I = 0
furthest3J = 0

furthest4I = 0
furthest4J = 0

print verticalMiddle
print horizontalMiddle
print okvir.shape
oldDistance = 0
distance = 0
for i in range(0, verticalMiddle):
    for j in range(0, horizontalMiddle):
        if (okvir[i,j] == 255):
            distance = np.power((verticalMiddle - i),2) + np.power((horizontalMiddle - j), 2)
        if distance > oldDistance:
            furthest1J = j
            furthest1I = i
            oldDistance = distance
            distance = 0
oldDistance = 0
distance = 0
for i in range(0,verticalMiddle):
    for j in range(horizontalMiddle, okvir.shape[1]):
        if (okvir[i,j] == 255):
            distance = np.power((verticalMiddle - i), 2) + np.power((horizontalMiddle - j), 2)
        if distance > oldDistance:
            furthest2J = j
            furthest2I = i
            oldDistance = distance
            distance = 0
oldDistance = 0
distance = 0
for i in range(verticalMiddle,okvir.shape[0]):
    for j in range(0, horizontalMiddle):
        if (okvir[i,j] == 255):
            distance = np.power((verticalMiddle - i), 2) + np.power((horizontalMiddle - j), 2)
        if distance > oldDistance:
            furthest3J = j
            furthest3I = i
            oldDistance = distance
            distance = 0
oldDistance = 0
distance = 0
for i in range(verticalMiddle,okvir.shape[0]):
    for j in range(horizontalMiddle, okvir.shape[1]):
        if (okvir[i,j] == 255):
            distance = np.power((verticalMiddle - i), 2) + np.power((horizontalMiddle - j), 2)
        if distance > oldDistance:
            furthest4J = j
            furthest4I = i
            oldDistance = distance
            distance = -1


# img[longest[0], longest[1]] = (255,0,0)

# cv2.circle(img, (horizontalMiddle, verticalMiddle), 63, (255, 0, 0), -1)
# cv2.circle(img, (furthest1J, furthest1I), 63, (255, 0, 0), -1)
# cv2.circle(img, (furthest2J, furthest2I), 63, (255, 255, 0), -1)
# cv2.circle(img, (furthest3J, furthest3I), 63, (255, 255, 255), -1)
# cv2.circle(img, (furthest4J, furthest4I), 63, (0, 255, 0), -1)

# plt.imshow(img)
# plt.show()

pts1 = np.float32([[furthest1J,furthest1I],[furthest2J,furthest2I],[furthest3J,furthest3I],[furthest4J,furthest4I]])
pts2 = np.float32([[0,0],[3000,0],[0,3000],[3000,3000]])

M = cv2.getPerspectiveTransform(pts1,pts2)

table = cv2.warpPerspective(imgClear,M,(3000,3000))
table = table[30:2970, 30:2970]

plt.imshow(table)
plt.show()

# plt.imshow(okvir, 'gray')
# plt.show()

# imgGrey = rgb2grey(img)
# imgForThres = (imgGrey * 255).astype('uint8')

#thresh =threshold_adaptive(imgForThres, 801, offset=50)
# # thresh = dilation(thresh, selem=square(10))
# # thresh = erosion(thresh, selem=square(40))
# #thresh = erosion(thresh, selem=square(30))
#
#thresh =np.invert(thresh)
#
# # thresh = dilation(thresh, selem=square(5))
# # thresh = dilation(thresh, selem=square(5))
# #thresh = erosion(thresh, selem=square(15))
#
# plt.imshow(thresh, 'gray')
# plt.show()
#


#ret,thresh = cv2.threshold(img,90,255,0)
# for i in range(0,thresh.shape[0]):
#     for j in range(0, thresh.shape[1]):
#         lastWhite = thresh.shape[1]
#         white = False
#         for k in range(thresh.shape[1]-1, 0, -1):
#             if thresh[i,k] == 255:
#                 lastWhite = k
#                 break
#         if thresh[i,j] == 255:
#             white = True
#
#         if np.logical_and(white, j < lastWhite):
#             thresh[i,j] = 255

# print thresh.shape
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
#
# cnt = contours[0]
# M = cv2.moments(cnt)
#
# rect = cv2.minAreaRect(cnt)
# print rect
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# im = cv2.drawContours(img,[box],0,(0,0,255),2)

# labeled_img = label(thresh)
# regions = regionprops(labeled_img)
#
# table_width = 0
# table_height = 0
# img_table = []
# tabla = []
#
# for region in regions:
#     bbox = region.bbox
#     h = bbox[2] - bbox[0]
#     w = bbox[3] - bbox[1]
#
#     print 'H: '+str(h)
#     print 'V: '+str(w)
#     print 'bbox[0]: ' + str(bbox[0])
#     print 'bbox[2]: ' + str(bbox[2])
#     print 'bbox[1]: ' + str(bbox[1])
#     print 'bbox[3]: ' + str(bbox[3])
#
#     if np.logical_and(h>1000, w>1000):  # put h_old and w_old here
#         img_table = img[bbox[0]: bbox[2], bbox[1]: bbox[3]]
#         thresh1 = thresh[bbox[0]: bbox[2], bbox[1]: bbox[3]]
#         print str(img_table.shape)
#         table_width = img_table.shape[0]
#         table_height = img_table.shape[1]
#         tabla.append(bbox[0])
#         tabla.append(bbox[1])
#         tabla.append(bbox[2])
#         tabla.append(bbox[3])
#         # plt.imshow(img_table)
#         # plt.show()
#
# # img_table = rgb2gray(img_table)
# # img_table = (img_table * 255).astype('uint8')
#
# #imgWithoutBorder = img[tabla[0] + table_height/100 : tabla[2] - table_height/100, tabla[1] + table_width/100 : tabla[3] - table_width/100]
# imgWithoutBorder = img[tabla[0] : tabla[2], tabla[1] : tabla[3]]
#
# #thresh = thresh.astype(np.uint8)
#
# thresh1 = erosion(thresh1, selem=square(13))
# thresh1 = dilation(thresh1, selem=square(50))
#
# plt.imshow(thresh1)
# plt.show()



# theta angle part
# lines = cv2.HoughLines(thresh, 1, np.pi/180, 2000)
#
# for rho,theta in lines[0]:
#     print 'Theta: '+str((theta*180) / np.pi)
#
#     if np.logical_and((theta*180) / np.pi < 47, (theta*180) / np.pi > 45):
#         imgWithoutBorder = ndimage.rotate(imgWithoutBorder, -1, cval=255)
#         imgForThres = rgb2gray(imgWithoutBorder)
#         imgForThres = (imgForThres * 255).astype('uint8')
#         thresh = threshold_adaptive(imgForThres, 201, offset=1)
#         thresh = np.invert(thresh)
#         thresh = erosion(thresh, selem=20)
#         thresh = dilation(thresh, selem=30)
#         plt.imshow(thresh)
#         plt.show()
#         labeled_img = label(thresh)
#         regions = regionprops(labeled_img)
#
#         for region in regions:
#             bbox = region.bbox
#             h = bbox[2] - bbox[0]
#             w = bbox[3] - bbox[1]
#
#             print 'H: ' + str(h)
#             print 'V: ' + str(w)
#             print 'bbox[0]: ' + str(bbox[0])
#             print 'bbox[2]: ' + str(bbox[2])
#             print 'bbox[1]: ' + str(bbox[1])
#             print 'bbox[3]: ' + str(bbox[3])
#
#             if np.logical_and(h > imgWithoutBorder.shape[1] * 9 / 10, w > imgWithoutBorder.shape[0] * 9 / 10):
#                 img_table = imgWithoutBorder[bbox[0]: bbox[2], bbox[1]: bbox[3]]
#                 # thresh = dilation(thresh, selem=square(5))
#                 # thresh = thresh[bbox[0]: bbox[2], bbox[1]: bbox[3]]
#         break
#     elif np.logical_and((theta*180) / np.pi > 43, (theta*180) / np.pi < 45):
#         imgWithoutBorder = ndimage.rotate(imgWithoutBorder, 1, cval=255)
#         imgForThres = rgb2gray(imgWithoutBorder)
#         imgForThres = (imgForThres * 255).astype('uint8')
#         thresh = threshold_adaptive(imgForThres, 201, offset=1)
#         thresh = np.invert(thresh)
#
#         labeled_img = label(thresh)
#         regions = regionprops(labeled_img)
#
#         for region in regions:
#             bbox = region.bbox
#             h = bbox[2] - bbox[0]
#             w = bbox[3] - bbox[1]
#
#             print 'H: ' + str(h)
#             print 'V: ' + str(w)
#             print 'bbox[0]: ' + str(bbox[0])
#             print 'bbox[2]: ' + str(bbox[2])
#             print 'bbox[1]: ' + str(bbox[1])
#             print 'bbox[3]: ' + str(bbox[3])
#
#             if np.logical_and(h > imgWithoutBorder.shape[1] * 9/ 10, w > imgWithoutBorder.shape[0] * 9 / 10):
#                 img_table = imgWithoutBorder[bbox[0]: bbox[2], bbox[1]: bbox[3]]
#                 # thresh = dilation(thresh, selem=square(5))
#                 # thresh = thresh[bbox[0]: bbox[2], bbox[1]: bbox[3]]
#
#         break
#
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     #cv2.line(imgWithoutBorder,(x1,y1),(x2,y2),(0,0,255),3)
#
#
# imgWithoutBorder = img_table
#theta angle part end


# print 'Prvi pixel: '+str(thresh[0,0])
# while (thresh[0,0]==False):
#     imgWithoutBorder = ndimage.rotate(imgWithoutBorder, -1)
#     imgWithoutBorder = imgWithoutBorder[0 + imgWithoutBorder.shape[1]/50 : imgWithoutBorder.shape[1] - imgWithoutBorder.shape[1]/50, 0 + imgWithoutBorder.shape[0]/50 : imgWithoutBorder.shape[0] - imgWithoutBorder.shape[0]/50]
#     thresh = ndimage.rotate(thresh, -1)
#     plt.imshow(imgWithoutBorder)
#     plt.show()



figures = []

for i in range(1,9):
    figRow = table[(i-1)*table.shape[1]/8: (table.shape[1]*i)/8, 0:table.shape[0]]
    plt.imshow(figRow)
    plt.show()
    for j in range(1, 9):
        fig = figRow[0:table.shape[1], (j-1)*table.shape[0]/8: (table.shape[0]*j)/8]
        # fig = np.logical_and(np.logical_and(fig[..., 0] < 150,fig[..., 1] < 150), fig[..., 2] < 150)
        # fig = erosion(fig, selem=square(3))
        imgForThres = rgb2gray(fig)
        imgForThres = (imgForThres * 255).astype('uint8')
        thresh = threshold_adaptive(imgForThres, 201, offset=1)
        #thresh = np.invert(thresh)
        #thresh = erosion(thresh, selem=square())
        # field(img=fig)
        plt.imshow(thresh, 'gray')
        plt.show()

# labeled_img = label(img_table)
# regions = regionprops(labeled_img)



# for region in regions:
#     bbox = region.bbox
#     h = bbox[2] - bbox[0]
#     w = bbox[3] - bbox[1]
#
#     print 'H: ' + str(h)
#     print 'V: ' + str(w)
#     print 'bbox[0]: ' + str(bbox[0])
#     print 'bbox[2]: ' + str(bbox[2])
#     print 'bbox[1]: ' + str(bbox[1])
#     print 'bbox[3]: ' + str(bbox[3])
#
#     #if np.logical_and(h > 1000, w > 1000):
#     img_figure = img[bbox[0]: bbox[2], bbox[1]: bbox[3]]
#     print str(img_figure.shape)
#     plt.imshow(img_figure)
#     plt.show()