from os import listdir
from skimage.color import gray2rgb
from skimage.filter import threshold_adaptive,threshold_otsu

from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential

from skimage.color import rgb2grey


from PIL import Image
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import closing
from skimage.morphology import dilation, square
from skimage.morphology import erosion
from skimage.morphology import opening
from scipy import ndimage
from random import randint



import cv2

import numpy as np

# for i in range(5856, 5891):
#     im = Image.open('scandinavianImg/DSC0'+str(i)+'.JPG')
#     broj = i-5855
#     im.save('scandinavianPNG/img'+str(broj)+'.png')
from keras.optimizers import SGD

resizeFactor = 30
resizeFactorNN = 900

def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype='int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal

def connect_number_and_name(number):
    if number == 0:
        return 'Pawn black'
    elif number == 1:
        return 'Pawn white'
    elif number == 2:
        return 'Empty white field'
    elif number == 3:
        return 'Empty black field'
    elif number == 4:
        return 'King white'
    elif number == 5:
        return 'King black'
    elif number == 6:
        return 'Queen white'
    elif number == 7:
        return 'Queen black'
    elif number == 8:
        return 'Rook white'
    elif number == 9:
        return 'Rook black'
    elif number == 10:
        return 'Knight white'
    elif number == 11:
        return 'Knight black'
    elif number == 12:
        return 'Bishop white'
    elif number == 13:
        return 'Bishop black'
    else:
     print 'Neither one'
     return 'Neither one'

def connect_number_and_model(number):
    if number == 0:
        return Image.open('model_figures/pawn_b.png')
    elif number == 1:
        return Image.open('model_figures/pawn_w.png')
    elif number == 2:
        return Image.new('RGBA', (60, 60), (255,255,255,255))
    elif number == 3:
        return Image.new('RGBA', (60, 60), (100,100,100,255))
    elif number == 4:
        return Image.open('model_figures/king_w.png')
    elif number == 5:
        return Image.open('model_figures/king_b.png')
    elif number == 6:
        return Image.open('model_figures/queen_w.png')
    elif number == 7:
        return Image.open('model_figures/queen_b.png')
    elif number == 8:
        return Image.open('model_figures/rook_w.png')
    elif number == 9:
        return Image.open('model_figures/rook_b.png')
    elif number == 10:
        return Image.open('model_figures/knight_w.png')
    elif number == 11:
        return Image.open('model_figures/knight_b.png')
    elif number == 12:
        return Image.open('model_figures/bishop_w.png')
    elif number == 13:
        return Image.open('model_figures/bishop_b.png')
    else:
     print 'Neither one'
     return 'Neither one'

def connect_name_and_number(name):
    if name == 'pawn_b':
        return 0
    elif name == 'pawn_w':
        return 1
    elif name == 'field_w':
        return 2
    elif name == 'field_b':
        return 3
    elif name == 'king_w':
        return 4
    elif name == 'king_b':
        return 5
    elif name == 'queen_w':
        return 6
    elif name == 'queen_b':
        return 7
    elif name == 'rook_w':
        return 8
    elif name == 'rook_b':
        return 9
    elif name == 'knight_w':
        return 10
    elif name == 'knight_b':
        return 11
    elif name == 'bishop_w':
        return 12
    elif name == 'bishop_b':
        return 13
    else:
     print 'neither one'
     return 2

def learn_NN(model):

    train = np.zeros((1,resizeFactorNN))
    out = np.array([])
    folders = listdir('dataset')
    for folder in folders:
        dir = listdir('dataset/'+str(folder))
        for file in dir:
            trainImage = cv2.imread('dataset/' + str(folder) + '/' + str(file))
            # trainImage = rgb2gray(trainImage)
            #trainImage = threshold_adaptive(trainImage, 201, offset=50)
            trainImage = rgb2gray(trainImage)
            trainImage = trainImage.reshape(1, resizeFactorNN)
            trainImage = trainImage * 255
            trainImage = trainImage[0]

            train = np.vstack([train, trainImage])
            # train.append(trainImage)
            out = np.append(out, connect_name_and_number(folder))
            # out.append(0)

    train_out = to_categorical(out.astype('int'), 14)
    train = np.delete(train, (0), axis=0)

    model.add(Dense(70, input_dim=resizeFactorNN))
    model.add(Activation('sigmoid')) #sigmoid
    model.add(Dense(50))
    model.add(Activation('sigmoid')) # tanh
    model.add(Dense(14))
    model.add(Activation('tanh')) # tanh

    sgd = SGD(lr=0.1, decay=0.0001, momentum=0.7)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    training = model.fit(train, train_out, nb_epoch=10000, batch_size=400, verbose=1)
    print training.history['loss'][-1]


def field(img, model):

    img = cv2.resize(img, (resizeFactor, resizeFactor), interpolation=cv2.INTER_AREA)
    imgForThres = rgb2gray(img)
    imgForThres = (imgForThres * 255).astype('uint8')
    thresh = threshold_adaptive(imgForThres, 13, offset=1)


    thresh = (thresh * 255).astype('uint8')


    #cv2.imwrite('dataset/' + str(randint(0,99999))+'.png', thresh)

    output = model.predict(thresh.reshape(1, resizeFactorNN), verbose=1)
    print connect_number_and_name(np.argmax(output))
    values.append(np.argmax(output))

    # plt.imshow(thresh, 'gray')
    # plt.show()


model = Sequential()
values = []

learn_NN(model)
filename = 'scandinavianPNG/img28.png'
img = cv2.imread(filename)
img = cv2.resize(img, (img.shape[1]/10, img.shape[0]/10), interpolation=cv2.INTER_AREA)
imgClear = cv2.imread(filename)
imgClear= cv2.resize(img, (imgClear.shape[1]/10, imgClear.shape[0]/10), interpolation=cv2.INTER_AREA)

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,100,255,0)
print 'Thresholding in progress...'
thresh = threshold_adaptive(img, 101, offset=20)
thresh = rgb2gray(thresh)
thresh = thresh < 0.5
thresh = dilation(thresh, selem=square(5))
thresh = (thresh * 255).astype('uint8')
print 'Thresholding done.'

#thresh = np.invert(thresh)

# thresh = erosion(thresh, selem=square(5))
# thresh = dilation(thresh, selem=square(50))

# plt.imshow(thresh, 'gray')
# plt.show()

labeled_img = label(thresh)
regions = regionprops(labeled_img)

h_old = 0
w_old = 0
verticalMiddle = 0
horizontalMiddle = 0
vertBegin = 0
horizBegin = 0
print 'Finding table...'
for region in regions:
    bbox = region.bbox
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]

    if np.logical_and(h>h_old, w>w_old):
        border = thresh[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        verticalMiddle = (border.shape[0] ) / 2
        horizontalMiddle = (border.shape[1]) / 2
        vertBegin = bbox[0]
        horizBegin = bbox[1]

        h_old = h
        w_old = w
print '\tVerticalCenter: '+str(verticalMiddle)
print '\tHorizontalCenter: '+str(horizontalMiddle)
print 'Table found.'
# plt.imshow(border, 'gray')
# plt.show()
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#
# #cv2.drawContours(img, contours, -1, (0,255,0), 3)
#
# vertical = []
# horizontal = []
#
# for cnt in contours:
#     vertical.append(cnt[0][0][1])
#     horizontal.append(cnt[0][0][0])
#
#
# top = np.argmin(vertical)
# right = np.argmax(horizontal)
# down = np.argmax(vertical)
# left = np.argmin(horizontal)
#
# print len(vertical)
# print len(horizontal)
#
# topCnt = contours[top]
# rightCnt = contours[right]
# downCnt = contours[down]
# leftCnt = contours[left]
#
# print top
# print right
# print down
# print left
#
# cv2.drawContours(img, [topCnt], 0, (0,255,0), 10) # ceo okvir
# # cv2.drawContours(img, [rightCnt], 0, (0,255,255), 50) # jedna tacka
# # cv2.drawContours(img, [downCnt], 0, (0,0,255), 50) # jedna tacka
# # cv2.drawContours(img, [leftCnt], 0, (255,0,0), 50) # jedna tacka
#
# #
# plt.imshow(img)
# plt.show()
#
# okvir = np.logical_and(np.logical_and(img[...,0] == 0, img[...,1] == 255), img[...,2] == 0)
# okvir = dilation(okvir, selem=square(10))

# labeled_img = label(okvir)
# regions = regionprops(labeled_img)
#
# h_old = 0
# w_old = 0
# verticalMiddle = 0
# horizontalMiddle = 0
#
# for region in regions:
#     bbox = region.bbox
#     h = bbox[2] - bbox[0]
#     w = bbox[3] - bbox[1]
#
#     if np.logical_and(h>h_old, w>w_old):
#         border = okvir[bbox[0] : bbox[2], bbox[1] : bbox[3]]
#         verticalMiddle = (bbox[0] + bbox[2] ) / 2
#         horizontalMiddle = (bbox[1] + bbox[3]) / 2
#
#         h_old = h
#         w_old = w
okvir = border
# okvir = erosion(okvir, selem=square(10))
furthest1I = 0
furthest1J = 0

furthest2I = 0
furthest2J = 0

furthest3I = 0
furthest3J = 0

furthest4I = 0
furthest4J = 0

oldDistance = 0
distance = 0
print 'Finding corners...'
okvir =  (okvir*255).astype('uint8')
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
print '\tTop left corner: ['+str(furthest1I)+','+str(furthest1J)+']'
print '\tTop right corner: ['+str(furthest2I)+','+str(furthest2J)+']'
print '\tBottom left corner: ['+str(furthest4I)+','+str(furthest4J)+']'
print '\tBottom right corner: ['+str(furthest3I)+','+str(furthest3J)+']'
print 'Corners found.'
cv2.circle(okvir, (horizontalMiddle, verticalMiddle), 10, (255, 255, 255), -1)
cv2.circle(okvir, (furthest1J, furthest1I), 10, (255, 255, 255), -1)
cv2.circle(okvir, (furthest2J, furthest2I), 10, (255, 255, 255), -1)
cv2.circle(okvir, (furthest3J, furthest3I), 10, (255, 255, 255), -1)
cv2.circle(okvir, (furthest4J, furthest4I), 10, (255, 255, 255), -1)


print 'Warping table in progress...'
pts1 = np.float32([[furthest1J + horizBegin,furthest1I + vertBegin],[furthest2J + horizBegin,furthest2I + vertBegin],[furthest3J + horizBegin,furthest3I + vertBegin],[furthest4J + horizBegin,furthest4I + vertBegin]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

table = cv2.warpPerspective(imgClear,M,(300,300))
table = table[3:297, 3:297]
print 'Warping table done.'


figures = []
print 'Handling specific fields in progress...'
for i in range(1,9):
    figRow = table[(i-1)*table.shape[1]/8: (table.shape[1]*i)/8, 0:table.shape[0]]
    # plt.imshow(figRow)
    # plt.show()
    for j in range(1, 9):
        fig = figRow[0:table.shape[1], (j-1)*table.shape[0]/8: (table.shape[0]*j)/8]
        # fig = np.logical_and(np.logical_and(fig[..., 0] < 150,fig[..., 1] < 150), fig[..., 2] < 150)
        # fig = erosion(fig, selem=square(3))
        field(img=fig, model=model)
print 'Handling specific fields done.'
background = Image.new('RGBA', (480, 480), (255, 255, 255, 255))
for i in range(0,64):
    red = i//8
    kolona = i%8
    offset = (kolona*60, red*60)
    if(red + kolona) % 2 == 1:
        img = Image.new('RGBA', (60, 60), (100,100,100,255))
    else:
        img = Image.new('RGBA', (60, 60), (255,255,255,255))
    background.paste(img, offset)
valueIndex = 0
for value in values:
    red = valueIndex // 8
    kolona = valueIndex % 8
    offset = (kolona*60, red*60)
    img = connect_number_and_model(value)
    background.paste(img, offset, mask=img)
    valueIndex += 1

print 'All is done.'
fig = plt.figure()

a=fig.add_subplot(1,2,1)
a.set_title('Original')
plt.imshow(imgClear)

a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(background)
a.set_title('Extracted values')

plt.show()
# plt.imshow(table)
# plt.show()