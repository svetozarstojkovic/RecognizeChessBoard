import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filter import threshold_adaptive
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import dilation, square

import cv2

from mapping import connect_number_and_model, connect_number_and_name, connect_number_and_code


def field(img, model, values):
    img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)
    img_for_thres = rgb2gray(img)
    img_for_thres = (img_for_thres * 255).astype('uint8')
    thresh = threshold_adaptive(img_for_thres, 13, offset=1)

    thresh = (thresh * 255).astype('uint8')

    # this part saves figures
    # cv2.imwrite('dataset/' + str(randint(0,99999))+'.png', thresh)

    output = model.predict(thresh.reshape(1, 900), verbose=0)
    # print connect_number_and_name(np.argmax(output))
    values.append(np.argmax(output))

    # plt.imshow(thresh, 'gray')
    # plt.show()


def main_process(filename, model, show):
    values = []

    img = cv2.imread(filename)
    ratio = float(img.shape[1])/img.shape[0]
    img = cv2.resize(img, (int(386*ratio), 386), interpolation=cv2.INTER_AREA)
    img_clear = img.copy()

    print 'Thresholding in progress...'
    thresh = threshold_adaptive(img, 101, offset=20)
    thresh = rgb2gray(thresh)
    thresh = thresh < 0.5
    thresh = dilation(thresh, selem=square(5))
    thresh *= 255

    print 'Thresholding done.'

    # plt.imshow(thresh, 'gray')
    # plt.show()

    labeled_img = label(thresh)
    regions = regionprops(labeled_img)

    print len(regions)

    h_old = 0
    w_old = 0
    vertical_middle = 0
    horizontal_middle = 0
    vertical_begin = 0
    horizontal_begin = 0
    print 'Finding table...'

    for region in regions:
        bbox = region.bbox
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]

        if np.logical_and(np.logical_and(h > h_old, w > w_old),
                          np.logical_and(h < thresh.shape[0], w < thresh.shape[1])):
            okvir = thresh[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            vertical_middle = (okvir.shape[0]) / 2
            horizontal_middle = (okvir.shape[1]) / 2
            vertical_begin = bbox[0]
            horizontal_begin = bbox[1]

            h_old = h
            w_old = w

    white_fields = (okvir == 1).sum()
    area = okvir.shape[0] * okvir.shape[1]

    perc = float(white_fields)/area

    print "Percentage of whites: "+str(perc)

    if perc < 0.55:
        labeled_img = label(okvir)
        regions = regionprops(labeled_img)

        for region in regions:
            bbox = region.bbox
            h = bbox[2] - bbox[0]
            w = bbox[3] - bbox[1]

            if np.logical_and(h < okvir.shape[0]/5, w < okvir.shape[1]/5):
                okvir[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0

    print 'Table found.'

    # plt.imshow(okvir, 'gray')
    # plt.show()

    furthest1_i = 0
    furthest1_j = 0

    furthest2_i = 0
    furthest2_j = 0

    furthest3_i = 0
    furthest3_j = 0

    furthest4_i = 0
    furthest4_j = 0

    old_distance = 0
    distance = 0
    print 'Finding corners...'

    okvir = (okvir*255).astype('uint8')
    for i in range(0, vertical_middle):
        for j in range(0, horizontal_middle):
            if okvir[i, j] == 255:
                distance = np.power((vertical_middle - i), 2) + np.power((horizontal_middle - j), 2)
            if distance > old_distance:
                furthest1_j = j
                furthest1_i = i
                old_distance = distance
                distance = 0

    old_distance = 0
    distance = 0
    for i in range(0, vertical_middle):
        for j in range(horizontal_middle, okvir.shape[1]):
            if okvir[i, j] == 255:
                distance = np.power((vertical_middle - i), 2) + np.power((horizontal_middle - j), 2)
            if distance > old_distance:
                furthest2_j = j
                furthest2_i = i
                old_distance = distance
                distance = 0
    old_distance = 0
    distance = 0
    for i in range(vertical_middle, okvir.shape[0]):
        for j in range(0, horizontal_middle):
            if okvir[i, j] == 255:
                distance = np.power((vertical_middle - i), 2) + np.power((horizontal_middle - j), 2)
            if distance > old_distance:
                furthest3_j = j
                furthest3_i = i
                old_distance = distance
                distance = 0
    old_distance = 0
    distance = 0
    for i in range(vertical_middle, okvir.shape[0]):
        for j in range(horizontal_middle, okvir.shape[1]):
            if okvir[i, j] == 255:
                distance = np.power((vertical_middle - i), 2) + np.power((horizontal_middle - j), 2)
            if distance > old_distance:
                furthest4_j = j
                furthest4_i = i
                old_distance = distance
                distance = -1

    # img[longest[0], longest[1]] = (255,0,0)
    # print '\tTop left corner: ['+str(furthest1_i)+','+str(furthest1_j)+']'
    # print '\tTop right corner: ['+str(furthest2_i)+','+str(furthest2J)+']'
    # print '\tBottom left corner: ['+str(furthest4_i)+','+str(furthest4_j)+']'
    # print '\tBottom right corner: ['+str(furthest3_i)+','+str(furthest3_j)+']'
    # print 'Corners found.'

    # cv2.circle(imgTable, (horizontal_middle, vertical_middle), 10, (255, 255, 255), -1)
    # cv2.circle(imgTable, (furthest1_j, furthest1_i), 10, (255, 255, 0), -1)
    # cv2.circle(imgTable, (furthest2J, furthest2_i), 10, (0, 255, 255), -1)
    # cv2.circle(imgTable, (furthest3_j, furthest3_i), 10, (0, 0, 255), -1)
    # cv2.circle(imgTable, (furthest4_j, furthest4_i), 10, (255, 0, 0), -1)

    # plt.imshow(imgTable)
    # plt.show()

    print 'Warping table in progress...'
    pts1 = np.float32([[furthest1_j + horizontal_begin, furthest1_i + vertical_begin],
                       [furthest2_j + horizontal_begin, furthest2_i + vertical_begin],
                       [furthest3_j + horizontal_begin, furthest3_i + vertical_begin],
                       [furthest4_j + horizontal_begin, furthest4_i + vertical_begin]])

    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    m = cv2.getPerspectiveTransform(pts1, pts2)

    table = cv2.warpPerspective(img_clear, m, (300, 300))
    table = table[3:297, 3:297]
    print 'Warping table done.'

    # plt.imshow(table)
    # plt.show()

    print 'Handling specific fields in progress...'
    for i in range(1, 9):
        figure_row = table[(i-1)*table.shape[1]/8: (table.shape[1]*i)/8, 0:table.shape[0]]
        for j in range(1, 9):
            fig = figure_row[0:table.shape[1], (j-1)*table.shape[0]/8: (table.shape[0]*j)/8]
            field(img=fig, model=model, values=values)
    print 'Handling specific fields done.'


    if show:
        background = Image.new('RGBA', (480, 480), (255, 255, 255, 255))
        for i in range(0, 64):
            red = i // 8
            column = i % 8
            offset = (column*60, red*60)
            if(red + column) % 2 == 1:
                img = Image.new('RGBA', (60, 60), (100, 100, 100, 255))
            else:
                img = Image.new('RGBA', (60, 60), (255, 255, 255, 255))
            background.paste(img, offset)
        value_index = 0
        for value in values:
            red = value_index // 8
            column = value_index % 8
            offset = (column*60, red*60)
            img = connect_number_and_model(value)
            background.paste(img, offset, mask=img)
            value_index += 1

        print 'Everything is done.'
        fig = plt.figure()

        a = fig.add_subplot(1, 2, 1)
        a.set_title('Original')
        plt.imshow(img_clear)

        a = fig.add_subplot(1, 2, 2)
        plt.imshow(background)
        a.set_title('Output')

        img_name = filename.split('/')
        img_name = img_name[1].split('.')
        print img_name[0]

        plt.show()

    return values
