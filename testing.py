from os import listdir
import matplotlib.pyplot as plt
from mapping import connect_number_and_code
import numpy as np


def generate_output(output_name, values):
    test_file = open(output_name, 'w')
    print output_name
    index = 0
    for value in values:
        if index % 8 == 0:
            test_file.write('\n')
        test_file.write(connect_number_and_code(value) + ';')
        index += 1

    test_file.close()


def statistics():
    file_names = listdir('exact_values')
    x = []
    y = []
    for filename in file_names:
        value = statistics_for_one_file(filename=filename, stats=False)
        x.append(value[0])
        y.append(value[1])

    print 'Average: '+str(np.mean(y) * 100)+' %'
    argmin = y.index(min(y))
    print 'Min accuracy: '+str(y[argmin] * 100) + ' % on image img'+ str(x[argmin])
    argmax = y.index(max(y))
    print 'Max accuracy: '+str(y[argmax] * 100) + ' % on image img'+ str(x[argmax])

    plt.plot(x, y, 'rs')
    plt.xlabel('images')
    plt.ylabel('accuracy')
    plt.axhline(y=np.mean(y), xmin=0, xmax=1, hold=None)
    plt.axis([0, len(x), 0, 1])
    plt.yticks(np.arange(0, 1.05, 0.05))
    # plt.xticks(np.arange(0, len(x)+2, 4))
    plt.show()


def statistics_for_one_file(filename, stats):
    exact = open('exact_values/'+filename, 'r')
    output = open('outputs/'+filename, 'r')
    exact_values = []
    for line in exact:
        pieces = line.split(';')
        for piece in pieces:
            if np.logical_and(piece != '', piece != '\n'):
                exact_values.append(piece)

    index = 0
    acc = 0

    for line in output:
        pieces = line.split(';')
        for piece in pieces:
            if np.logical_and(piece != '', piece != '\n'):
                if exact_values[index] == piece:
                    acc += 1
                index += 1

    ret = []
    ret.append(filename.split('.')[0][3:])
    ret.append(float(acc)/64)

    if stats:
        print 'For image: '+filename + ' accuracy is ' + str((float(acc)/64)*100)+' %'

    return ret