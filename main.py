from os.path import isfile

from keras.models import load_model, Sequential

from neural_network import learn_nn
from recognition import main_process
from testing import generate_output, statistics, statistics_for_one_file

if isfile('neural_network/model.h5'):
    print("Loading model from disk")
    model = load_model('neural_network/model.h5')
    print("Loaded model from disk")
else:
    print("Creating new neural network")
    model = Sequential()
    model = learn_nn(model)
    model.save('neural_network/model.h5')
    print("Neural network saved on disk")

# this is for creting new outputs for testing images

# for index in range(1, 81):
#     print 'Doing picture: '+str(index)
#     values = main_process('smaller_images/img'+str(index)+'.png', model=model, show=False)
#     generate_output(output_name='outputs/img'+str(index)+'.txt', values=values)



# this is for single view of image

# main_process('smaller_images/img36.png', model=model, show=True)
main_process('images_from_phone/img4.jpg', model=model, show=True)


# statistics()

#statistics_for_one_file('img36.txt', stats=True)