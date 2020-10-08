# Load the images
import os
import glob
import numpy as np
import os.path as path
# from scipy import misc
import imageio
# 1. create  numpy array of images live... by append?...
import cv2
from tqdm import tqdm
import argparse
from yolo import YOLO
import matplotlib.pyplot as plt
from collections.abc import Sequence
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# from numba import jit
# first with one video....
# do this for each video?..... or make two array sthen join them...just for one video
# initate empty 
# initialise better size?...

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

# @jit(nopython=True)
def dataCreator(vidname,videoLabel):
    # number of frames for each class wanted to use for data.
    # NEED WAY TO ensure enough frames ....

    frameNumber = 150
    # capture N frames for use for each class...
    vidcap = cv2.VideoCapture(vidname)

    oneVideosImages = [] 
    oneVideosLabels = []
    success = True
    success,image = vidcap.read()

    dataCreatedNumber = 0
    while success:
      # filename passed not written....
      width, height, inference_time, results = yolo.inference(image)
      halfWidth = 150
      for detection in results:
          croppedImage = image
          id, name, confidence, x, y, w, h = detection
          # decide which hands to use for training
          # learn how to implem,ent detector on larger. image....
          # later turn this into function
          if confidence >0.7 and w < 200 and h < 200:
              # print("conditions passed")    
              cx = x + (w / 2)
              cy = y + (h / 2)
              y1 = int(cy - halfWidth)
              y2 = int(cy + halfWidth)
              x1 = int(cx - halfWidth)
              x2 = int(cx + halfWidth)
              croppedImage = image[y1:y2, x1:x2]
              croppedImage = croppedImage/255
              # print(croppedImage.shape)
              if croppedImage.shape == (300, 300, 3):
                dataCreatedNumber = dataCreatedNumber+1
                print(dataCreatedNumber)
                oneVideosImages.append(croppedImage)
                oneVideosLabels.append(videoLabel)
              # this ensures equly sized dataset 
                if dataCreatedNumber > frameNumber:
                  return oneVideosImages, oneVideosLabels
                  
      success,image = vidcap.read()
      # cv2.imshow("preview", croppedImage)
      # # rval, frame = vc.read()

      # key = cv2.waitKey(20)
      # # if key == 27:  # exit on ESC
      # #   break
    return oneVideosImages, oneVideosLabels
# maybe build untill there are enough frames to ensure equal amounts ...

def arrayCreator():
    videos = ['videos/touch_iryamim.mov', 'videos/non_touch_iryamim.mov']
    images = []
    labels = []
    for videoLabel ,vidname in enumerate(videos):
        oneVideosImages,oneVideosLabels = dataCreator(vidname,int(videoLabel))
        images.extend(oneVideosImages)
        labels.extend(oneVideosLabels)
        print("video frames captured  for one video")
    return images, labels

def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )

    # recurse
    shape = get_shape(lst[0], shape)

    return shape

def dataPrep():    

    images, labels = arrayCreator()
    n_images = len(images)
    # images, labels = zip(*labledImages)

    # get shape of lists..
    print(get_shape(images, shape=()))


    images = np.array(images)
    # print(images)
    labels = np.array(labels)
    print(labels)
    # ISSUE HEREEEEE
    # print out numpy aray in colab to see..
    image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
    # image_size = np.asarray([300, 300, 3])
    # Split into test and training sets
    TRAIN_TEST_SPLIT = 0.8
    # Split at the given index
    split_index1 = int(TRAIN_TEST_SPLIT * n_images)
    shuffled_indices = np.random.permutation(n_images)
    train_indices = shuffled_indices[0:split_index1]
    split_index2 = int(0.9 * n_images)
    # 10 perfent for test and 10 percent for validation.
    test_indices = shuffled_indices[split_index1:split_index2]
    valid_indicies = shuffled_indices[split_index2:]

    # Split at test and valid at the given index create validation set!
    # Split the images and the labels
    x_train = images[train_indices, :, :, :]
    y_train = labels[train_indices]

    x_test = images[test_indices, :, :, :]
    y_test = labels[test_indices]

    x_valid = images[valid_indicies, :, :, :]
    y_valid = labels[valid_indicies]


    return x_train,y_train,x_test,y_test,x_valid,y_valid , image_size
    # Check input data batchs!
x_train,y_train,x_test,y_test,x_valid,y_valid,image_size = dataPrep()    


## MODEL CREATION ##
# Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
from matplotlib import pyplot as plt

# Model Hyperparamaters
N_LAYERS = 4

def cnn(size, n_layers):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN
    # Define model hyperparamters
    MIN_NEURONS = 20
    MAX_NEURONS = 150
    KERNEL = (3, 3)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    nuerons = nuerons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape))
            # idea to reduce perameters.. may need to change due to size of photos
            model.add(MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(Conv2D(nuerons[i], KERNEL))

        model.add(Activation('relu'))

    # Add max pooling layer with dropout
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print a summary of the model 
    model.summary()

    return model

# Instantiate the model
model = cnn(size=image_size, n_layers=N_LAYERS)
## MODEL TRAINING ##
# Training Hyperparamters
EPOCHS = 10
BATCH_SIZE = 50

# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, mode='auto')
callbacks = [early_stopping]
# Train the model
history = model.fit(x_train, y_train ,epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,validation_data=(x_test, y_test))
## MODEL EVALUATION ##
# Make a prediction on the test set
test_predictions = model.predict(x_test)
test_predictions = np.round(test_predictions)

# Report the accuracy wish to report this for each minibatch or epoc....
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy: " + str(accuracy))


print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# model.save("currentModel")
# print("Saved model to disk")
# from numpy import loadtxt
# from keras.models import load_model
# # load model
# # model = load_model('drive/My Drive/model.h1')
# # summarize model.
# # model.summary()

# ## MODEL VALIDATION TEST ##
# validation_predictions = model.predict(x_valid)
# validation_predictions = np.round(validation_predictions)

# # Report the accuracy wish to report this for each minibatch or epoc....
# accuracy = accuracy_score(y_valid, validation_predictions)
# print("Accuracy: " + str(accuracy))
import cv2
import os

video_name = 'labledVids/generated_video.avi'
video = cv2.VideoWriter(video_name, 0, 1,(1280,720))

print("starting webcam for trail...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
halfWidth = 150
while rval:
    width, height, inference_time, results = yolo.inference(frame)
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        if confidence >0.7 and w < 200 and h < 200:
              cx = x + (w / 2)
              cy = y + (h / 2)
              y1 = cy - halfWidth
              y2 = cy + halfWidth
              x1 = cx - halfWidth
              x2 = cx + halfWidth
              croppedImage = frame[int(y1):int(y2), int(x1):int(x2)]
              # grey = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
              # run crop though cnn...
              croppedImage = croppedImage/255
              if croppedImage.shape == (300, 300, 3):
                # make compatable to be read..x
                croppedImage = np.expand_dims(croppedImage, axis=0)
                prediction = model.predict(croppedImage)
              # run frame through new trained model..show touching and non touching...
              # 60 percent as confidence for now..
              if prediction > 0.6: 
                color = (0, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # text = "%s (%s)" % (name, round(confidence, 2))
              if prediction < 0.6:
                color = (0, 100, 100)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # text = "%s (%s)" % (name, round(confidence, 2))
              # display bounding box as red if touching...
        # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    # 0.5, color, 2)
    cv2.imshow("preview", frame)
    video.write(frame)
    # cv2.destroyAllWindows()
    # save it!
    # cv2.imwrite("frame%d.jpg" % count, image)
    # save the frame to a video... to compare inputs and ooutputs for quick development..
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
video.release()
cv2.destroyWindow("preview")
vc.release()