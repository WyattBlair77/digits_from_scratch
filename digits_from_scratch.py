# Imports
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.image import decode_jpeg
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
import os

path = "/Users/wyatt_blair/Desktop/digits_from_scratch"

print(os.listdir(path))
import matplotlib.pyplot as plt
import random
from sklearn import ensemble

# Image size and Number of Different Classes
img_rows = 28
img_cols = 28
num_classes = 10


# Gets data ready to be read by the model
def data_prep(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)

    x = raw[:, 1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255

    return out_x, out_y


def rotate(matrix):
    if matrix is None or len(matrix) < 1:
        return
    else:
        if len(matrix) == 1:
            return matrix
        else:
            # solution matrix
            soln = [row[:] for row in matrix]
            # size of matrix
            m = len(matrix[0])

            for x in range(0, m):
                for j in range(0, m):
                    soln[j][m - 1 - x] = matrix[x][j]
            return soln


# EMNIST dataset
file = path + "/emnist-digits-train.csv"
data = np.loadtxt(file, skiprows=1, delimiter=",")
# data = pd.read_csv(file)
seed = 10
np.random.seed(seed)
# Initialize variables to be fed into the model
x, y = data_prep(data, train_size=41999, val_size=18000)

# Construct the model
model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(img_cols, img_rows, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

# Fit the model to the prepped-data
model.fit(x, y, batch_size=200, epochs=10, validation_split=0.4)

# #Test model against other numbers
# test_file = path + "/emnist-digits-test.csv"
# test_data = np.loadtxt(test_file, skiprows = 1, delimiter 0l_size = 4000)

# model.evaluate(test_x, test_y, batch_size = 100)

# Test the model with my own handwritten digits
test_pic = path + "/Numbers_v2/"


def create_test_image_paths(folder_path):
    image_paths = []
    for i in range(1, 10):
        temp = folder_path + str(i) + ".jpg"
        image_paths.append(temp)
    return image_paths


img_rows = 28
img_cols = 28
num_classes = 10

# More imports
from PIL import *
from tensorflow.image import decode_jpeg
from tensorflow.python import reshape
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage import data
from skimage import transform

image_size = 28


# A function designed to turn jpegs into numpy arrays that the model can read
def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    # img_array = img_array.mean(axis = -1, keepdims = 1)
    return preprocess_input(img_array)


print()
count = 0
images = create_test_image_paths(test_pic)
for image in images:
    count += 1
    num = read_and_prep_images([image])
    num = num / 255
    num = num[:].mean(axis=-1, keepdims=1)

    num = np.rot90(num, k=3, axes=(1, 2))
    num = np.flip(num, 2)
    plt.imshow(np.squeeze(num))
    for i in range(len(num)):
        for j in range(len(num[i])):
            for k in range(len(num[i][j])):
                for l in range(len(num[i][j][k])):
                    if num[i][j][k][l] > 0.5:
                        temp = 1 - num[i][j][k][l]
                        num[i][j][k][l] = temp
                    else:
                        temp = num[i][j][k][l]
                        num[i][j][k][l] = 1 - temp

    plt.imshow(np.squeeze(num))

    # Model predicts what the handwritten digit is
    print("Count: ", str(count))
    print("The all knowing computer predicts: ")
    pred = model.predict(num, steps=1)

    print("Your number is ", np.argmax(pred))
    print("Prediction array: ")
    for i in range(0, 10):
        print(i, ": ", pred[0][i])
    print()

num = read_and_prep_images([image])
num = num / 255
num = num[:].mean(axis=-1, keepdims=1)
num = np.rot90(num, k=3, axes=(1, 2))
num = np.flip(num, 2)
plt.imshow(np.squeeze(num))
for i in range(len(num)):
    for j in range(len(num[i])):
        for k in range(len(num[i][j])):
            for l in range(len(num[i][j][k])):
                if num[i][j][k][l] > 0.5:
                    temp = 1 - num[i][j][k][l]
                    num[i][j][k][l] = temp
                else:
                    temp = num[i][j][k][l]
                    num[i][j][k][l] = 1 - temp

plt.imshow(np.squeeze(num))

