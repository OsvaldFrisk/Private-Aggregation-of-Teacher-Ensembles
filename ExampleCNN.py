from PATE.Teacher import Teacher
from PATE.kerastf.MnistModel import load_mnist, CNN
from sklearn.metrics import accuracy_score
import numpy as np

X_train, X_validation, y_train, y_validation = load_mnist()
# y_validation = y_validation.argmax(axis=1) + 1


# # teachers = Teacher(CNN, epochs=2, n_teachers=10, verbose=True, n_classes=10)
# # teachers.fit(X_train, y_train)
# # predicted = teachers.predict(X_validation)

# model = CNN()
# model.fit(X_train, y_train)
# predicted = model.predict(X_validation)

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
input_shape = (28, 28, 1)

from tensorflow import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

# convert class vectors to binary class matrices
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=1,
          verbose=1)

predicted = model.predict(X_validation)
class_preds = predicted.argmax(axis=1)
y_test = y_validation.argmax(axis=1)

print(f"Accuracy: {round(accuracy_score(y_test, class_preds)*100, 1)}%")
