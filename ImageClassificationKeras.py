import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np

#Labelling the Data
labels = ['Error','Correct']
img_size = 224

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir,label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path,img))[...,::-1] #Convert BGR to RGB format
                resized_arr = cv2.resize(img_arr,(img_size,img_size))#Reshaping images to preferredsize
                data.append([resized_arr,class_num])
            except Exception as e:
                print(e)
    return np.array(data)

#Now build the data by calling above function
train = get_data('<Add Path of your train dataset folder>')
val = get_data('<Add Path of your test dataset folder>')
print('Shape of the Train data set is: ',len(train))
print('Shape of the Test data set is: ',len(val))

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

#Normalize the data
x_train = np.array(x_train)/255
x_val = np.array(x_val) /255

x_train.reshape(-1,img_size,img_size,1)
y_train = np.array(y_train)

x_val.reshape(-1,img_size,img_size,1)
y_val = np.arr(y_val)

#Augment the Data
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center = False,
    featurewise_std_normialization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 30,
    zoom_range =0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False)

model = Sequential()
model.add(Cov2D(32,3,padding = "same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32,3,padding="same",activation="relu"))
model.add(MaxPool2D)

model.add(Conv2D(64,3,padding="same",activation="relu"))
model.add(MaxPool2D)
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation="softmax"))
print(model.summary())

#Now minimize the loss function
opt = Adam(lr=0.000001)
model.compile(optimizer =opt,loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics = ['accuracy'])

#Now enhance the model by specifying epochs
history=model.fit(x_train,y_train,epochs=10,validation_data=(x_val,y_val))

#Display the Classification Report
predictions = model.predict_clsses(x_val)
predictions=predictions.reshape(1,-1)[0]
print(classification_report(y_val,predictions,target_names=['Error (Class 0)','Correct (Class1)']))

#Now test the image prediction against this model
imageToBeValidated =load_img('<Give the path of image for whom prediction is to be done>')
img = np.array(imageToBeValidated)
img =img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
print('Classification Label has been predicted withclassificcation score: ',label[0][1])



