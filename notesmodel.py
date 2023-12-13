# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:58:11 2023

@author: prana
"""
from tensorflow.keras.preprocessing. image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import pickle


train=ImageDataGenerator(rescale= 1/255,
                         rotation_range=40,
                         width_shift_range=0.5,
                         height_shift_range=0.4,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True)
validation =ImageDataGenerator(rescale= 1/255,
                         rotation_range=40,
                         width_shift_range=0.5,
                         height_shift_range=0.4,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True)

train_dataset =train.flow_from_directory('Notes/train',
                                       target_size= (200,200),
                                       batch_size = 3,
                                       class_mode = 'categorical')

validation_dataset =validation.flow_from_directory('Notes/validation',
                                       target_size= (200,200),
                                       batch_size = 3,
                                       class_mode = 'categorical')

model = tf.keras.models.Sequential([
              tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(200,200,3)),
              tf.keras.layers.MaxPool2D(2,2),
              tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
              tf.keras.layers.MaxPooling2D(2,2),
              tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
              tf.keras.layers.MaxPooling2D(2,2),
              tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
              tf.keras.layers.MaxPooling2D(2,2),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(512,activation="relu"),
              tf.keras.layers.Dense(4,activation='softmax')
])

model.compile(loss= 'categorical_crossentropy',optimizer = RMSprop(learning_rate=0.001),metrics= ['accuracy'])

model.fit(train_dataset,epochs=30,validation_data=validation_dataset)

dir_path = "/content/images/test"
img = image.load_img("Notes/validation/INDIA10NEW/INDIA10NEW_163.jpg",target_size=(200,200))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])
val=model.predict(images)
max_value = np.argmax(val)
print(val,max_value)
if max_value==0:
   print("10Rs Note")
elif max_value==1:
   print("20Rs Rs Note")
elif max_value==2:
   print("500Rs Rs Note")
elif max_value==3:
   print("2000Rs Rs Note")
   
with open("notesmodel.pkl",'wb') as files:
  pickle.dump(model,files)
   


