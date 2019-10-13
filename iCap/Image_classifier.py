#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
import keras
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import SVG,Image, display
from keras.utils.vis_utils import model_to_dot


# In[2]:

def Image_classifer():
    img_width, img_height = 224, 224
    # In[3]:
    train_data_dir = r"C:\Users\deepu\Documents\Digithon\Claims\WindShieldDamageDataSet\Train"
    validation_data_dir = r"C:\Users\deepu\Documents\Digithon\Claims\WindShieldDamageDataSet\Test"
    nb_train_samples = 400
    nb_validation_samples = 100
    epochs = 50
    batch_size = 16
    num_classes = 2
    # In[4]:

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    # In[5]:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size =(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size =(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size =(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    model.compile(loss ='binary_crossentropy',
                        optimizer ='rmsprop',
                      metrics =['accuracy'])
    # load model
    model.load_weights("windshield_damage.h5")
    # image path
    img_path = r"C:\Users\deepu\Desktop\iCAP\Work Directory\Model Input\Accidental Damage.jpg"
    # load a single image
    new_image = load_image(img_path)
    # check prediction
    pred = model.predict_classes(new_image)
    # In[13]:
    #display(Image(filename=img_path))
    if pred[0]==0:
        print("Windshield Damaged")
        return("Windshield Damaged")
    else:
        print("Windshield Not Damaged")
        return ("Windshield Not Damaged")


    # In[ ]:
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    print(img_tensor.shape)
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    return img_tensor

Image_classifer()

