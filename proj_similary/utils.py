import os
import time
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import skimage.io as io
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# random SEED 고정 함수
def set_seed(num_seed):
    random.seed(num_seed)
    os.environ['PYTHONHASHSEED'] = str(num_seed)
    np.random.seed(num_seed)
    tf.random.set_seed(num_seed)

def resize_img(dir_path):

    target_height = 360
    target_width = 360
    
    file_list = []

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            file_list.append(dir_path + path)
    
    datasets = []
        
    for idx, each_img in enumerate(file_list):
        
        img = io.imread(each_img)
        img_resize = resize(img, (target_height, target_width, 4))
        img_resize = np.reshape(img_resize, (1, target_height, target_width, 4))

        if idx == 0:
            datasets = img_resize
        else:
            datasets = np.concatenate([datasets, img_resize])            

    return datasets

def aug_img(x_data, y_data):
    
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    datagen.fit(x_data)
    aug_data = datagen.flow(x_data, y_data, batch_size=32, subset='training')
    
    return aug_data