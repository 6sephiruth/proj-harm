
from utils import *

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# from sklearn.neural_network import MLPClassifier

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# x_train = np.reshape(x_train, (60000, -1))
# x_test = np.reshape(x_test, (10000, -1))

# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# model.fit(x_train, y_train)

# print(model.score(x_test, y_test))

# exit()

# Uncomment this to designate a specific GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '2'

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

# 랜덤 시드 설정
SEED = 0
set_seed(SEED)

x_raw_train = resize_img(f'./similary_img/train/')
y_raw_train = np.array([0, 1, 2, 1, 0, 3, 1, 4, 3, 2, 0, 3, 3, 3, 2, 4, 0, 0, 2, 2, 3, 2, 3])


x_raw_test = resize_img(f'./similary_img/test/')
y_raw_test = np.array([4, 1, 3, 2, 0])

# from sklearn.neural_network import MLPClassifier
# x_train = np.reshape(x_raw_train, (23, -1))
# x_test = np.reshape(x_raw_test, (5, -1))
# model = MLPClassifier(verbose=1)
# model.fit(x_train, y_raw_train)
# print(model.score(x_test, y_raw_test))
# exit()


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_raw_train)
x_data = []
y_data = []

e = 0
for x_batch, y_batch in datagen.flow(x_raw_train, y_raw_train):
    if e == 44:
        break
    if e == 0:
        x_train = x_batch
        y_train = y_batch
    else:
        x_train = np.concatenate([x_train, x_batch])
        y_train = np.concatenate([y_train, y_batch])
    e += 1

e = 0
for x_batch, y_batch in datagen.flow(x_raw_test, y_raw_test):
    if e == 20:
        break
    if e == 0:
        x_test = x_batch
        y_test = y_batch
    else:
        x_test = np.concatenate([x_test, x_batch])
        y_test = np.concatenate([y_test, y_batch])
    e += 1

from sklearn.neural_network import MLPClassifier
print(x_train.shape)
print(x_test.shape)

x_train = np.reshape(x_train, (len(x_train), -1))
x_test = np.reshape(x_test, (len(x_test), -1))

model = MLPClassifier(verbose=1)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

exit()
# for i in range(len(x_data)):
#     plt.imshow(x_data[i]*255)
#     plt.savefig(f'./img/{i}.png')

# from sklearn.neural_network import MLPClassifier

# model = MLPClassifier(random_state=1, max_iter=300).fit(x_raw_train, y_raw_train)



# print(model.predict(x_raw_test))
