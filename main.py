import cv2
import dlib
import align_face as af
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from keras.utils import to_categorical
K.set_image_data_format('channels_first')
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import database as db
from sklearn.svm import LinearSVC
import pickle
import face_recognition as fr

np.set_printoptions(threshold=np.nan)


img_name = input('Enter image file name: ')
image = cv2.imread('test/' + str(img_name))

fr.test(image)