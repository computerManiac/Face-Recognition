from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import numpy as np
import cv2
import pickle
from face_recognition import *
from fr_utils import *
from inception_blocks_v2 import *

fmodel = faceRecoModel(input_shape=(3,96,96))
load_weights_from_FaceNet(fmodel)

data = {}

names = ['danielle','younes','tian','andrew','kian','dan','sebastiano','bertrand','kevin','felix','benoit','arnaud']

index=0
for i in names:
    image = cv2.imread('/images/' + i + '.jpg',1)
    img = image[...,::-1]
    img = np.array(img)
    img = np.around(np.transpose(img,(2,0,1))/255,decimals=12)
    data['name'][index] = i
    data['emb'][index] = get_encodings(img,fmodel)

fname = open('images.pickle','wb')
pickle.dump(data,fname)
