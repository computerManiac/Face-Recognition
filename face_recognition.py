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

np.set_printoptions(threshold=np.nan)

face_model = faceRecoModel(input_shape=(3,96,96))
svc = LinearSVC()
train_images = 'database/images'
mode = "record"

#triplet loss incase you want to train the model
def triplet_loss(_,y_pred,alpha=2.0):

	anchor,pos,neg = y_pred[0], y_pred[1], y_pred[2]

	pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,pos)), axis=-1)
	neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,neg)), axis=-1)
	loss = tf.subtract(pos_dist,neg_dist) + alpha

	total_loss = tf.reduce_sum(tf.maximum(loss,0.0))

	return total_loss

face_model.compile(loss=triplet_loss, optimizer='adam', metrics=['accuracy'])
load_weights_from_FaceNet(face_model)

def get_encodings(model,image,get_box=False):

	face,box = af.align(image)
	emb = []
	
	if len(face) > 1:
		for f in face:
			f = cv2.resize(f, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
			#f = f[...,::-1]
			f = np.array([np.around(np.transpose(f,[2,0,1])/255.0,decimals=12)])
			emb.append(model.predict_on_batch(f))
	else:
		face = cv2.resize(face[0], dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
		face = np.array([np.around(np.transpose(face,[2,0,1])/255.0,decimals=12)])
		emb.append(model.predict_on_batch(face))

	if get_box:
		return emb,box

	return emb

def compare_img(imgA,imgB,model):

	embA = get_encodings(model,imgA)
	embB = get_encodings(model,imgB)

	dist = np.linalg.norm(embA - embB)
	return dist

def train():

	names = {}
	image_files = [f for f in listdir(train_images) if isfile(join(train_images,f))]

	data = []
	
	for image_name in image_files:
		img = cv2.imread('database/images/' + str(image_name))
		emb = get_encodings(face_model,img)
		#name = input("({}/{}) Enter name for {} : ".format(image_files.index(image_name)+1,len(image_files),image_name))
		name = "Robert Downey"
		data.append([name,emb[0]])

	connection = db.db_connect()
	#db.db_create(connection)
	
	for d in data:
		db.db_insert(connection,str(d[0]),str(d[1]))

	'''data = db.db_getData(connection)

	get_names = db.get_names(connection)

	for n in get_names:
		names[n] = get_names.index(n)

	train_x = []
	train_y = []

	for d in data:
		b = np.array([float(x) for x in d[1][3:-2].split(' ') if x!= ''])
		train_x.append(b)
		train_y.append(names[str(d[0])])

	svc.fit(train_x,train_y)

	with open('svm_model.pickle','wb') as f:
		pickle.dump(svc,f)'''

	connection.close()

def test(image):

	with open('svm_model.pickle','rb') as f:
		svm = pickle.load(f)

	face_emb,boxes = get_encodings(face_model,image,get_box=True)
	face_encodings = [svm.predict(np.array(fe)) for fe in face_emb]

	connection = db.db_connect()
	get_names = db.get_names(connection)

	names = []
	for f in face_encodings:
		#names.append(get_names[f[0]])
		print(get_names[f[0]])

	'''
	win = dlib.image_window()
	win.set_image(image)

	for b in boxes:
		win.add_overlay(b)
		print("Face {} is {}".format(boxes.index(b)+1,names[boxes.index(b)]))'''

def train_svm(iters=10):
	names = {}
	connection = db.db_connect()
	with open('svm_model.pickle','rb') as f:
		svm = pickle.load(f)

	data = db.db_getData(connection)

	get_names = db.get_names(connection)

	for n in get_names:
		names[n] = get_names.index(n)

	train_x = []
	train_y = []

	for d in data:
		b = np.array([float(x) for x in d[1][3:-2].split(' ') if x!= ''])
		train_x.append(b)
		train_y.append(names[str(d[0])])

	i = 1
	while (i <= iters):
		print('Done with ({}/{})'.format(i,iters))
		svm.fit(train_x,train_y)
		i += 1

	with open('svm_model.pickle','wb') as f:
		pickle.dump(svm,f)

	print("Done Saving")

	connection.close()


if __name__ == "__main__":
	c = int(input('1.Insert 2.Train SVM : '))

	if c==1:
		train()
	else:
		iterations = input('Enter no. of Iterations: ')
		train_svm(int(iterations))