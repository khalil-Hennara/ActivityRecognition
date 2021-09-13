#!/usr/bin/env python
# coding: utf-8
import sys
import time
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-n","--name", help="if we want re-train")

parser.add_argument("-T","--done", help="if we want load dictionary and data",type=int)

parser.add_argument("-e","--epoch", help="the number of epoch to train by default is 60",type=int)

parser.add_argument("-b","--batch", help="the batch size by default is 64",type=int)

parser.add_argument("-m","--model", help="the model name we want to re-train")

args=parser.parse_args()
	
def usage():
    print("""
    This script it's for training model.
    you need to run this script with at least one parameter the model_name you want to save the wight with.
    exp:
       $python train.py model_v1
    
    if you wont re train you should second parameter which is T that mean don't extract data agine just readit from the X_data.npy file 
    exp:
       $python train.py model_v2 T
       
       you can change the strucher of model as you want just open the file and go to the model section line 
       
       you can change the hiper parameter by passing as argument to the script
       exp:
       	
    """)
	

import numpy as np
import tensorflow.keras as keras
import pickle
import os 



def extract_array(path,label_map):
  action_names=os.listdir(path)
  data=[]
  label=[]
  for action in action_names:
    print(action)
    for folder in os.listdir(os.path.join(path,action)):
      sequnce=[]
      for frame in sorted(os.listdir(os.path.join(path,action,folder)),key=lambda x: int(x[:x.find('.')])):
        arr=np.load(os.path.join(path,action,folder,frame))
        sequnce.append(arr)
      data.append(sequnce)
      label.append(label_map[action])
  return data,label



def get_run_logdir(model_name):
  root_logdir = os.path.join(os.curdir, "my_logs")
  run_id = time.strftime("run_%Y_%m_%d")
  return os.path.join(root_logdir, run_id,model_name)



print("Reading The action....")

if args.done==1:
	print("read label...")
	with open("label_map.pkl",'rb') as f:
    		label_map=pickle.load(f)	
	print("load data...")
	X_data=np.load("X_data.npy")
	y_data=np.load("label.npy")

else:
	action_names=os.listdir('npy_data')
	label_map=dict(zip(action_names,range(len(action_names))))
	with open('label_map.pkl', 'wb') as f:
	  pickle.dump(label_map, f, pickle.HIGHEST_PROTOCOL)
	
	print("start extract array .....")
	X_data,y_data=extract_array('npy_data',label_map)
	X_data=np.array(X_data)
	print("savaing data so you can use it later")
	np.save("X_data.npy",X_data)
	np.save("label.npy",y_data)
	
classes=len(label_map)
y_cat=keras.utils.to_categorical(y_data,classes)

if args.model:

	model=keras.models.load_model(args.model)
	
else:
	model=keras.models.Sequential([
		     keras.layers.LSTM(128,return_sequences=True,input_shape=(None,X_data.shape[2])),
		     keras.layers.LSTM(256,return_sequences=True),
		     keras.layers.GlobalAveragePooling1D(),
		     keras.layers.Dense(64,activation='relu'),
		     keras.layers.Dropout(0.2),
		     keras.layers.Dense(32,activation='relu'),
		     keras.layers.Dropout(0.1),
		     keras.layers.Dense(y_cat.shape[1],activation='softmax')
	])
	model.summary()
	model.compile(loss='categorical_crossentropy',metrics='Recall')






if __name__=='__main__':
	name="model"
	if args.name:
		name=args.name
	logs=get_run_logdir(name)
	tensorbord_callback=keras.callbacks.TensorBoard(logs,histogram_freq=1)
	#early_stop=keras.callbacks.EarlyStopping(patience=15)
	checkpoint=keras.callbacks.ModelCheckpoint(name+'.h5',save_best_only=True)
	epochs=60
	batch=64
	if args.epoch:
		epochs=args.epoch
	if args.batch:
		batch=args.batch
	model.fit(X_data,y_cat,batch_size=batch,epochs=epochs,validation_split=0.2,
		callbacks=[tensorbord_callback,checkpoint])
	




