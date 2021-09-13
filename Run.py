#!/usr/bin/env python
# coding: utf-8

# Import  Dependencies

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("model", help="The model name we want to use it for prediction.")

parser.add_argument("label", help="the dictionary file contains the label number.")

def usage():
	print("""
		This script is using for real time test you should pass two argument.
		the first one is the model the file we want use for predection
		the second one is for dictionary contains the action name and it's id.
		exp:
			$python Run.py model_v1.h5 label_map.pkl 
	""")

# Function wee need

args=parser.parse_args()

print("import dependencies....")
import cv2 as cv
import numpy as np
import os
import time
import mediapipe as mp
import pickle
import tensorflow.keras as keras 
from multiprocessing import Pool

print("Build Function...")

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


# import model and label file
print("load model...")
if args.label:
	with open(args.label,'rb') as f:
	    action_name=pickle.load(f)
	key,value=zip(*action_name.items())
	Null=len(key)
	actions=dict(zip(value,key))
	actions[Null]="None"

else:
	usage()
	sys.exit()

if args.model:
	model=keras.models.load_model(args.model)
else:
	usage()
	sys.exit()


#creat holistic opject that give us opertunity to make the program paraller
holistic=mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
holistic_process=holistic.process


print("start....")

sequence = []
sentence = []
threshold = 0.6
res=[0]*6

cap = cv.VideoCapture(0)
key=Null

while True:
    
    start=time.time()

    # Read feed
    ret, frame = cap.read()

    # Make detections
    RGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    result=holistic_process(RGB)
    key_point=extract_keypoints(result)
    sequence.append(key_point)
    

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        key=np.argmax(res)
        if res[key]<=0.7:
        	  key=Null
        sequence.clear()


	
    
    cv.putText(frame, "activity: {}".format(actions[key]), (20, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)	
    end=1/(time.time()-start)
    cv.putText(frame, str(end), (30,350), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # Show to screen
    
    cv.imshow('OpenCV', frame)

    # Break gracefully
    if cv.waitKey(10) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break

cap.release()






