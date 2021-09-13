#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import mediapipe as mp
import cv2 as cv
import time
import os
import sys


# In[4]:


def usage():
    print("""This script is to extract landmark from video and save it as numpy array for traning step  
             all you need is to collect al data and then call this script.
             you also need to create folder npy_data
             
             exp:
                $python extract_landMark.py
    
    """)


# In[5]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh, rh])

def extract_nmpy_array(action_name):
    try:
        folders=os.path.join(DATA_PATH,'data',action_name)
    except FileNotFoundError:
        usage()
        print("You need to collect data first.")
        sys.exit()
    try:
        os.mkdir(os.path.join(DATA_PATH,'npy_data',action_name))
    except FileNotFoundError:
            print('No such file or directory: npy_data')
            usage()
            sys.exit()
    for folder in os.listdir(folders):
        try:
            os.mkdir(os.path.join(DATA_PATH,'npy_data',action_name,folder))
        except:
            pass
        folder_path=os.path.join(DATA_PATH,'data',action_name,folder)
        for imag in os.listdir(folder_path):
            image=os.path.join(DATA_PATH,'data',action_name,folder,imag)
            image=cv.imread(image)
            RGB=cv.cvtColor(image,cv.COLOR_BGR2RGB)
            results=holistic.process(RGB)
            keyPoint=extract_keypoints(results)
            npy_path=os.path.join(DATA_PATH,'npy_data',action_name,folder,imag[:-3]+'npy')
            np.save(npy_path,keyPoint)


# In[6]:


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
holistic=mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# In[7]:


DATA_PATH=os.getcwd()

if __name__=="__main__":
    if not len(sys.argv[1:]):
        for i in os.listdir(os.path.join(DATA_PATH,'data')):
            print("extract data from folder: {}...".format(i))
            extract_nmpy_array(i)
    else:
        usage()


# In[5]:





# In[ ]:




