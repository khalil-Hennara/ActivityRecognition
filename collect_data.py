#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2 as cv
import time
import os
import sys



def usage():
    print("""This Script for collecting Data you need first to create data folder in the current directory and you also need to pass at least one argument which is The movement name.\n
           
          exp:
             $python collect_data left
          this well create folder left and but all video inside 
          
          the second parameter is the number of video to collect by default is 200
          
          exp:
              $python collect_data left 100
              
          the last one the duration of movement in other word how much the length of motion the laptop camera can take
          30 frame in second so the movement duration take one second, you pass the number of frame you want by default it is 30.
          
          exp:
              $python collect_data left 100 23
          
          """)

def collect_data(name_of_action,loop=40,frames=30):
    try:
        os.mkdir(os.path.join('data',name_of_action))
    except FileNotFoundError:
        usage()
        print('No such file or directory: data')
        sys.exit()
        
    DATA_PATH=os.getcwd()+os.path.join('/data',name_of_action)
    
    cap=cv.VideoCapture(0)
    _,frame=cap.read()
    
    time.sleep(5)
    for folder in range(loop):
        os.mkdir(os.path.join('data',name_of_action,str(folder)))
        for index in range(frames):
            _,frame=cap.read()
            #y,x,c=frame.shape
            image=frame.copy()
            if index == 0: 
                    cv.putText(frame, 'STARTING COLLECTION', (120,200), 
                               cv.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv.LINE_AA)
                    cv.putText(frame, 'Number {}'.format(folder), (15,30), 
                               cv.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 1, cv.LINE_AA)
                    # Show to screen
                    cv.imshow('OpenCV Feed', frame)
                    cv.waitKey(500)
            else: 
                cv.putText(frame, 'Number {}'.format(folder), (15,12), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                # Show to screen
                cv.imshow('OpenCV Feed', frame)
                
            npy_path = os.path.join(DATA_PATH, str(folder), str(index)+'.jpg')
            cv.imwrite(npy_path,image)
        
        key=cv.waitKey(20) & 0xFF
        if key==ord('q') or folder>=loop-1:
            cv.destroyAllWindows()
            break

    cap.release()    



if __name__=='__main__':
        if len(sys.argv[1:])==1:
            collect_data(sys.argv[1])
        elif len(sys.argv[1:])==2:
        	   collect_data(sys.argv[1],sys.argv[2])
        
        elif len(sys.argv[1:])==3:
        	   collect_data(sys.argv[1],sys.argv[2],sys.argv[3])
        else:
        	   usage()
        	   
        



