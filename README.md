# Activity Recognition in Real Time.

in this project we try to recognise user activity using MediaPipe and Sequnce model.
First we collect data a bout 3500 video for 7 deffrent move after we collect this vedio we extract the landmark using mediapipe in this way we now don't need to handling video but only the position of the user in the scene and that what make this project real time one.

to handel this type of problem people using CNN for video feauter extraction,let's explain step by step.

### Stack of frame
to recognise activity in the vedio you can't process frame at time but you need to handel a stack of frame in this way you add another feature which is the time. 
becouse the movement happen during time so whene combine some frame to gother we now what happen during those frame.
for more informaion about this object this is some resource

https://www.researchgate.net/publication/323885414_Locomotion_Activity_Recognition_Using_Stacked_Denoising_Autoencoders

https://dl.acm.org/doi/10.1145/3372422.3372434

https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2016/media/files/0516.pdf

### sequnce model
now what happen that to handel this amount of feature you well need time and that make alot of aplication too slow.
the next step it's to take this feature from each frame (as we say we have stack of frame and after we process this frame) and pass it to sequnce model where each frame represent time step now we handel the activity this way it's very effactive but it's slow not becous the sequnce model but becouse the CNN model where we try to analyis the video.


### Solation
using mediapipe to handel feature extraction now we don't need any CNN to analyis the video we just need to extract landmark from each frame and stor it as numpy array stack those array to gother as we say previous and pass to the sequnce model where each array represent time step.

In This project I creat some script to 

# How to use
collect_data,extract_landMark,train,Run

the first one is for collecting data all you need is to run this script is to create folder data then 
to define the action name for example:


```python
$mkdir data
$python collect_data.py left
```

the second one is extract_landMark this one is to extract feature from video we alrady collected I do that for peformance 
 you can use this script like this first create folder npy_data then call the script


```python
$mkdir npy_data
$python extract_landMark.py
```

the train script you can call the --help to see how to use it the simple way is 


```python
$python train.py
```

finaly you can try the model you have train by using Run.py script this you can pass --help
to find how to run 


```python
$python Run.py model_name.h5 label.pkl
```

to stop the programe press q or ctrl+c in the terminal 

I hope to enjoy using this program 

this project is ment to controle box game using movement and the predection of the model I made the hard part and I try to connect the output of the model with game but I don't have apowerful hardware so I coldn't play game and run the program

this link contain model and label file and the data we trained on :https://drive.google.com/drive/folders/1mDJL5PYyzwDXwjQv-OBOHs8VhCVRYbDJ?usp=sharing 
