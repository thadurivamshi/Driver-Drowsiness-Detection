'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

#Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import time
import dlib
import cv2
from playsound import playsound
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

#start the counter
start_time = time.time()


#Initialize Pygame and load music
pygame.mixer.init()

# set the alarm duration in seconds
alarm_duration = 3
    
# load the alarm sound file
alarm_sound = pygame.mixer.Sound('./audio/alert.wav')
# pygame.mixer.music.load('audio/alert.wav')

#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

#COunts no. of consecutuve frames below threshold value
COUNTER = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A+B) / (2*C)
    return ear

#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye   
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']     
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']    

#Start webcam video capture
video_capture = cv2.VideoCapture(0)

#Give some time for camera to initialize(not required)
time.sleep(2)

while(True):
    #Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height,width = frame.shape[:2] 

    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Detect facial points
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio  = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Detect if eye aspect ratio is less than threshold
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                alarm_sound.play()
                cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                time.sleep(0.5)
                alarm_sound.stop()
                # danger_alarm()
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
            
            else:
                COUNTER += 1
                cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                
        else:
            # pygame.mixer.music.stop()
            alarm_sound.stop()
            COUNTER = 0
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        cv2.putText(frame,'Score:'+str(COUNTER),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        end_time = time.time()
        duration = int(end_time - start_time)

        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60
        cv2.putText(frame,'Duration '+str(hours)+":"+str(minutes)+":"+str(seconds),(width-200,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    #Show video feed
    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()



