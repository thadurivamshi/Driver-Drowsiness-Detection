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
import face_recognition
import statistics



#Start webcam video capture
video_capture = cv2.VideoCapture(0)
who_is_driving_status="Camera started"

def capture_screenshot():

    # Check if the camera was opened successfully
    if not video_capture.isOpened():
        print("Unable to open camera")
        return

    # Read the first frame of the video stream
    ret, frame = video_capture.read()

    # Wait for a key press and take a screenshot if 's' is pressed
    cv2.imwrite('past.jpg', frame)

# First capture
capture_screenshot()

def check_if_person_changed() :
    image1 = face_recognition.load_image_file("past.jpg")

    # Read the first frame of the video stream
    ret, frame = video_capture.read()

    cv2.imwrite('past.jpg', frame)
    image2 = face_recognition.load_image_file("past.jpg")

    face_locations1 = face_recognition.face_locations(image1)
    face_locations2 = face_recognition.face_locations(image2)

    if face_locations2 :
        if face_locations1 :
            # Get the face encodings (feature vectors) for each image
            encoding1 = face_recognition.face_encodings(image1)[0]
            encoding2 = face_recognition.face_encodings(image2)[0]

            # Compare the face encodings to see if they match
            results = face_recognition.compare_faces([encoding1], encoding2)
            print(results)

            if results[0] == True :
                return "Keep Driving"
            
            else :
                return "Person change detected."

        else :
            return  "Driver not found"
        
    else :
        return "Driver not found in capture area."









font = cv2.FONT_HERSHEY_COMPLEX_SMALL

#start the counter
start_time = time.time()
last_screenshot_time=time.time()
last_time_median_found=time.time()
aspect_ratio_archive_for_median = []
past_median=-1


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
    # A = distance.euclidean(eye[1], eye[5])
    A=np.linalg.norm(eye[1]- eye[5])
    # B = distance.euclidean(eye[2], eye[4])
    B=np.linalg.norm(eye[2]- eye[4])
    # C = distance.euclidean(eye[0], eye[3])
    C=np.linalg.norm(eye[0]- eye[3])
    ear = (A+B) / (2*C)
    return ear

def distace_between_lips(lip_high,lip_low):
    # A=distance.euclidean(lip_high, lip_low)
    A=np.linalg.norm(lip_high[1]-lip_low[1])
    return A

#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye   
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']     
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']    



#Give some time for camera to initialize(not required)
time.sleep(2)

while(True):
    if(time.time()-last_screenshot_time>15):
        last_screenshot_time=time.time()
        who_is_driving_status=  check_if_person_changed()
        print(who_is_driving_status)
        if(who_is_driving_status=="Person change detected."):
            with open("log.txt", "a") as f:
                f.write("Person change detected"+str(time.time()-start_time)+"\n")
            start_time=time.time()

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
        if(time.time()-last_time_median_found>5*60):
            last_time_median_found=0
            median = statistics.median(aspect_ratio_archive_for_median)
            aspect_ratio_archive_for_median.clear()
            if(past_median>-1):
                if(median<past_median*0.8):
                    print("try to sleep ASAP")
            
            past_median=median
            

        

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        temp=shape

        #Get array of coordinates of leftEye and rightEye
        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio  = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
        aspect_ratio_archive_for_median.append(eyeAspectRatio)

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # print(shape)
        # print(" \n")
        # # Calculate distance between top and bottom lip landmarks
        # top_lip = temp[50][1]
        # bottom_lip = temp[57][1]
        distance=distace_between_lips(shape[50],shape[57])
        # distance = (top_lip - bottom_lip)
        
        # # Check if distance is greater than a threshold (indicating the mouth is open)
        if distance > 30:
            print("Yawning detected!")
            cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)

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
        cv2.putText(frame, who_is_driving_status ,(0,10), font, 1,(255,255,255),1,cv2.LINE_AA)

    #Show video feed
    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()



