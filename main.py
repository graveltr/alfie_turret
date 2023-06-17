import numpy as np 
import cv2 

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")

while True:
    ret, frame = cap.read()
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_grayscale, 1.3, 5)
    cats = cat_cascade.detectMultiScale(frame_grayscale, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

    for (x,y,w,h) in cats:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()