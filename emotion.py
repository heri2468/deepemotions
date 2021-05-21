import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot Open Webcam")
    
while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame,actions = ['emotion'])
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade .detectMultiScale(
        gray,     
        1.1,
        4
    )
    

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame,
            result['dominant_emotion'],
            (50,50),
            font,3,
            (0,255,0),
            10,
            cv2.FILLED
           )
    
    cv2.imshow('Original_Video',frame)

    k = cv2.waitKey(30) & 0xFF
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
