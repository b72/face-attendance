# TechVidvan Face detection

import cv2
import numpy as np
import face_recognition
import pickle

# Load encodings and class names from file
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)
    knownEncodes = data["encodings"]
    classNames = data["classNames"]

print(classNames)

scale = 0.25
box_multiplier = 1/scale

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    Current_image = cv2.resize(img, (0,0), None, scale, scale)
    Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(Current_image, model='hoq')
    face_encodes = face_recognition.face_encodings(Current_image, face_locations)

    for encodeFace, faceLocation in zip(face_encodes, face_locations):
        matches = face_recognition.compare_faces(knownEncodes, encodeFace, tolerance=0.6)
        faceDis = face_recognition.face_distance(knownEncodes, encodeFace)
        matchIndex = np.argmin(faceDis) if len(faceDis) > 0 else None

        if matchIndex is not None and matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = 'Unknown'

        y1, x2, y2, x1 = faceLocation
        y1, x2, y2, x1 = int(y1*box_multiplier), int(x2*box_multiplier), int(y2*box_multiplier), int(x1*box_multiplier)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.rectangle(img, (x1, y2-20), (x2, y2), (0,255,0), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()