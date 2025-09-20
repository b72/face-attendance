import cv2
import numpy as np
import face_recognition
import pickle
import os

# Define the path for training images

path = 'employees'

images = []
classNames = []

# Reading the training images and classes and storing into the corresponsing lists
for img in os.listdir(path):
    image = cv2.imread(f'{path}/{img}')
    images.append(image)
    classNames.append(os.path.splitext(img)[0])

print(classNames)

# Function for Find the encoded data of the imput image
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        print(encode)
        encodeList.append(encode)
    return encodeList

# Find encodings of training images

knownEncodes = findEncodings(images)
print('Encoding Complete')
# Save encodings and class names to a file
with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": knownEncodes, "classNames": classNames}, f)

print("Encodings saved to encodings.pkl")