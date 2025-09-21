import cv2
import numpy as np
import face_recognition
import pickle
import os

# Define the path for training images

path = 'employees'
encodings = []
classNames = []

# Reading the training images and classes and storing into the corresponsing lists
for emp_id in os.listdir(path):
    emp_folder = os.path.join(path, emp_id)
    if os.path.isdir(emp_folder):
        for img_file in os.listdir(emp_folder):
            img_path = os.path.join(emp_folder, img_file)
            image = cv2.imread(img_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_encodings(image_rgb)
                if faces:
                    encodings.append(faces[0])
                    classNames.append(emp_id)

print('Encoding Complete')
# Save encodings and class names to a file
with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": encodings, "classNames": classNames}, f)

print("Encodings saved to encodings.pkl")