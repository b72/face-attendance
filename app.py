# TechVidvan Face detection

import cv2
import numpy as np
import face_recognition
import pickle
import tkinter as tk
from PIL import Image, ImageTk
import os
from datetime import datetime
import requests

API_URL = " http://127.0.0.1:8000/incident"  # Replace with your actual API endpoint

# Load encodings and class names from file
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)
    knownEncodes = data["encodings"]
    classNames = data["classNames"]

scale = 0.5
box_multiplier = 1/scale

# Ensure capture folder exists
if not os.path.exists("capture"):
    os.makedirs("capture")

# Tkinter UI setup
root = tk.Tk()
root.title("Face Recognition UI")

# Left: Live feed
live_label = tk.Label(root)
live_label.grid(row=0, column=0, padx=10, pady=10)

# Right: Captured face and info
right_frame = tk.Frame(root)
right_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="n")

# Reference image label
ref_label = tk.Label(root)
ref_label.grid(row=0, column=2, padx=10, pady=10)

cap = cv2.VideoCapture(0)
captured_ids = set()

def update():
    success, img = cap.read()
    if not success:
        root.after(10, update)
        return

    Current_image = cv2.resize(img, (0,0), None, scale, scale)
    Current_image_rgb = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(Current_image_rgb, model='hog')
    face_encodes = face_recognition.face_encodings(Current_image_rgb, face_locations)

    # Clear previous widgets in right_frame
    for widget in right_frame.winfo_children():
        widget.destroy()

    # Draw bounding boxes and labels on the live image
    if face_encodes:
        for idx, (encodeFace, faceLocation) in enumerate(zip(face_encodes, face_locations)):
            matches = face_recognition.compare_faces(knownEncodes, encodeFace, tolerance=0.6)
            faceDis = face_recognition.face_distance(knownEncodes, encodeFace)
            matchIndex = np.argmin(faceDis) if len(faceDis) > 0 else None

            if matchIndex is not None and matches[matchIndex]:
                detected_name = classNames[matchIndex].upper()
                # Make API  detection
                payload = {
                    "emp_id": detected_name,
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                try:
                    response = requests.post(API_URL, json=payload)
                    print(f"API response: {response.status_code} {response.text}")
                except Exception as e:
                    print(f"API call failed: {e}")
            else:
                detected_name = 'Unknown'

            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = int(y1*box_multiplier), int(x2*box_multiplier), int(y2*box_multiplier), int(x1*box_multiplier)

            # Draw bbox and label on live image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-20), (x2, y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, detected_name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)

            # Save captured face
            captured_face_img = img[y1:y2, x1:x2]
            if captured_face_img is not None and detected_name != 'Unknown' and detected_name not in captured_ids:
                capture_path = f"capture/{detected_name}.jpg"
                cv2.imwrite(capture_path, captured_face_img)
                captured_ids.add(detected_name)

            # UI for each detection (right side)
            face_pil = Image.fromarray(cv2.cvtColor(captured_face_img, cv2.COLOR_BGR2RGB))
            face_pil = face_pil.resize((100, 100))
            face_imgtk = ImageTk.PhotoImage(image=face_pil)
            face_label = tk.Label(right_frame, image=face_imgtk)
            face_label.imgtk = face_imgtk
            face_label.grid(row=idx, column=0, padx=5, pady=5)

            info_label = tk.Label(right_frame, text=f"Emp ID: {detected_name}", font=("Arial", 12))
            info_label.grid(row=idx, column=1, padx=5, pady=5)

            # Reference image
            ref_img_path = os.path.join("employees", f"{detected_name.lower()}.jpg")
            if os.path.exists(ref_img_path):
                ref_img = Image.open(ref_img_path).resize((100, 100))
                ref_imgtk = ImageTk.PhotoImage(ref_img)
                ref_label = tk.Label(right_frame, image=ref_imgtk)
                ref_label.imgtk = ref_imgtk
                ref_label.grid(row=idx, column=2, padx=5, pady=5)
            else:
                ref_label = tk.Label(right_frame, text="No ref", font=("Arial", 10))
                ref_label.grid(row=idx, column=2, padx=5, pady=5)
    else:
        info_label = tk.Label(right_frame, text="No face detected", font=("Arial", 12))
        info_label.grid(row=0, column=0, padx=5, pady=5)

    # Show live feed with bounding boxes
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    live_label.imgtk = imgtk
    live_label.configure(image=imgtk)

    root.after(10, update)

update()
root.mainloop()
cap.release()
cv2.destroyAllWindows()