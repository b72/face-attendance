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
import threading
import subprocess
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

API_URL = "http://127.0.0.1:8000/incident"  # Replace with your actual API endpoint

# Load encodings and class names from file
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)
    knownEncodes = data["encodings"]
    classNames = data["classNames"]

print(f'Known faces loaded: {len(classNames)}')
print('Encodings Loaded')

scale = 0.25
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

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # Default to webcam if not set

# If VIDEO_SOURCE is a digit, use as webcam index, else use as URL
if VIDEO_SOURCE.isdigit():
    cap = cv2.VideoCapture(int(VIDEO_SOURCE))
else:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

captured_ids = set()
frame = None

def grab_frames():
    global frame
    while True:
        success, img = cap.read()
        if success:
            frame = img

# Start thread
threading.Thread(target=grab_frames, daemon=True).start()

def update():
    global frame
    if frame is None:
        root.after(10, update)
        return

    img = frame.copy()
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

            if matchIndex is not None and matches[matchIndex] and faceDis[matchIndex] < 0.5:
                detected_name = classNames[matchIndex].upper()
                # # Make API  detection
                # payload = {
                #     "emp_id": detected_name,
                #     "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # }
                # try:
                #     response = requests.post(API_URL, json=payload)
                #     print(f"API response: {response.status_code} {response.text}")
                # except Exception as e:
                #     print(f"API call failed: {e}")
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

            # Save unknown faces
            if captured_face_img is not None and detected_name == 'Unknown':
                unknown_dir = "unknown"
                if not os.path.exists(unknown_dir):
                    os.makedirs(unknown_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Crop only the face region from the original frame (without any labels)
                face_img = frame[y1:y2, x1:x2]
                unknown_path = os.path.join(unknown_dir, f"unknown_{timestamp}_{idx}.jpg")
                cv2.imwrite(unknown_path, face_img)

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
            emp_folder = os.path.join("employees", detected_name.lower())
            ref_img_path = None
            if os.path.isdir(emp_folder):
                img_files = os.listdir(emp_folder)
                if img_files:
                    ref_img_path = os.path.join(emp_folder, img_files[0])
            if ref_img_path and os.path.exists(ref_img_path):
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
    im_pil = Image.fromarray(img_rgb).resize((800, 600))
    imgtk = ImageTk.PhotoImage(image=im_pil)
    live_label.imgtk = imgtk
    live_label.configure(image=imgtk)

    root.after(10, update)

# Entry and button for capturing employee image
emp_id_var = tk.StringVar()
emp_id_entry = tk.Entry(root, textvariable=emp_id_var, font=("Arial", 12))
emp_id_entry.grid(row=1, column=0, padx=10, pady=5)

def capture_employee_image():
    global frame
    emp_id = emp_id_var.get().strip().lower()
    if frame is not None and emp_id:
        save_dir = os.path.join("employees", emp_id)
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"{timestamp}.jpg")
        
        # Detect face in the current frame
        Current_image = cv2.resize(frame, (0,0), None, scale, scale)
        Current_image_rgb = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(Current_image_rgb, model='hog')

        if face_locations:
            y1, x2, y2, x1 = face_locations[0]
            y1, x2, y2, x1 = int(y1*box_multiplier), int(x2*box_multiplier), int(y2*box_multiplier), int(x1*box_multiplier)
            
            margin = 60
            y1 = max(0, y1 - margin)
            x1 = max(0, x1 - margin)
            y2 = min(frame.shape[0], y2 + margin)
            x2 = min(frame.shape[1], x2 + margin)
            
            face_img = frame[y1:y2, x1:x2]
            cv2.imwrite(save_path, face_img)
            capture_status.config(text=f"Saved employee face image: {save_path}", fg="green")
            # Show captured face in UI
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).resize((150, 150))
            face_imgtk = ImageTk.PhotoImage(image=face_pil)
            capture_face_label.imgtk = face_imgtk
            capture_face_label.configure(image=face_imgtk)
        else:
            capture_status.config(text="No face detected. Try again.", fg="red")
            capture_face_label.configure(image='')

capture_btn = tk.Button(root, text="Capture Employee Image", command=capture_employee_image, font=("Arial", 12))
capture_btn.grid(row=2, column=0, padx=10, pady=5)

def regenerate_encodings():
    regen_status_label.config(text="Regenerating encodings...", fg="blue")
    try:
        subprocess.run(["python", "create_encoding.py"], check=True)
        # Reload encodings
        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)
            global knownEncodes, classNames
            knownEncodes = data["encodings"]
            classNames = data["classNames"]
        employee_count_label.config(text=f"Employees: {len(set(classNames))}")
        regen_status_label.config(text="Encodings regenerated successfully.", fg="green")
    except Exception as e:
        regen_status_label.config(text=f"Error: {e}", fg="red")

# Regenerate button and info frame
regen_frame = tk.Frame(root)
regen_frame.grid(row=3, column=1, padx=10, pady=5, sticky="nw")

regenerate_btn = tk.Button(regen_frame, text="Regenerate Encodings", command=regenerate_encodings, font=("Arial", 12))
regenerate_btn.pack(anchor="w")

employee_count_label = tk.Label(regen_frame, text=f"Employees: {len(set(classNames))}", font=("Arial", 12))
employee_count_label.pack(anchor="w")

regen_status_label = tk.Label(regen_frame, text="", font=("Arial", 12))
regen_status_label.pack(anchor="w")

# Status label for capture feedback
capture_status = tk.Label(root, text="", font=("Arial", 12))
capture_status.grid(row=3, column=0, padx=10, pady=5)

# Label for displaying captured face image
capture_face_label = tk.Label(root)
capture_face_label.grid(row=4, column=0, padx=10, pady=5)

update()
root.mainloop()
cap.release()
cv2.destroyAllWindows()