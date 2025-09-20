# Face Recognition Attendance System

This project is a real-time face recognition attendance system using Python, OpenCV, face_recognition, Tkinter, and FastAPI. It detects faces from a webcam, matches them against employee images, displays results in a UI, saves captured faces, and logs detection incidents to a SQLite database via an API.

## Features

- **Live webcam feed** with bounding boxes and employee ID labels.
- **Employee matching** using images in the `employees` folder.
- **UI**: Left side shows live video, right side shows detected faces, employee IDs, and reference images.
- **First-time detection**: Captures and saves the face image, sends an API request with `emp_id` and `date_time`.
- **FastAPI backend**: Receives detection incidents and stores them in a SQLite database.

## Folder Structure

```
face_recog/
│
├── app.py                # Main UI and detection app
├── create_encoding.py    # Script to generate face encodings from employee images
├── api_server.py         # FastAPI backend for incident logging
├── encodings.pkl         # Pickled face encodings and employee IDs
├── capture/              # Saved detected face images
├── employees/            # Employee reference images (e.g., emp001.jpg, emp002.jpg)
├── incidents.db          # SQLite database for detection logs
├── requirements.txt      # Python dependencies
```

## Setup Instructions

### 1. Clone the repository and prepare folders

- Place employee images (e.g., `emp001.jpg`, `emp002.jpg`) in the `employees` folder.

### 2. Create and activate a Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** You may need to install Visual C++ Build Tools for `dlib`/`face_recognition` on Windows.

### 4. Generate face encodings

```powershell
python create_encoding.py
```

### 5. Start the FastAPI backend

```powershell
uvicorn api_server:app --reload
```

### 6. Run the main app

```powershell
python app.py
```

## API

- **Endpoint:** `POST /incident`
- **Payload:**  
  ```json
  {
    "emp_id": "EMPLOYEE_ID",
    "date_time": "YYYY-MM-DD HH:MM:SS"
  }
  ```
- **Database:** Incidents are stored in `incidents.db`.

## Customization

- Add more employee images to the `employees` folder.
- Adjust detection parameters (`scale`, `tolerance`) in `app.py` for performance/accuracy.
- Modify UI layout in `app.py` as needed.

## License

MIT

## Credits

- [face_recognition](https://github.com/ageitgey/face_recognition)
- [OpenCV](https://opencv.org/)
- [FastAPI](https://fastapi.tiangolo.com/)