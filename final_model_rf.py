import os
import cv2
import numpy as np
import threading
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from picamera2 import Picamera2
import imutils
from matplotlib import pyplot as plt
import subprocess
import time

# Path to the dataset
dataset_path = "img"

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare training data
labels = []
faces = []

# Iterate through the dataset to read images and extract faces
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_rects = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in face_rects:
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (100, 100))
                face = cv2.equalizeHist(face)
                faces.append(face.flatten())
                labels.append(person_name)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Initialize the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(faces, labels_encoded)

# Initialize the Picamera2
camera = Picamera2()
camera_config = camera.create_still_configuration(main={"size": (320, 240)})
camera.configure(camera_config)

frame = None
gray = None
face_rects = []

# Lock for thread synchronization
lock = threading.Lock()

# Track last greetings and times
last_greeted_time = time.time()
recognized_names = set()
unknown_greeted_time = time.time()

def process_frame():
    global frame, gray, face_rects
    while True:
        temp_frame = camera.capture_array()
        temp_frame = imutils.resize(temp_frame, width=320)
        temp_gray = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
        temp_face_rects = face_cascade.detectMultiScale(temp_gray, scaleFactor=1.1, minNeighbors=5)

        with lock:
            frame = temp_frame
            gray = temp_gray
            face_rects = temp_face_rects

# Start a thread to process frames
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()

camera.start()  # Start the camera

plt.ion()  # Turn on interactive mode for matplotlib

while True:
    with lock:
        if frame is None:
            continue

        display_frame = frame.copy()
        display_gray = gray.copy()
        display_face_rects = face_rects

    current_time = time.time()
    detected_name = "Unknown"  # Default to Unknown if no face is detected

    for (x, y, w, h) in display_face_rects:
        face = display_gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))
        face_resized = cv2.equalizeHist(face_resized)
        face_resized_flattened = face_resized.flatten()

        # Predict the identity of the face
        prediction_probs = classifier.predict_proba([face_resized_flattened])
        best_match_idx = np.argmax(prediction_probs)
        best_match_prob = prediction_probs[0][best_match_idx]

        if best_match_prob > 0.5:  # Adjusted threshold for known faces
            detected_name = label_encoder.inverse_transform([best_match_idx])[0]

            # Check if this person has been greeted within the last 10 seconds
            if detected_name in recognized_names and (current_time - last_greeted_time < 10):
                continue  # Skip greeting this person
            else:
                recognized_names.add(detected_name)
                last_greeted_time = current_time
                greeting = f"Hello, {detected_name}! Welcome to Nine Pointers"
                subprocess.run(['espeak', greeting])

        else:
            # Check if unknown face should be greeted (every 3 seconds)
            if current_time - unknown_greeted_time >= 3:
                unknown_greeted_time = current_time
                greeting = "Welcome to Nine Pointers"
                subprocess.run(['espeak', greeting])

        # Debugging: print the prediction probabilities and best match probability
        print(f"Prediction probabilities: {prediction_probs}")
        print(f"Best match probability: {best_match_prob}, Name: {detected_name}")

        # Draw a rectangle around the face and label it
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(display_frame, (x, y-35), (x+w, y), (0, 0, 255), cv2.FILLED)
        cv2.putText(display_frame, detected_name, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting frame using matplotlib
    plt.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    plt.title('Video')
    plt.draw()
    plt.pause(0.001)

    # Break the loop on 'q' key press
    if plt.waitforbuttonpress(0.001):
        break

# Release the camera and close the window
camera.stop()
plt.close()
