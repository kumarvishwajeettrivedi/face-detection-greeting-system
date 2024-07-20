from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PEOPLE_FOLDER'] = 'peoples'
app.config['IMG_FOLDER'] = 'img'

# Ensure the 'peoples' and 'img' folders exist
os.makedirs(app.config['PEOPLE_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMG_FOLDER'], exist_ok=True)

# Load the pre-trained model for face detection
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Function to perform non-maxima suppression for overlapping bounding boxes
def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, xx2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

# Function to process uploaded image and extract faces
def process_uploaded_image(image_path, person_name):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjusted confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            width = endX - startX
            height = endY - startY
            aspect_ratio = width / float(height)
            if 0.6 < aspect_ratio < 1.5:  # Adjusted aspect ratio filter
                boxes.append((startX, startY, endX, endY))

    boxes = np.array(boxes)
    boxes = non_max_suppression_fast(boxes, 0.3)

    # Save each detected face in the img directory
    for i, (startX, startY, endX, endY) in enumerate(boxes):
        face = image[startY:endY, startX:endX]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        unique_id = uuid.uuid4()
        filename = f"{person_name}_{unique_id}.jpg"
        
        output_img_dir = os.path.join(app.config['IMG_FOLDER'], person_name.lower())
        os.makedirs(output_img_dir, exist_ok=True)
        face_pil.save(os.path.join(output_img_dir, filename))

        enhance_brightness = ImageEnhance.Brightness(face_pil)
        enhance_contrast = ImageEnhance.Contrast(face_pil)
        enhance_sharpness = ImageEnhance.Sharpness(face_pil)

        for j in range(1, 6):
            enhanced_image = enhance_brightness.enhance(1 + j * 0.1)
            enhanced_image.save(os.path.join(output_img_dir, f"{person_name}_{unique_id}_face_bright{j}.jpg"))

            enhanced_image = enhance_contrast.enhance(1 + j * 0.1)
            enhanced_image.save(os.path.join(output_img_dir, f"{person_name}_{unique_id}_face_contrast{j}.jpg"))

            enhanced_image = enhance_sharpness.enhance(1 + j * 0.1)
            enhanced_image.save(os.path.join(output_img_dir, f"{person_name}_{unique_id}_face_sharp{j}.jpg"))

            blurred_image = face_pil.filter(ImageFilter.GaussianBlur(j * 0.5))
            blurred_image.save(os.path.join(output_img_dir, f"{person_name}_{unique_id}_face_blur{j}.jpg"))

    return filename

# Function to check and sync img folders with peoples folders
def sync_img_folders():
    people_folders = [folder.lower() for folder in os.listdir(app.config['PEOPLE_FOLDER'])]
    img_folders = os.listdir(app.config['IMG_FOLDER'])

    for img_folder in img_folders:
        if img_folder.lower() not in people_folders:
            img_folder_path = os.path.join(app.config['IMG_FOLDER'], img_folder)
            if os.path.isdir(img_folder_path):
                print(f"Deleting folder {img_folder_path}")
                os.rmdir(img_folder_path)

# Function to clean up empty people folders
def cleanup_empty_people_folders():
    people_folders = os.listdir(app.config['PEOPLE_FOLDER'])
    for folder in people_folders:
        folder_path = os.path.join(app.config['PEOPLE_FOLDER'], folder)
        if os.path.isdir(folder_path) and not os.listdir(folder_path):
            print(f"Deleting empty folder {folder_path}")
            os.rmdir(folder_path)

# Route to handle file upload and face processing
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        person_name = request.form.get('person_name')  # Get person's name from form

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and person_name:
            # Normalize person_name to lowercase
            person_name = person_name.lower()
            # Save the uploaded file to the people folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            people_folder_path = os.path.join(app.config['PEOPLE_FOLDER'], person_name)
            os.makedirs(people_folder_path, exist_ok=True)

            filename = f"{timestamp}.jpg"
            file_path = os.path.join(people_folder_path, filename)
            file.save(file_path)

            # Process the uploaded file and save faces to img/{person_name}
            new_image_path = process_uploaded_image(file_path, person_name)

            # Cleanup empty folders and sync img folders
            cleanup_empty_people_folders()
            sync_img_folders()

            # Redirect to root URL after successful upload and processing
            return redirect(url_for('index'))

    # Get list of people for displaying on index page
    people = []
    for person_name in os.listdir(app.config['PEOPLE_FOLDER']):
        person_folder_path = os.path.join(app.config['PEOPLE_FOLDER'], person_name)
        if os.path.isdir(person_folder_path):
            person_images = os.listdir(person_folder_path)
            if person_images:
                first_image = os.path.join(person_name, person_images[0])
                people.append({'name': person_name, 'image': first_image})

    return render_template('upload.html', people=people)

# Route to show individual person's images
@app.route('/person/<person_name>', methods=['GET', 'POST'])
def person_images(person_name):
    person_name = person_name.lower()  # Normalize person_name to lowercase
    person_folder_path = os.path.join(app.config['PEOPLE_FOLDER'], person_name)

    # Check if the directory exists
    if not os.path.exists(person_folder_path):
        flash(f'Person {person_name} not found.')
        return redirect(url_for('index'))

    # List images only if directory exists
    images = [os.path.join(person_name, img) for img in os.listdir(person_folder_path)]
    return render_template('person_images.html', person_name=person_name, images=images)

# Route to delete a single image of a person
@app.route('/delete/<person_name>/<filename>', methods=['POST'])
def delete_image(person_name, filename):
    person_name = person_name.lower()  # Normalize person_name to lowercase
    person_folder_path = os.path.join(app.config['PEOPLE_FOLDER'], person_name)
    file_path = os.path.join(person_folder_path, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Check and sync img folders after deletion
    sync_img_folders()

    # Clean up empty folders after deletion
    cleanup_empty_people_folders()

    return redirect(url_for('person_images', person_name=person_name))

# Route to serve images from the peoples folder
@app.route('/peoples/<path:filename>')
def peoples_static(filename):
    return send_from_directory(app.config['PEOPLE_FOLDER'], filename)

# Root URL (index)
@app.route('/')
def index():
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)
