# face-detection-greeting-system

This project focuses on developing a robust face recognition and greeting model using computer vision tools like OpenCV, DeepFace, and TensorFlow, integrated with hardware components such as Raspberry Pi. The primary objective is to accurately identify individuals from a database and greet them by name in real-time. This system holds significant potential for deployment in various environments including offices, hospitals, and other public spaces for purposes ranging from enhanced personalized interactions to security and attendance monitoring.

## Initial Phase

- **Face Detection**: Utilized OpenCV's `haarcascade_frontalface_default.xml` for human face detection.
- **Visualization**: Employed Matplotlib for live footage visualization.
- **Model Training**: Implemented K-Nearest Neighbors (KNN) and Convolutional Neural Network (CNN) algorithms for real-time face detection.
- **Hardware Setup**: Raspberry Pi module managed processing, training, and model deployment, supported by the Pi Camera Module 2 for image capture.

## Optimization Phase

- **Model Enhancement**: Shifted to training the model offline using the Random Forest algorithm to mitigate lag issues.
- **Image Preprocessing**: Integrated Pillow library for image preprocessing to accelerate face detection and streamline real-time processing.

## User Interface Development

- **UI Integration**: Developed a user-friendly interface using Flask as the backend server.
- **Frontend Tools**: Leveraged HTML, CSS, Bootstrap, and JavaScript for frontend development to enhance user interaction and display real-time face detection results.

## Final Approach

- **Advanced Model Integration**: Transitioned to DeepFace, achieving approximately 88% precision for a larger database.
- **Hardware Challenges**: Due to Raspberry Pi's hardware limitations, explored alternatives like Grove Vision AI V2 with ARM Cortex-M55 processor and ARM Ethos-U55 NPU, supported by Xiao ESP32S3 for wireless communication.
- **Cost-Effective Solution**: Considered Realtek AMB82-Mini 8-in-1 SoC for embedded AI capabilities and efficient high-resolution processing up to 2K at 0.4 TOPs, ensuring cost-effective deployment.

![working image](https://raw.githubusercontent.com/kumarvishwajeettrivedi/face-detection-greeting-system/main/Screenshot%20from%202024-07-04%2018-18-29.png)

Main folder contains:
• img folder: Contains processed images of individuals with correct labeling
and timestamp. The images in this directory are used for training the ML
model.
• People folder: Contains images that are directly uploaded by users for
future reference. If they want to delete an image, it will allow them to delete
all its references within the folder.
• Templates folder: Contains HTML, CSS, and JavaScript code used for
creating the user interface.
• Uploads folder: This folder contains images during the processing time.
• App.py: This is the backend file that hosts the user interface.
• First, launch this file; it will run on localhost:5000.
• This can be changed once the model goes into production and has a
domain.
•
• Final_model_rf.py: This file contains the ML model that detects faces in
real-time.


![working image](https://raw.githubusercontent.com/kumarvishwajeettrivedi/face-detection-greeting-system/main/Screenshot%20from%202024-07-04%2016-47-29.png)



Method to Upload a Picture:
1. First, make sure to run the server. Navigate to the directory where app.py is
located, then in the terminal, write: python3 app.py.
2. In your browser, go to: localhost:5000 (This can be replaced with the actual
domain).
3. A page similar to the one in image 1.3 will appear. Fill out the input column
and click the "Upload" button to upload an image.
Method to Delete a Picture:
1. Click the "View Image" button to see all images of the person.
2. Click the "Delete" button to delete the image and all its references.


The project presents a robust face recognition and greeting system developed
using advanced computer vision tools such as OpenCV, DeepFace, and
TensorFlow, integrated with the Raspberry Pi hardware platform. The primary
goal was to achieve accurate real-time identification of individuals from a pre-
constructed database and greet them by name. This system demonstrates
significant potential for deployment in various environments including offices,
hospitals, and public spaces, with applications ranging from personalized
interactions to enhanced security and attendance monitoring.
