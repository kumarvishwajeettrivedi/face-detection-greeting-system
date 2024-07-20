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

.
.
.
![working image](https://raw.githubusercontent.com/kumarvishwajeettrivedi/face-detection-greeting-system/main/Screenshot%20from%202024-07-04%2016-47-29.png)




