import cv2
import dlib
import os

# Load face detection model
detector = dlib.get_frontal_face_detector()

# Load face recognition model (you'll need a trained model or use a pre-trained one)
face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
# Load employee data (employee_id and corresponding facial features)
employee_data = {
    "employee1": [feature_vector1],  # Replace with actual feature vectors
    "employee2": [feature_vector2],
    # Add more employees and their feature vectors
}

# Define a threshold for recognition
threshold = 0.6  # Adjust this value based on your recognition accuracy

# Get the list of image files from the 'images' directory
image_files = [os.path.join("images", filename) for filename in os.listdir("images")]

for image_file in image_files:
    # Read an image from the file
    frame = cv2.imread(image_file)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = face_recognition_model(frame, face)

        # Perform face recognition (compare with employee data)
        match = False
        for employee_id, employee_features in employee_data.items():
            # Calculate the distance between the detected face and stored features
            distance = sum([(a - b) ** 2 for a, b in zip(landmarks, employee_features)]) ** 0.5
            if distance < threshold:
                match = True
                print(f"Employee: {employee_id} - Image: {image_file}")
                break

        if not match:
            print(f"Unknown - Image: {image_file}")

# This code will process all images in the 'images' directory
