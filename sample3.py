import cv2
import numpy as np
import face_recognition
import winsound
import pymsgbox
import os
from datetime import datetime

# Load reference images for face recognition from the "images" folder
reference_images = []
reference_encodings = []

image_folder = "images"

for filename in os.listdir(image_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        reference_image = face_recognition.load_image_file(image_path)
        reference_encoding = face_recognition.face_encodings(reference_image)[0]

        reference_images.append(reference_image)
        reference_encodings.append(reference_encoding)

# Open the video file (replace 'input_video.mp4' with your video file)
cap = cv2.VideoCapture("input_video1.mp4")

# Get the frame rate of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize a list to keep track of recognized faces (as face encodings)
recognized_face_encodings = []

while True:
    # Capture a frame from the video file
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop when we reach the end of the video

    # Find faces in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face encoding is already in the list of recognized face encodings
        if any([(np.array_equal(face_encoding, known_face)) for known_face in recognized_face_encodings]):
            continue  # Skip this face if it has already been recognized

        # Compare the current face encoding with all reference face encodings
        matches = face_recognition.compare_faces(reference_encodings, face_encoding)

        if any(matches):
            label = "Match"  # Face matches one of the reference images

            # Play an alert sound
            winsound.Beep(1000, 1000)  # You can adjust the frequency and duration of the beep

            # Display an alert message box
            pymsgbox.alert("Matching face detected!", "Alert")

            # Capture the current date and time
            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

            # Save the matching frame as an image with timestamp
            image_filename = f"match_{timestamp}.jpg"
            cv2.imwrite(image_filename, frame)

            # Add the recognized face encoding to the list
            recognized_face_encodings.append(face_encoding)

        else:
            label = "No Match"  # Face does not match any reference image

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the label
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop when 'q' is pressed or at the end of the video
    if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
        break

# Release the video file and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
