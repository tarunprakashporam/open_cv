import cv2
import face_recognition

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(frame, model="cnn")

    for (top, right, bottom, left) in face_locations:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the frame with face detection results
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
