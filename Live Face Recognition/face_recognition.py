import threading  # Import threading module for parallel processing

import cv2  # Import OpenCV library for computer vision tasks
from deepface import DeepFace  # Import DeepFace library for face recognition

# Open the default camera (pass 0) using cv2.VideoCapture
# Note: On some systems, passing cv2.CAP_DSHOW may be necessary for proper camera initialization
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set the frame width and height of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Counter to keep track of frames processed
counter = 0

# Variable to store the result of face matching
face_match = False

# Load a reference image for face matching
reference_img = cv2.imread("Camera.jpg")

# Function to check if a given frame contains a matching face
def check_face(frame):
    global face_match  # Use the global face_match variable
    try:
        # Verify if the face in the frame matches the reference image using DeepFace
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True  # Set face_match to True if the face matches
        else:
            face_match = False  # Set face_match to False if the face does not match
    except ValueError:
        face_match = False  # Set face_match to False if there is an error

# Main loop to continuously capture frames from the camera
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if ret:  # If a frame is successfully read
        if counter % 30 == 0:  # Process every 30 frames
            try:
                # Start a new thread to check the face in the frame
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass  # Ignore any errors that occur

        counter += 1  # Increment the counter

        # Display the result of face matching on the frame
        if face_match:
            # Display "MATCH!" in green if there is a match
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            # Display "NO MATCH!" in red if there is no match
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Display the frame in a window named "video"
        cv2.imshow("video", frame)

        # Check for ESC key press (key code 27) to exit the loop
        key = cv2.waitKey(1)
        if key == 27:  # 27 is the key code for the ESC key
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
