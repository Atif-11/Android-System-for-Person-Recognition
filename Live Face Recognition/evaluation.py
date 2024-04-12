import cv2
import time
from deepface import DeepFace

def test_face_recognition(video_path, reference_img_path):
    cap = cv2.VideoCapture(video_path)
    reference_img = cv2.imread(reference_img_path)

    if reference_img is None:
        raise ValueError("Reference image could not be loaded. Check the file path.")

    total_frames = 0
    correct_detections = 0
    processing_times = []
    failed_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video or cannot fetch frame

        start_time = time.time()

        try:
            result = DeepFace.verify(frame, reference_img.copy(), enforce_detection=False)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Check the result
            if 'verified' in result and result['verified']:
                correct_detections += 1
            else:
                failed_detections += 1

        except Exception as e:
            print("An error occurred while verifying a face:", str(e))
            failed_detections += 1

        total_frames += 1

    cap.release()

    # Calculate the metrics
    if total_frames == 0:
        raise ValueError("No frames were processed. Check the video source.")

    accuracy = (correct_detections / total_frames) * 100
    average_time = sum(processing_times) / len(processing_times) if processing_times else 0

    return accuracy, average_time, total_frames, failed_detections

# Define paths
video_path = 'evaluation_video.mp4'
reference_img_path = 'Camera.jpg'

# Run the test and handle exceptions
try:
    accuracy, average_time, total_frames, failed_detections = test_face_recognition(video_path, reference_img_path)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average processing time per frame: {average_time:.2f} ms")
    print(f"Total frames processed: {total_frames}")
    print(f"Frames failed to process: {failed_detections}")
except Exception as error:
    print("Failed to test face recognition:", error)
