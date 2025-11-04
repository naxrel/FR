import cv2
import os
import time

# --- Constants ---
CASCADE_PATH = "haarcascade_frontalface_default.xml"
DATASET_PATH = "dataset"
IMAGES_TO_CAPTURE = 50

def get_camera_index():
    """Asks the user for the camera index."""
    cam_index_str = input("Enter camera index (0 is default, 1, 2, etc.): ")
    if cam_index_str.strip() == "":
        print("Using default camera 0.")
        return 0
    try:
        return int(cam_index_str)
    except ValueError:
        print("Invalid input. Using default camera 0.")
        return 0

def create_dataset():
    """
    Captures and saves face images from the webcam for a new user.
    """
    user_name = input("Enter your name: ")
    if not user_name:
        print("Name cannot be empty.")
        return

    # Create a directory for the user
    user_dataset_path = os.path.join(DATASET_PATH, user_name)
    os.makedirs(user_dataset_path, exist_ok=True)
    print(f"Dataset directory created at: {user_dataset_path}")

    # Load the Haar cascade for face detection
    try:
        face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    except cv2.error as e:
        print(f"Error loading cascade file: {e}")
        print(f"Make sure '{CASCADE_PATH}' is in the same directory.")
        return

    # Ask user for camera index
    cam_index = get_camera_index()
    print(f"Opening camera at index {cam_index}...")

    # Open the webcam
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {cam_index}.")
        print("Please check if the camera is connected or try a different index.")
        return

    print("\nStarting face capture. Look at the camera and move your head slightly.")
    print("Press 'q' to quit early.")

    count = 0
    while count < IMAGES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip the frame horizontally (like a mirror)
        frame = cv2.flip(frame, 1)
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Save the captured face (the grayscale version)
            face_roi = gray[y:y+h, x:x+w]
            
            # Only save if a face is clearly detected
            if face_roi.size > 0:
                count += 1
                img_path = os.path.join(user_dataset_path, f"image_{count}.jpg")
                cv2.imwrite(img_path, face_roi)

                # Display capture status on the frame
                status_text = f"Capturing... {count}/{IMAGES_TO_CAPTURE}"
                cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Brief pause to allow for slight head movement
            time.sleep(0.05) 

        # Display the video feed
        cv2.imshow('Create Dataset', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted by user.")
            break
            
    print(f"\nCaptured {count} images successfully.")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(CASCADE_PATH):
        print(f"Error: Cascade file not found.")
        print(f"Please download '{CASCADE_PATH}' and place it in this directory.")
    else:
        create_dataset()

