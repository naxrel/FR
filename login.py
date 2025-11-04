import cv2
import json
import os
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk

# --- Constants ---
CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "model.yml"
MAPPING_PATH = "name_mapping.json"
CONF_THRESHOLD = 65  # Confidence threshold (lower is better for LBPH)

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

class FaceLoginApp:
    def __init__(self, root, cam_index):
        self.root = root
        self.root.title("Face Recognition Login")
        
        # Set a reasonable default window size
        self.root.geometry("700x600")

        self.cap = None
        self.cam_index = cam_index

        # --- Load Models ---
        self.models_loaded = self.load_models()
        if not self.models_loaded:
            self.show_error_and_exit()
            return

        # --- Configure Fonts ---
        self.header_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
        self.status_font = tkFont.Font(family="Helvetica", size=14)

        # --- Create Widgets ---
        
        # Header Label
        self.header_label = tk.Label(root, text="Face Recognition Login", font=self.header_font)
        self.header_label.pack(pady=15)

        # Video Feed Label
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10, padx=10)

        # Status Label
        self.status_var = tk.StringVar()
        self.status_var.set("Scanning for face...")
        self.status_label = tk.Label(root, textvariable=self.status_var, font=self.status_font, fg="black")
        self.status_label.pack(pady=15)

        # Quit Button
        self.quit_button = tk.Button(root, text="Quit", command=self.on_closing, font=self.status_font)
        self.quit_button.pack(pady=10)

        # --- Start Application ---
        self.open_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()

    def load_models(self):
        """Loads the cascade, recognizer, and name mapping."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH) or not os.path.exists(CASCADE_PATH):
            print("Error: Required model or cascade files are missing.")
            print("Please run 01_create_dataset.py and 02_train_model.py first.")
            print(f"Check for: {MODEL_PATH}, {MAPPING_PATH}, {CASCADE_PATH}")
            return False
            
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(MODEL_PATH)
            
            self.face_detector = cv2.CascadeClassifier(CASCADE_PATH)

            with open(MAPPING_PATH, 'r') as f:
                name_mapping = json.load(f)
            # Create a reverse mapping (ID -> Name)
            self.id_to_name = {v: k for k, v in name_mapping.items()}
            print("Model and name mapping loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def show_error_and_exit(self):
        """Displays an error in the GUI and prepares to exit."""
        error_label = tk.Label(self.root, text="Error: Failed to load models.\nSee console for details.", fg="red", font=self.header_font)
        error_label.pack(pady=50, padx=20)
        self.root.after(5000, self.root.destroy) # Auto-close after 5 seconds

    def open_camera(self):
        """Opens the selected webcam."""
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open webcam at index {self.cam_index}.")
            self.status_var.set(f"Error: Could not open camera {self.cam_index}")
            self.status_label.config(fg="red")
            return
        print(f"Camera {self.cam_index} opened.")

    def update_frame(self):
        """Reads a frame from the webcam, processes it, and updates the GUI."""
        if self.cap is None or not self.cap.isOpened():
            return # Stop loop if camera isn't working

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            self.root.after(10, self.update_frame) # Try again
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

        logged_in_name = None
        status_color = "black"
        status_text = "Scanning for face..."

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            id_num, confidence = self.recognizer.predict(face_roi)

            if confidence < CONF_THRESHOLD:
                name = self.id_to_name.get(id_num, "Unknown")
                text = f"Logged In: {name}"
                color = (0, 255, 0)  # Green
                
                # Update status
                logged_in_name = name
                status_text = f"Welcome, {name}!"
                status_color = "green"

            else:
                text = "Login Failed"
                color = (0, 0, 255)  # Red
                
                # Update status only if not already logged in
                if logged_in_name is None:
                    status_text = "Login Failed"
                    status_color = "red"

            # Draw rectangle and text on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Update GUI status label
        self.status_var.set(status_text)
        self.status_label.config(fg=status_color)

        # Convert OpenCV (BGR) frame to PIL (RGB) image
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        
        # Convert PIL image to Tkinter format
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the video label with the new image
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Schedule the next frame update
        self.root.after(10, self.update_frame)

    def on_closing(self):
        """Called when the 'Quit' button or window 'X' is pressed."""
        print("Closing application...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    # Check for files before starting GUI
    if not all(os.path.exists(f) for f in [CASCADE_PATH, MODEL_PATH, MAPPING_PATH]):
        print("--- Missing Files Error ---")
        print(f"Please make sure the following files are in the same directory:")
        print(f"1. {CASCADE_PATH} (Download from OpenCV's GitHub)")
        print(f"2. {MODEL_PATH} (Generated by 02_train_model.py)")
        print(f"3. {MAPPING_PATH} (Generated by 02_train_model.py)")
        print("\nExiting. Please run the setup scripts first.")
    else:
        cam_index = get_camera_index()
        root = tk.Tk()
        app = FaceLoginApp(root, cam_index)
        root.mainloop()
