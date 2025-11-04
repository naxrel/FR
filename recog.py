import cv2
import os
import numpy as np
import json

# --- Constants ---
DATASET_PATH = "dataset"
MODEL_PATH = "model.yml"
MAPPING_PATH = "name_mapping.json"

def train_model():
    """
    Reads all user face images from the dataset, trains the 
    LBPH recognizer, and saves the trained model and name mapping.
    """
    print("Starting training process...")
    
    # Check if the dataset directory exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset directory '{DATASET_PATH}' not found.")
        print("Please run '01_create_dataset.py' first.")
        return

    # LBPH (Local Binary Patterns Histograms) Face Recognizer
    # This is the recognizer included in opencv-contrib
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    ids = []
    name_to_id = {}
    current_id = 0
    
    # Walk through the dataset directory
    for dir_name in os.listdir(DATASET_PATH):
        user_path = os.path.join(DATASET_PATH, dir_name)
        
        if not os.path.isdir(user_path):
            continue
            
        # Assign a numerical ID to this user
        if dir_name not in name_to_id:
            name_to_id[dir_name] = current_id
            user_id = current_id
            current_id += 1
            print(f"Training on user: {dir_name} (ID: {user_id})")
        else:
            user_id = name_to_id[dir_name]

        # Loop through all images for this user
        for img_name in os.listdir(user_path):
            if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(user_path, img_name)
            
            # Read the image in grayscale
            face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if face_img is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            # Append the face image and its corresponding ID
            faces.append(face_img)
            ids.append(user_id)

    if not faces:
        print("No faces found to train. Exiting.")
        return

    print(f"\nFound {len(faces)} face images.")
    print("Training the model... This may take a moment.")

    # Train the recognizer
    try:
        recognizer.train(faces, np.array(ids))
        
        # Save the trained model
        recognizer.write(MODEL_PATH)
        
        # Save the name-to-ID mapping
        with open(MAPPING_PATH, 'w') as f:
            json.dump(name_to_id, f, indent=4)
            
        print(f"\nTraining complete.")
        print(f"Model saved as: {MODEL_PATH}")
        print(f"Name mapping saved as: {MAPPING_PATH}")

    except cv2.error as e:
        print(f"An error occurred during training: {e}")
        print("This can happen if there is only one user. Fisherfaces/Eigenfaces")
        print("require at least 2 users. LBPH (this one) should be fine.")
        print("Please check your dataset.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    train_model()
