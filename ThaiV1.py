import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import simpledialog, ttk
import multiprocessing
import os
import random

# Disable GPU for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Outfit directory
OUTFIT_DIR = 'Thai_outfit'
MALE_OUTFIT_PREFIX = 'thai_outfit_1_'
FEMALE_OUTFIT_PREFIX = 'thai_outfit_2_'
NUM_MALE_OUTFITS = 10
NUM_FEMALE_OUTFITS = 10

# Overlay function
def overlay_image(background, overlay, x, y, width, height, alpha_blend=True):
    # Ensure dimensions are positive
    if width <= 0 or height <= 0:
        print(f"Invalid dimensions for resizing: width={width}, height={height}. Skipping overlay.")
        return background

    # Resize overlay image
    try:
        overlay_resized = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_LINEAR)
    except cv2.error as e:
        print(f"Resize failed with width={width}, height={height}: {e}")
        return background

    h_resized, w_resized, _ = overlay_resized.shape

    # Calculate ROI coordinates, ensuring they are within background bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(background.shape[1], x + w_resized), min(background.shape[0], y + h_resized)

    # Calculate dimensions of the actual ROI
    roi_w = x2 - x1
    roi_h = y2 - y1

    if roi_w <= 0 or roi_h <= 0:
        print("Overlay region is outside background bounds or has zero dimensions. Skipping overlay.")
        return background

    # Extract the portion of the resized overlay that will actually be placed
    overlay_crop_x1 = 0 if x >= 0 else -x
    overlay_crop_y1 = 0 if y >= 0 else -y
    overlay_crop_x2 = overlay_crop_x1 + roi_w
    overlay_crop_y2 = overlay_crop_y1 + roi_h
    
    overlay_final = overlay_resized[overlay_crop_y1:overlay_crop_y2, overlay_crop_x1:overlay_crop_x2]

    if overlay_final.shape[2] == 4:  # RGBA image
        alpha_channel = overlay_final[:, :, 3]
        if alpha_blend:
            # Normalize alpha channel to [0, 1]
            alpha = alpha_channel / 255.0
            alpha_inv = 1.0 - alpha

            # Split background and overlay into BGR channels
            for c in range(0, 3):
                background[y1:y2, x1:x2, c] = (background[y1:y2, x1:x2, c] * alpha_inv + 
                                               overlay_final[:, :, c] * alpha).astype(np.uint8)
        else: # No alpha blending, just use mask
            # Create a 3-channel mask from the alpha channel
            _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            
            roi = background[y1:y2, x1:x2]
            overlay_bgr = overlay_final[:, :, :3]

            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv2.bitwise_and(overlay_bgr, overlay_bgr, mask=mask)
            background[y1:y2, x1:x2] = cv2.add(bg, fg)

    else: # BGR image, no alpha blending possible
        background[y1:y2, x1:x2] = overlay_final

    return background


def load_outfits(shared_outfit_images, shared_male_outfits, shared_female_outfits):
    # Load male outfits
    for i in range(1, NUM_MALE_OUTFITS + 1):
        outfit_path = os.path.join(OUTFIT_DIR, f'{MALE_OUTFIT_PREFIX}{i}.png')
        print(f"Attempting to load male outfit: {outfit_path}")
        outfit = cv2.imread(outfit_path, cv2.IMREAD_UNCHANGED)
        if outfit is not None:
            print(f"Loaded: {outfit_path}")
            shared_male_outfits.append(outfit)
        else:
            print(f"Failed to load: {outfit_path}")

    # Load female outfits
    for i in range(1, NUM_FEMALE_OUTFITS + 1):
        outfit_path = os.path.join(OUTFIT_DIR, f'{FEMALE_OUTFIT_PREFIX}{i}.png')
        print(f"Attempting to load female outfit: {outfit_path}")
        outfit = cv2.imread(outfit_path, cv2.IMREAD_UNCHANGED)
        if outfit is not None:
            print(f"Loaded: {outfit_path}")
            shared_female_outfits.append(outfit)
        else:
            print(f"Failed to load: {outfit_path}")
    
    # Initialize shared_outfit_images with a default set (e.g., male outfits)
    if shared_male_outfits:
        shared_outfit_images.extend(shared_male_outfits)
    elif shared_female_outfits:
        shared_outfit_images.extend(shared_female_outfits)

    if not shared_outfit_images:
        print("Error: No outfits loaded. Ensure the 'Thai_outfit/' folder and files exist.")


def process_video(shared_outfit_images, outfit_index, x_offset, y_offset, global_width_adjust, size_multiplier, shared_frame_buffer, alpha_blend_mode, frame_dims):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get actual frame dimensions from the camera
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Corrected: Assign elements individually to multiprocessing.Array
    frame_dims[0] = frame_h
    frame_dims[1] = frame_w
    frame_dims[2] = 3

    print(f"Webcam frame dimensions: {frame_w}x{frame_h}")

    # Variables for moving OpenCV window
    video_window_x = 450 + 20 # UI width + small gap
    video_window_y = -100
    first_frame_displayed = False

    if not shared_outfit_images:
        print("Error: No outfits loaded. Exiting video processing.")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Use actual frame dimensions from the current frame
            current_frame_h, current_frame_w, _ = frame.shape 

            # Key points for outfit placement
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # Convert to pixel coordinates
            left_shoulder_coords = np.array([left_shoulder.x * current_frame_w, left_shoulder.y * current_frame_h])
            right_shoulder_coords = np.array([right_shoulder.x * current_frame_w, right_shoulder.y * current_frame_h])
            left_hip_coords = np.array([left_hip.x * current_frame_w, left_hip.y * current_frame_h])
            right_hip_coords = np.array([right_hip.x * current_frame_w, right_hip.y * current_frame_h])
            left_ankle_coords = np.array([left_ankle.x * current_frame_w, left_ankle.y * current_frame_h])
            right_ankle_coords = np.array([right_ankle.x * current_frame_w, right_ankle.y * current_frame_h])

            # Calculate midpoints and distances
            shoulder_mid_x = int((left_shoulder_coords[0] + right_shoulder_coords[0]) / 2)
            shoulder_mid_y = int((left_shoulder_coords[1] + right_shoulder_coords[1]) / 2)
            hip_mid_x = int((left_hip_coords[0] + right_hip_coords[0]) / 2)
            hip_mid_y = int((left_hip_coords[1] + right_hip_coords[1]) / 2)
            ankle_mid_y = int((left_ankle_coords[1] + right_ankle_coords[1]) / 2)

            shoulder_width = np.linalg.norm(right_shoulder_coords - left_shoulder_coords)
            
            # Estimate neck position: using mid-shoulder as base and adjusting vertically
            # neck_y_offset_factor: Value to adjust neck_y relative to shoulder_mid_y
            # Positive value moves neck_y down, negative moves it up.
            neck_y_offset_factor = 0.04 # <--- TUNE THIS FOR OVERALL NECK POSITION
            neck_x = shoulder_mid_x
            neck_y = int(shoulder_mid_y + shoulder_width * neck_y_offset_factor)

            # Calculate overall body height from neck to mid-ankle
            person_total_height_to_ankle = ankle_mid_y - neck_y
            if person_total_height_to_ankle <= 0 or ankle_mid_y == 0: 
                person_total_height_to_ankle = np.linalg.norm(np.array([shoulder_mid_x, shoulder_mid_y]) - np.array([hip_mid_x, hip_mid_y])) * 2.8 
                print("Warning: person_total_height_to_ankle calculated using fallback method.")
            
            try:
                current_outfit = shared_outfit_images[outfit_index.value]
            except IndexError:
                print("IndexError: Outfit index out of range. Resetting to 0.")
                outfit_index.value = 0
                continue

            original_outfit_width = current_outfit.shape[1]
            original_outfit_height = current_outfit.shape[0]
            outfit_aspect_ratio = original_outfit_width / original_outfit_height

            # Determine the desired width and height
            outfit_shoulder_to_arm_factor = 1.7 # <--- TUNE THIS VALUE FOR BASE OUTFIT WIDTH

            desired_width = int(shoulder_width * outfit_shoulder_to_arm_factor)
            
            # IMPORTANT: TUNE THESE VALUES BASED ON YOUR OUTFIT IMAGES
            outfit_neck_relative_y = 0.10  # Percentage from top of outfit image to its neckline
            outfit_bottom_relative_y = 0.95 # Percentage from top of outfit image to its bottom/hem

            outfit_effective_height_ratio = outfit_bottom_relative_y - outfit_neck_relative_y
            if outfit_effective_height_ratio <= 0:
                print("Warning: outfit_effective_height_ratio is zero or negative. Check outfit_neck_relative_y and outfit_bottom_relative_y.")
                outfit_effective_height_ratio = 1.0 

            outfit_length_scale_factor = 1.0 # <--- TUNE THIS VALUE FOR BASE OUTFIT LENGTH
            
            # Calculate desired_height based on person's height and outfit's effective height ratio
            desired_height = int((person_total_height_to_ankle / outfit_effective_height_ratio) * size_multiplier.value * outfit_length_scale_factor)
            
            # Recalculate calculated_width to maintain the outfit's aspect ratio based on the new desired_height
            calculated_width = int(desired_height * outfit_aspect_ratio)
            
            # Final width adjustment applied to the aspect-ratio-maintained width
            calculated_width = int(calculated_width * global_width_adjust.value)


            # Calculate the top-left corner (x, y) for overlay
            x_outfit = int(neck_x - (calculated_width / 2) + x_offset.value) # Use x_offset only
            y_outfit = int(neck_y - (desired_height * outfit_neck_relative_y) + y_offset.value) # Use y_offset only

            frame = overlay_image(frame, current_outfit, x_outfit, y_outfit, calculated_width, desired_height, alpha_blend_mode.value)

        cv2.imshow("Thai Outfit Try-On", frame)
        
        # Move the OpenCV window only once after it's created
        if not first_frame_displayed:
            cv2.moveWindow("Thai Outfit Try-On", video_window_x, video_window_y)
            first_frame_displayed = True

        # Copy the frame data to the shared buffer
        if frame.shape[0] == frame_dims[0] and frame.shape[1] == frame_dims[1]:
            shared_frame_buffer[:] = frame.tobytes() 
        else:
            print(f"Warning: Webcam frame size changed from expected {frame_dims[1]}x{frame_dims[0]} to {frame.shape[1]}x{frame.shape[0]}. Not updating shared_frame_buffer.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_app(shared_outfit_images, shared_male_outfits, shared_female_outfits, outfit_index, x_offset, y_offset, global_width_adjust, size_multiplier, shared_frame_buffer, alpha_blend_mode, current_gender_mode, frame_dims):
    root = tk.Tk()
    root.title("Thai Outfit Try-On")
    # Set UI window size and initial position (top-left)
    root.geometry("450x800+0+0") # WIDTHxHEIGHT+X+Y

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

    def capture_frame():
        # Ensure frame_dims has been populated by the video process
        if frame_dims[0] > 0 and frame_dims[1] > 0:
            # Reconstruct the image from shared_frame_buffer bytes
            np_frame = np.frombuffer(shared_frame_buffer.get_obj(), dtype=np.uint8).reshape((frame_dims[0], frame_dims[1], frame_dims[2]))
            filename = os.path.join(desktop_path, f"captured_thai_outfit_image_{random.randint(1000,9999)}.png")
            cv2.imwrite(filename, np_frame)
            print(f"Captured frame saved as {filename}")
        else:
            print("Cannot capture frame: Webcam not initialized or frame dimensions unknown.")

    def switch_gender_male():
        current_gender_mode.set("male")
        shared_outfit_images[:] = [] 
        shared_outfit_images.extend(shared_male_outfits) 
        outfit_index.set(0) 

    def switch_gender_female():
        current_gender_mode.set("female")
        shared_outfit_images[:] = [] 
        shared_outfit_images.extend(shared_female_outfits) 
        outfit_index.set(0) 

    def toggle_alpha_blend():
        alpha_blend_mode.set(not alpha_blend_mode.value)
        blend_button.config(text=f"Alpha Blend: {'ON' if alpha_blend_mode.value else 'OFF'}")

    def auto_adjust_outfit():
        x_offset.set(0)
        y_offset.set(0)
        size_multiplier.set(1.0) 
        global_width_adjust.set(1.0)
        print("Auto Adjust: Outfit position and size reset to default based on current pose.")

    control_frame = ttk.Frame(root, padding="10")
    control_frame.pack(fill="both", expand=True)

    ttk.Label(control_frame, text="Outfit Navigation", font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
    ttk.Button(control_frame, text="Previous Outfit", command=lambda: outfit_index.set((outfit_index.get() - 1) % len(shared_outfit_images))).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    ttk.Button(control_frame, text="Next Outfit", command=lambda: outfit_index.set((outfit_index.get() + 1) % len(shared_outfit_images))).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    ttk.Label(control_frame, text="Gender Selection", font=("Helvetica", 12, "bold")).grid(row=2, column=0, columnspan=2, pady=5)
    ttk.Button(control_frame, text="Show Male Outfits", command=switch_gender_male).grid(row=3, column=0, padx=5, pady=5, sticky="ew")
    ttk.Button(control_frame, text="Show Female Outfits", command=switch_gender_female).grid(row=3, column=1, padx=5, pady=5, sticky="ew")

    ttk.Label(control_frame, text="Adjust X Position", font=("Helvetica", 10)).grid(row=4, column=0, columnspan=2, pady=2)
    ttk.Button(control_frame, text="Left (-)", command=lambda: x_offset.set(x_offset.get() - 5)).grid(row=5, column=0, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Right (+)", command=lambda: x_offset.set(x_offset.get() + 5)).grid(row=5, column=1, padx=5, pady=2, sticky="ew")

    ttk.Label(control_frame, text="Adjust Y Position", font=("Helvetica", 10)).grid(row=6, column=0, columnspan=2, pady=2)
    ttk.Button(control_frame, text="Up (-)", command=lambda: y_offset.set(y_offset.get() - 5)).grid(row=7, column=0, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Down (+)", command=lambda: y_offset.set(y_offset.get() + 5)).grid(row=7, column=1, padx=5, pady=2, sticky="ew")

    ttk.Label(control_frame, text="Outfit Size Multiplier (Overall)", font=("Helvetica", 10)).grid(row=8, column=0, columnspan=2, pady=2)
    ttk.Button(control_frame, text="Increase Size", command=lambda: size_multiplier.set(size_multiplier.get() + 0.01)).grid(row=9, column=0, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Decrease Size", command=lambda: size_multiplier.set(size_multiplier.get() - 0.01)).grid(row=9, column=1, padx=5, pady=2, sticky="ew")

    ttk.Label(control_frame, text="Outfit Width Adjustment", font=("Helvetica", 10)).grid(row=10, column=0, columnspan=2, pady=2)
    ttk.Button(control_frame, text="Wider (+)", command=lambda: global_width_adjust.set(global_width_adjust.get() + 0.02)).grid(row=11, column=0, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Narrower (-)", command=lambda: global_width_adjust.set(global_width_adjust.get() - 0.02)).grid(row=11, column=1, padx=5, pady=2, sticky="ew")
    
    # New Auto Adjust Button
    ttk.Button(control_frame, text="Auto Adjust Outfit", command=auto_adjust_outfit).grid(row=12, column=0, columnspan=2, padx=5, pady=10, sticky="ew") 

    blend_button = ttk.Button(control_frame, text=f"Alpha Blend: {'ON' if alpha_blend_mode.value else 'OFF'}", command=toggle_alpha_blend)
    blend_button.grid(row=13, column=0, columnspan=2, padx=5, pady=10, sticky="ew") # Adjusted row for blend button

    capture_button = tk.Button(
        root,
        text="Capture Image",
        command=capture_frame,
        bg="#00CED1",  # เปลี่ยนเป็นสีฟ้าน้ำทะเล (Dark Turquoise)
        fg="black",  # เปลี่ยนเป็นสีดำ
        activebackground="#008B8B", # สีเมื่อกดค้าง (Dark Cyan)
        activeforeground="white", # สีตัวอักษรเมื่อกดค้าง
        font=("Helvetica", 16, "bold"), # ตัวอักษรสีดำ หนา (จาก fg="black" และ "bold")
        height=2,
        width=25
    )
    capture_button.pack(pady=20, fill="x", padx=10) 

    control_frame.grid_columnconfigure(0, weight=1)
    control_frame.grid_columnconfigure(1, weight=1)

    root.mainloop()


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_outfit_images = manager.list()
    shared_male_outfits = manager.list()
    shared_female_outfits = manager.list()
    
    # Initialize frame_dims as a shared array to store actual webcam dimensions
    frame_dims = manager.Array('i', [0, 0, 0]) # [height, width, channels]

    # First, try to get webcam dimensions *before* creating shared_frame_buffer
    temp_cap = cv2.VideoCapture(0)
    if temp_cap.isOpened():
        initial_frame_w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        initial_frame_h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_cap.release()
        print(f"Detected webcam initial dimensions: {initial_frame_w}x{initial_frame_h}")
    else:
        initial_frame_w = 640  # Default if webcam not immediately available
        initial_frame_h = 480  # Default if webcam not immediately available
        print(f"Could not detect webcam dimensions. Using default: {initial_frame_w}x{initial_frame_h}")

    # Create shared_frame_buffer with the detected/default dimensions
    shared_frame_buffer = multiprocessing.Array('B', initial_frame_h * initial_frame_w * 3)

    outfit_index = manager.Value('i', 0)
    x_offset = manager.Value('i', 0) 
    y_offset = manager.Value('i', 0)
    # Removed neck_offset_y and neck_offset_x
    global_width_adjust = manager.Value('d', 1.0) # Variable for fine-tuning width independently
    size_multiplier = manager.Value('d', 1.0) 
    alpha_blend_mode = manager.Value('b', True) 
    current_gender_mode = manager.Value('str', 'male') 

    load_outfits(shared_outfit_images, shared_male_outfits, shared_female_outfits)

    if shared_male_outfits:
        shared_outfit_images.extend(shared_male_outfits)
        current_gender_mode.set("male")
    elif shared_female_outfits: 
        shared_outfit_images.extend(shared_female_outfits)
        current_gender_mode.set("female")


    video_process = multiprocessing.Process(
        target=process_video,
        args=(shared_outfit_images, outfit_index, x_offset, y_offset, global_width_adjust, size_multiplier, shared_frame_buffer, alpha_blend_mode, frame_dims)
    )
    video_process.start()

    run_app(shared_outfit_images, shared_male_outfits, shared_female_outfits, outfit_index, x_offset, y_offset, global_width_adjust, size_multiplier, shared_frame_buffer, alpha_blend_mode, current_gender_mode, frame_dims)

    video_process.join()