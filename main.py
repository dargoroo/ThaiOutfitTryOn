import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import simpledialog, ttk
from tkinter import messagebox
import multiprocessing
import os
import random
import time

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

# Cartoon directory
CARTOON_DIR = 'img'
NUM_CARTOON_TYPES = 6 # 0-5
NUM_CHARS_PER_TYPE = 10 # 0-9

# Global variables for mouse dragging in process_video (REMOVED)

# Overlay function
def overlay_image(background, overlay, x, y, width, height, alpha_blend=True):
    # Ensure dimensions are positive
    if width <= 0 or height <= 0:
        # print(f"Invalid dimensions for resizing: width={width}, height={height}. Skipping overlay.")
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
        # print("Overlay region is outside background bounds or has zero dimensions. Skipping overlay.")
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


def load_cartoon_images(shared_cartoon_images):
    print(f"Loading cartoon images from: {CARTOON_DIR}")
    for i in range(NUM_CARTOON_TYPES):
        type_list = manager.list() # Use manager.list() for nested lists as well
        for j in range(NUM_CHARS_PER_TYPE):
            cartoon_path = os.path.join(CARTOON_DIR, f'image_{i}_{j}.png')
            cartoon = cv2.imread(cartoon_path, cv2.IMREAD_UNCHANGED)
            if cartoon is not None:
                type_list.append(cartoon)
            # else:
            #     print(f"Warning: Failed to load cartoon image: {cartoon_path}")
        if type_list:
            shared_cartoon_images.append(type_list)
        else:
            # Append an empty list if no characters for this type, to maintain structure
            shared_cartoon_images.append(manager.list()) 
            print(f"Warning: No images loaded for cartoon type {i} from {CARTOON_DIR}.")
    if not shared_cartoon_images:
        print("Error: No cartoon images loaded. Ensure the 'img/' folder and files exist.")


# Mouse callback function for OpenCV window (REMOVED as drag-and-drop is removed)

def process_video(shared_outfit_images, outfit_index, x_offset, y_offset, global_width_adjust, size_multiplier,
                  shared_frame_buffer, alpha_blend_mode, frame_dims,
                  shared_cartoon_images, show_cartoon, selected_cartoon_type, selected_cartoon_char,
                  cartoon_x_offset, cartoon_y_offset, cartoon_scale, # cartoon_drag_mode REMOVED
                  current_frame_width_proxy, current_frame_height_proxy,
                  display_video_width, display_video_height,
                  display_control_width):
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # No longer attempting to set webcam resolution to 1280*2 x 720*2,
    # as the camera only supports 1920x1080 and resizing is removed.
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_video_width.value)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_video_height.value)

    # Get actual frame dimensions from the camera
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_dims[0] = frame_h
    frame_dims[1] = frame_w
    frame_dims[2] = 3

    print(f"Webcam frame dimensions (actual): {frame_w}x{frame_h}")

    video_window_x = display_control_width.value + 20
    video_window_y = 0
    first_frame_displayed = False

    if not shared_outfit_images and not shared_cartoon_images:
        print("Error: No outfits or cartoon images loaded. Exiting video processing.")
        cap.release()
        return
    
    # Pack shared variables into a dict for mouse callback param (removed cartoon_drag_mode)
    # This dictionary is no longer needed since mouse_callback is removed, but keeping it
    # as a placeholder comment for future reference if similar structures are re-added.
    # mouse_callback_params = {
    #     'show_cartoon': show_cartoon,
    #     'cartoon_x_offset': cartoon_x_offset,
    #     'cartoon_y_offset': cartoon_y_offset,
    #     'cartoon_scale': cartoon_scale,
    #     'shared_cartoon_images': shared_cartoon_images,
    #     'selected_cartoon_type': selected_cartoon_type,
    #     'selected_cartoon_char': selected_cartoon_char,
    #     'current_frame_width': current_frame_width_proxy,
    #     'current_frame_height': current_frame_height_proxy,
    # }
    
    cv2.namedWindow("Thai Outfit Try-On") # Removed cv2.WINDOW_NORMAL
    # cv2.resizeWindow("Thai Outfit Try-On", display_video_width.value, display_video_height.value) # REMOVED
    # cv2.setMouseCallback("Thai Outfit Try-On", mouse_callback, mouse_callback_params) # REMOVED

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Removed frame resizing here, frame will retain actual webcam dimensions (1920x1080)

        current_frame_h, current_frame_w, _ = frame.shape
        # Update current frame dimensions in proxies for mouse callback (will be actual webcam dims)
        current_frame_width_proxy.set(current_frame_w)
        current_frame_height_proxy.set(current_frame_h)


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # --- Outfit Rendering ---
        if shared_outfit_images: # Only attempt if outfits are loaded
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                
                left_shoulder_coords = np.array([left_shoulder.x * current_frame_w, left_shoulder.y * current_frame_h])
                right_shoulder_coords = np.array([right_shoulder.x * current_frame_w, right_shoulder.y * current_frame_h])
                left_hip_coords = np.array([left_hip.x * current_frame_w, left_hip.y * current_frame_h])
                right_hip_coords = np.array([right_hip.x * current_frame_w, right_hip.y * current_frame_h])
                left_ankle_coords = np.array([left_ankle.x * current_frame_w, left_ankle.y * current_frame_h])
                right_ankle_coords = np.array([right_ankle.x * current_frame_w, right_ankle.y * current_frame_h])

                shoulder_mid_x = int((left_shoulder_coords[0] + right_shoulder_coords[0]) / 2)
                shoulder_mid_y = int((left_shoulder_coords[1] + right_shoulder_coords[1]) / 2)

                # --- Added these two lines for NameError fix ---
                hip_mid_x = int((left_hip_coords[0] + right_hip_coords[0]) / 2)
                hip_mid_y = int((left_hip_coords[1] + right_hip_coords[1]) / 2)
                # ---------------------------------------------

                ankle_mid_y = int((left_ankle_coords[1] + right_ankle_coords[1]) / 2)

                shoulder_width = np.linalg.norm(right_shoulder_coords - left_shoulder_coords)
                
                neck_y_offset_factor = 0.05 
                neck_x = shoulder_mid_x
                neck_y = int(shoulder_mid_y + shoulder_width * neck_y_offset_factor)

                person_total_height_to_ankle = ankle_mid_y - neck_y
                
                if person_total_height_to_ankle <= 0 or ankle_mid_y == 0: 
                    person_total_height_to_ankle = np.linalg.norm(np.array([shoulder_mid_x, shoulder_mid_y]) - np.array([hip_mid_x, hip_mid_y])) * 2.8 
                    # print("Warning: person_total_height_to_ankle calculated using fallback method.")
                
                try:
                    current_outfit = shared_outfit_images[outfit_index.value]
                except IndexError:
                    print("IndexError: Outfit index out of range. Resetting to 0.")
                    outfit_index.value = 0
                    continue

                original_outfit_width = current_outfit.shape[1]
                original_outfit_height = current_outfit.shape[0]
                outfit_aspect_ratio = original_outfit_width / original_outfit_height

                outfit_shoulder_to_arm_factor = 1.7 

                desired_width = int(shoulder_width * outfit_shoulder_to_arm_factor)
                
                outfit_neck_relative_y = 0.10  
                outfit_bottom_relative_y = 0.95 

                outfit_effective_height_ratio = outfit_bottom_relative_y - outfit_neck_relative_y
                if outfit_effective_height_ratio <= 0:
                    outfit_effective_height_ratio = 1.0 

                outfit_length_scale_factor = 1.0 
                
                desired_height = int((person_total_height_to_ankle / outfit_effective_height_ratio) * size_multiplier.value * outfit_length_scale_factor)
                
                calculated_width = int(desired_height * outfit_aspect_ratio)
                calculated_width = int(calculated_width * global_width_adjust.value)

                x_outfit = int(neck_x - (calculated_width / 2) + x_offset.value) 
                y_outfit = int(neck_y - (desired_height * outfit_neck_relative_y) + y_offset.value)

                frame = overlay_image(frame, current_outfit, x_outfit, y_outfit, calculated_width, desired_height, alpha_blend_mode.value)

        # --- Cartoon Character Rendering ---
        if show_cartoon.value and shared_cartoon_images:
            try:
                # Ensure the selected cartoon type exists and has characters
                cartoon_type_idx = selected_cartoon_type.value
                cartoon_char_idx = selected_cartoon_char.value

                if cartoon_type_idx < len(shared_cartoon_images) and \
                   shared_cartoon_images[cartoon_type_idx] and \
                   cartoon_char_idx < len(shared_cartoon_images[cartoon_type_idx]):
                    
                    current_cartoon = shared_cartoon_images[cartoon_type_idx][cartoon_char_idx]
                    
                    cartoon_display_width = int(current_cartoon.shape[1] * cartoon_scale.value)
                    cartoon_display_height = int(current_cartoon.shape[0] * cartoon_scale.value)

                    # Center the cartoon by default, then apply offset
                    cartoon_center_x = int(current_frame_w / 2)
                    cartoon_center_y = int(current_frame_h / 2)

                    cartoon_x = int(cartoon_center_x - cartoon_display_width / 2 + cartoon_x_offset.value)
                    cartoon_y = int(cartoon_center_y - cartoon_display_height / 2 + cartoon_y_offset.value)

                    frame = overlay_image(frame, current_cartoon, cartoon_x, cartoon_y, 
                                        cartoon_display_width, cartoon_display_height, alpha_blend_mode.value)
                else:
                    # print(f"Cartoon image index out of range or empty: Type {cartoon_type_idx}, Char {cartoon_char_idx}. Skipping cartoon rendering.")
                    pass # Don't print warning every frame if no images for a type
            except IndexError as e:
                print(f"Error accessing cartoon image: {e}. Check cartoon indices. Type: {selected_cartoon_type.value}, Char: {selected_cartoon_char.value}")
                show_cartoon.set(False) # Turn off cartoon to prevent repeated errors


        cv2.imshow("Thai Outfit Try-On", frame)
        
        # Move the OpenCV window only once after it's created
        if not first_frame_displayed:
            cv2.moveWindow("Thai Outfit Try-On", video_window_x, video_window_y)
            first_frame_displayed = True

        # Copy the frame data to the shared buffer
        if frame.shape[0] == frame_dims[0] and frame.shape[1] == frame_dims[1]:
            shared_frame_buffer[:] = frame.tobytes() 
        else:
            # This might happen if camera resolution changes, or if initial capture was wrong.
            # Reinitialize shared_frame_buffer if size truly changes mid-stream
            # For simplicity, we'll just not update the buffer if dimensions mismatch,
            # but ideally you'd reallocate if the camera truly changed resolution.
            # print(f"Warning: Webcam frame size changed from expected {frame_dims[1]}x{frame_dims[0]} to {frame.shape[1]}x{frame.shape[0]}. Not updating shared_frame_buffer.")
            pass


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_app(shared_outfit_images, shared_male_outfits, shared_female_outfits, outfit_index, x_offset, y_offset, global_width_adjust, size_multiplier,
            shared_frame_buffer, alpha_blend_mode, current_gender_mode, frame_dims,
            shared_cartoon_images, show_cartoon, selected_cartoon_type, selected_cartoon_char,
            cartoon_x_offset, cartoon_y_offset, cartoon_scale, # cartoon_drag_mode REMOVED
            display_control_width, display_video_width, display_video_height):

    root = tk.Tk()
    root.title("Thai Outfit Try-On")
    root.geometry(f"{display_control_width.value}x1000+0+0") # ใช้ค่าจากตัวแปร

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

    # Tkinter variables for cartoon scale (x, y offsets will be controlled by buttons only)
    # tk_cartoon_scale = tk.DoubleVar(value=cartoon_scale.get()) # REMOVED

    # --- UI Update from Video Process (for drag/drop) --- (REMOVED)
    # def update_cartoon_ui_from_video():
    #     # Update cartoon_x_offset and cartoon_y_offset for drag/drop
    #     # These are not linked to a Tkinter Entry directly anymore, but the video process might change them.
    #     # However, for manual control via buttons, we don't need to push values back to UI here.
    #     # The scale entry will still update automatically.

    #     root.after(100, update_cartoon_ui_from_video)

    # root.after(100, update_cartoon_ui_from_video)


    def capture_frame():
        if frame_dims[0] > 0 and frame_dims[1] > 0:
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
        # Ensure shared_gender_mode is also set if it exists in your actual code
        # If 'shared_gender_mode' is a typo and should be 'current_gender_mode', use that.
        # Assuming it's meant to be current_gender_mode:
        # current_gender_mode.set("female")
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

    # --- Cartoon Functions ---
    def toggle_cartoon():
        show_cartoon.set(not show_cartoon.value)
        cartoon_toggle_button.config(text=f"Cartoon: {'ON' if show_cartoon.value else 'OFF'}")
        # Reset cartoon position/scale when toggling visibility
        cartoon_x_offset.set(0)
        cartoon_y_offset.set(0)
        cartoon_scale.set(1.0)
        # tk_cartoon_scale.set(1.0) # REMOVED

    def next_cartoon_char():
        current_type = selected_cartoon_type.value
        if current_type < len(shared_cartoon_images) and shared_cartoon_images[current_type]:
            new_char_index = (selected_cartoon_char.value + 1) % len(shared_cartoon_images[current_type])
            selected_cartoon_char.set(new_char_index)
            # Reset position/scale when character changes
            cartoon_x_offset.set(0)
            cartoon_y_offset.set(0)
            cartoon_scale.set(1.0)
            # tk_cartoon_scale.set(1.0) # REMOVED
        else:
            print("No characters available for selected cartoon type.")

    def prev_cartoon_char():
        current_type = selected_cartoon_type.value
        if current_type < len(shared_cartoon_images) and shared_cartoon_images[current_type]:
            new_char_index = (selected_cartoon_char.value - 1) % len(shared_cartoon_images[current_type])
            selected_cartoon_char.set(new_char_index)
            # Reset position/scale when character changes
            cartoon_x_offset.set(0)
            cartoon_y_offset.set(0)
            cartoon_scale.set(1.0)
            # tk_cartoon_scale.set(1.0) # REMOVED
        else:
            print("No characters available for selected cartoon type.")

    def update_cartoon_type(event=None):
        selected_index = cartoon_type_combobox.current()
        selected_cartoon_type.set(selected_index)
        selected_cartoon_char.set(0) # Reset character index when type changes
        # Reset position/scale when type changes
        cartoon_x_offset.set(0)
        cartoon_y_offset.set(0)
        cartoon_scale.set(1.0)
        # tk_cartoon_scale.set(1.0) # REMOVED

    control_frame = ttk.Frame(root, padding="10")
    control_frame.pack(fill="both", expand=True)

    # Outfit Controls (remain unchanged)
    ttk.Label(control_frame, text="Outfit Navigation", font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
    ttk.Button(control_frame, text="Previous Outfit", command=lambda: outfit_index.set((outfit_index.get() - 1) % len(shared_outfit_images))).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    ttk.Button(control_frame, text="Next Outfit", command=lambda: outfit_index.set((outfit_index.get() + 1) % len(shared_outfit_images))).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    ttk.Label(control_frame, text="Gender Selection", font=("Helvetica", 12, "bold")).grid(row=2, column=0, columnspan=2, pady=5)
    ttk.Button(control_frame, text="Show Male Outfits", command=switch_gender_male).grid(row=3, column=0, padx=5, pady=5, sticky="ew")
    ttk.Button(control_frame, text="Show Female Outfits", command=switch_gender_female).grid(row=3, column=1, padx=5, pady=5, sticky="ew")

    ttk.Label(control_frame, text="Adjust Outfit X Position", font=("Helvetica", 10)).grid(row=4, column=0, columnspan=2, pady=2)
    ttk.Button(control_frame, text="Left (-)", command=lambda: x_offset.set(x_offset.get() - 5)).grid(row=5, column=0, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Right (+)", command=lambda: x_offset.set(x_offset.get() + 5)).grid(row=5, column=1, padx=5, pady=2, sticky="ew")

    ttk.Label(control_frame, text="Adjust Outfit Y Position", font=("Helvetica", 10)).grid(row=6, column=0, columnspan=2, pady=2)
    ttk.Button(control_frame, text="Up (-)", command=lambda: y_offset.set(y_offset.get() - 5)).grid(row=7, column=0, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Down (+)", command=lambda: y_offset.set(y_offset.get() + 5)).grid(row=7, column=1, padx=5, pady=2, sticky="ew")

    ttk.Label(control_frame, text="Outfit Size Multiplier (Overall)", font=("Helvetica", 10)).grid(row=8, column=0, columnspan=2, pady=2)
    ttk.Button(control_frame, text="Increase Size", command=lambda: size_multiplier.set(size_multiplier.get() + 0.01)).grid(row=9, column=0, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Decrease Size", command=lambda: size_multiplier.set(size_multiplier.get() - 0.01)).grid(row=9, column=1, padx=5, pady=2, sticky="ew")

    ttk.Label(control_frame, text="Outfit Width Adjustment", font=("Helvetica", 10)).grid(row=10, column=0, columnspan=2, pady=2)
    ttk.Button(control_frame, text="Wider (+)", command=lambda: global_width_adjust.set(global_width_adjust.get() + 0.02)).grid(row=11, column=0, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Narrower (-)", command=lambda: global_width_adjust.set(global_width_adjust.get() - 0.02)).grid(row=11, column=1, padx=5, pady=2, sticky="ew")

    ttk.Button(control_frame, text="Auto Adjust Outfit", command=auto_adjust_outfit).grid(row=12, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

    blend_button = ttk.Button(control_frame, text=f"Alpha Blend: {'ON' if alpha_blend_mode.value else 'OFF'}", command=toggle_alpha_blend)
    blend_button.grid(row=13, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

    # --- Cartoon Controls ---
    ttk.Separator(control_frame, orient="horizontal").grid(row=14, column=0, columnspan=3, sticky="ew", pady=10) # Adjusted columnspan
    ttk.Label(control_frame, text="Cartoon Character Controls", font=("Helvetica", 12, "bold")).grid(row=15, column=0, columnspan=3, pady=5) # Adjusted columnspan

    cartoon_toggle_button = ttk.Button(control_frame, text=f"Cartoon: {'ON' if show_cartoon.value else 'OFF'}", command=toggle_cartoon)
    cartoon_toggle_button.grid(row=16, column=0, columnspan=3, padx=5, pady=5, sticky="ew") # Adjusted columnspan

    ttk.Label(control_frame, text="Select Cartoon Type:", font=("Helvetica", 10)).grid(row=17, column=0, sticky="w", pady=2)
    cartoon_types = ["พระอภัยมณี", "พระอภัยมณีและนางเงือก", "นางเงือก", "นางยักษ์", "สินสมุทร", "สุดสาคร"]
    cartoon_type_combobox = ttk.Combobox(control_frame, values=cartoon_types, state="readonly")
    cartoon_type_combobox.set(cartoon_types[selected_cartoon_type.value])
    cartoon_type_combobox.grid(row=17, column=1, padx=5, pady=2, sticky="ew", columnspan=2) # Adjusted columnspan
    cartoon_type_combobox.bind("<<ComboboxSelected>>", update_cartoon_type)

    ttk.Label(control_frame, text="Change Character:", font=("Helvetica", 10)).grid(row=18, column=0, sticky="w", pady=2) # Adjusted column to 0
    ttk.Button(control_frame, text="Previous Char", command=prev_cartoon_char).grid(row=18, column=1, padx=5, pady=2, sticky="ew") # Adjusted row, column
    ttk.Button(control_frame, text="Next Char", command=next_cartoon_char).grid(row=18, column=2, padx=5, pady=2, sticky="ew") # Adjusted row, column

    # --- Cartoon X Offset Buttons --- (RE-ADDED)
    ttk.Label(control_frame, text="Cartoon X Offset:", font=("Helvetica", 10)).grid(row=19, column=0, sticky="w", pady=2)
    ttk.Button(control_frame, text="Left (-50)", command=lambda: cartoon_x_offset.set(cartoon_x_offset.get() - 50)).grid(row=19, column=1, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Right (+50)", command=lambda: cartoon_x_offset.set(cartoon_x_offset.get() + 50)).grid(row=19, column=2, padx=5, pady=2, sticky="ew")

    # --- Cartoon Y Offset Buttons --- (RE-ADDED)
    ttk.Label(control_frame, text="Cartoon Y Offset:", font=("Helvetica", 10)).grid(row=20, column=0, sticky="w", pady=2)
    ttk.Button(control_frame, text="Up (-50)", command=lambda: cartoon_y_offset.set(cartoon_y_offset.get() - 50)).grid(row=20, column=1, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Down (+50)", command=lambda: cartoon_y_offset.set(cartoon_y_offset.get() + 50)).grid(row=20, column=2, padx=5, pady=2, sticky="ew")

    # --- Cartoon Scale Buttons (Adjusted row numbers) ---
    ttk.Label(control_frame, text="Cartoon Scale:", font=("Helvetica", 10)).grid(row=21, column=0, sticky="w", pady=2)
    ttk.Button(control_frame, text="Decrease Scale", command=lambda: cartoon_scale.set(max(0.1, cartoon_scale.get() - 0.05))).grid(row=21, column=1, padx=5, pady=2, sticky="ew")
    ttk.Button(control_frame, text="Increase Scale", command=lambda: cartoon_scale.set(cartoon_scale.get() + 0.05)).grid(row=21, column=2, padx=5, pady=2, sticky="ew")

    # Capture Button
    capture_button = tk.Button(
        root,
        text="Capture Image",
        command=capture_frame,
        bg="#00CED1",
        fg="black",
        activebackground="#008B8B",
        activeforeground="white",
        font=("Helvetica", 16, "bold"),
        height=2,
        width=25
    )
    capture_button.pack(pady=20, fill="x", padx=10)

    # Adjust grid column configuration as we now consistently have 3 columns
    control_frame.grid_columnconfigure(0, weight=1)
    control_frame.grid_columnconfigure(1, weight=1)
    control_frame.grid_columnconfigure(2, weight=1)

    root.mainloop()


if __name__ == "__main__":
    # --- IMPORTANT: Set start method for multiprocessing on macOS/Linux ---
    multiprocessing.set_start_method('spawn', force=True)

    manager = multiprocessing.Manager()
    shared_outfit_images = manager.list()
    shared_male_outfits = manager.list()
    shared_female_outfits = manager.list()

    frame_dims = manager.Array('i', [0, 0, 0]) # [height, width, channels]

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

    shared_frame_buffer = multiprocessing.Array('B', initial_frame_h * initial_frame_w * 3)

    outfit_index = manager.Value('i', 0)
    x_offset = manager.Value('i', 0)
    y_offset = manager.Value('i', 0)
    global_width_adjust = manager.Value('d', 1.0)
    size_multiplier = manager.Value('d', 1.0)
    alpha_blend_mode = manager.Value('b', True)
    current_gender_mode = manager.Value('str', 'male')

    # --- Cartoon specific shared variables ---
    shared_cartoon_images = manager.list() # Nested list: [type][char]
    show_cartoon = manager.Value('b', False) # True to display cartoon
    selected_cartoon_type = manager.Value('i', 0) # 0 for พระอภัยมณี
    selected_cartoon_char = manager.Value('i', 0) # 0 for image_x_0.png
    cartoon_x_offset = manager.Value('i', 0) # Manual X offset from center
    cartoon_y_offset = manager.Value('i', 0) # Manual Y offset from center
    cartoon_scale = manager.Value('d', 1.0) # Scaling factor for cartoon
    # cartoon_drag_mode = manager.Value('b', False) # Flag for mouse dragging (REMOVED)

    # Proxies to pass current frame width/height to mouse callback
    current_frame_width_proxy = manager.Value('i', initial_frame_w)
    current_frame_height_proxy = manager.Value('i', initial_frame_h)

    # --- NEW: Display Window Dimensions ---
    # Width of the control UI (Tkinter)
    display_control_width = manager.Value('i', 450)
    # Width and Height of the video display (OpenCV window) - Reverted to actual webcam dimensions
    display_video_width = manager.Value('i', initial_frame_w)
    display_video_height = manager.Value('i', initial_frame_h)

    # Load images
    load_outfits(shared_outfit_images, shared_male_outfits, shared_female_outfits)
    load_cartoon_images(shared_cartoon_images)

    if shared_male_outfits:
        shared_outfit_images.extend(shared_male_outfits)
        current_gender_mode.set("male")
    elif shared_female_outfits: 
        shared_outfit_images.extend(shared_female_outfits)
        current_gender_mode.set("female")

    video_process = multiprocessing.Process(
        target=process_video,
        args=(shared_outfit_images, outfit_index, x_offset, y_offset, global_width_adjust, size_multiplier,
              shared_frame_buffer, alpha_blend_mode, frame_dims,
              shared_cartoon_images, show_cartoon, selected_cartoon_type, selected_cartoon_char,
              cartoon_x_offset, cartoon_y_offset, cartoon_scale, # cartoon_drag_mode REMOVED
              current_frame_width_proxy, current_frame_height_proxy,
              display_video_width, display_video_height,
              display_control_width)
    )
    video_process.start()

    run_app(shared_outfit_images, shared_male_outfits, shared_female_outfits, outfit_index, x_offset, y_offset, global_width_adjust, size_multiplier,
            shared_frame_buffer, alpha_blend_mode, current_gender_mode, frame_dims,
            shared_cartoon_images, show_cartoon, selected_cartoon_type, selected_cartoon_char,
            cartoon_x_offset, cartoon_y_offset, cartoon_scale, # cartoon_drag_mode REMOVED
            display_control_width, display_video_width, display_video_height)

    video_process.join()