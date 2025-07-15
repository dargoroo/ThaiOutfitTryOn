# 🇹🇭 Thai Outfit Try-On App

A real-time virtual try-on application using computer vision (MediaPipe Pose and OpenCV) to overlay traditional Thai outfits and cartoon characters onto a user's live webcam feed.

## ✨ Features

* **Real-time Virtual Try-On**: Seamlessly overlays selected Thai outfits onto the user's body in real-time.
* **Pose Estimation**: Utilizes MediaPipe Pose to accurately detect human body keypoints for precise outfit placement.
* **Outfit Customization**:
    * Navigate between various male and female Thai outfits.
    * Adjust outfit position (X/Y offset) and size (scale, width adjustment) to fit different body types.
    * Toggle alpha blending for outfit overlay.
* **Cartoon Character Integration**:
    * Toggle display of pre-loaded Thai-themed cartoon characters (e.g., from Phra Aphai Mani epic).
    * Select different cartoon types and characters within each type.
    * Adjust cartoon character position (X/Y offset) and scale.
* **Image Capture**: Capture and save the current frame with the overlaid outfit/character to your desktop.
* **Multi-process Architecture**: Separates video processing and GUI for smooth performance.

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

* Python 3.8+
* pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/dargoroo/ThaiOutfitTryOn.git](https://github.com/dargoroo/ThaiOutfitTryOn.git)
    cd ThaiOutfitTryOn
    ```

2.  **Install required Python packages:**
    ```bash
    pip install opencv-python mediapipe numpy tk
    ```
    *(Note: `tk` is usually bundled with Python. If you face issues, you might need to install `python3-tk` on Linux or ensure Tkinter is included in your Python installation on Windows/macOS.)*

3.  **Prepare Assets:**
    * Ensure you have a folder named `Thai_outfit` in the root directory of the project, containing your `.png` outfit images (e.g., `thai_outfit_1_1.png`, `thai_outfit_2_1.png`).
    * Ensure you have a folder named `img` in the root directory, containing your `.png` cartoon character images (e.g., `image_0_0.png`, `image_1_0.png`).

### Running the Application

Execute the `main.py` script:

```bash
python main.py
