# 🇹🇭 Thai+Chinese Outfit Try-On App

A real-time virtual try-on application using computer vision (MediaPipe Pose and OpenCV) to overlay traditional Thai outfits and cartoon characters onto a user's live webcam feed.

## 🚀 Getting Started

### Prerequisites
* **Python 3.10+** (Recommended)
* **Git**

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/dargoroo/ThaiOutfitTryOn.git](https://github.com/dargoroo/ThaiOutfitTryOn.git)
    cd ThaiOutfitTryOn
    ```

2.  **Create and Activate Virtual Environment:**
    * **macOS / Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    * **Windows:**
        ```bash
        python -m venv .venv
        .venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    ```bash
    python main.py
    ```

> **Note:** The application includes pre-loaded outfit assets in the `Thai_outfit` and `img` directories.