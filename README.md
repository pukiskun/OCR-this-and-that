# Configuration-Driven OCR Engine (MVP 1)

A modular, configuration-driven Document Alignment and OCR Extraction engine built with **Python**, **Streamlit**, **OpenCV**, and **EasyOCR**. 

This system allows you to build templates (bounding boxes) interactively on a reference document, save those extraction zones as a template, and apply them to future distorted/skewed document uploads. The application utilizes classic Computer Vision algorithms (ORB Keypoints and RANSAC Homography) to align perspectives before extracting field values, ensuring it remains fast and does not require complex deep learning training pipelines.

Repository Link: [https://github.com/pukiskun/OCR-this-and-that.git](https://github.com/pukiskun/OCR-this-and-that.git)

---

## 🛠️ Features

*   **Interactive Template Builder (Tab 1)**: Click coordinates on a reference document (e.g., a standard ID card, form, or invoice) to define and label bounding boxes. It generates and saves a standard `template.json` configuration file.
*   **Warp-Perspective Document Alignment (Tab 2)**: Automatically detects features using **ORB** descriptor matchers to calculate perspective shifts and warp skewed, rotated, or translated document uploads back to the reference coordinate space.
*   **EasyOCR Field Extraction**: Crops aligned segments based on saved template coordinates and extracts text fields using the high-accuracy PyTorch-based **EasyOCR** engine.
*   **Fail-Safe Browser Downloads**: Encodes configurations as raw binary byte streams to bypass desktop download manager (e.g., Internet Download Manager - IDM) interception issues.
*   **State-Sharing Workflow**: Active templates created in Tab 1 are instantly shared and selectable in Tab 2 without manual file downloads or uploads.
*   **Clipboard Fallback**: Provides a click-to-copy code block showing the raw JSON configuration in case browser configurations block file downloads.

---

## 🏗️ System Architecture & Modularity

The codebase is split into independent, reusable modules:

*   **`streamlit_app.py`**: The main Streamlit web application. Handles interactive state management, custom UI canvas rendering, coordinate scaling, and downloads.
*   **`alignment.py`**: High-performance OpenCV perspective registration pipeline using adaptive thresholding, ORB features, BFMatcher, and RANSAC homography.
*   **`extractor.py`**: Handles cropped segment OCR parsing via EasyOCR.
*   **`utils.py`**: Handles template file load/save operations and safety boundary box cropping.
*   **`requirements.txt`**: Specifies dependencies using headless Linux packages (`opencv-python-headless`) for cloud compatibility.

---

## 🚀 Getting Started

### Local Setup & Launch

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/pukiskun/OCR-this-and-that.git
    cd OCR-this-and-that
    ```

2.  **Install Python Dependencies**:
    Make sure you have Python 3.10+ installed. Install required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit Server**:
    Start the local Streamlit server:
    ```bash
    streamlit run streamlit_app.py
    ```
    Once loaded, the terminal will print the URL (usually `http://localhost:8501`). Open this address in your web browser.

---

## ☁️ Deployment to Streamlit Community Cloud (Free)

This application is fully optimized to run on Streamlit's free hosting tier:

1.  Push your code changes to your GitHub repository:
    ```bash
    git add .
    git commit -m "Migrate to Streamlit and fix download logic"
    git remote add origin https://github.com/pukiskun/OCR-this-and-that.git
    git branch -M main
    git push -u origin main
    ```
2.  Go to [Streamlit Community Cloud](https://share.streamlit.io) and log in using your GitHub account.
3.  Click **New app** and select your repository (`pukiskun/OCR-this-and-that`).
4.  Set the Branch to `main` and the Main file path to `streamlit_app.py`.
5.  Click **Deploy**. Streamlit will automatically read your `requirements.txt`, install dependencies (including PyTorch and OpenCV-headless), and launch your application under a public sharing link!

---

## 💡 Technologies Used

*   **UI/Frontend**: [Streamlit](https://streamlit.io/) & [streamlit-image-coordinates](https://github.com/blackary/streamlit-image-coordinates)
*   **Computer Vision**: OpenCV (`opencv-python-headless`)
*   **OCR Engine**: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
*   **Data Structures**: Pandas, Numpy, JSON
