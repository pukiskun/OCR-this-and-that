# AI Agent Blueprint: Template-Driven OCR Engine (MVP 1)

## 🎯 Context & Objective
You are an AI coding assistant operating within the Antigravity IDE. We are building an MVP for a **Configuration-Driven OCR Engine**. 
The goal is to build a Human-in-the-Loop system where users define extraction rules (bounding boxes) on a reference image via a UI. The system saves these rules as a JSON template and applies them to future document batches using classic Computer Vision (Image Registration) and an OCR engine.

**CRITICAL CONSTRAINT:** Do NOT use any Machine Learning or Deep Learning models for document alignment, dewarping, or classification in this MVP. Rely strictly on classic OpenCV algorithms (ORB, Homography) to ensure the system is lightweight and fast.

## 🛠️ Tech Stack
*   **Language:** Python 3.10+
*   **UI/Frontend:** `gradio` (specifically Gradio 4+ for native embed capabilities and modern Image components)
*   **Computer Vision:** `opencv-python` (cv2)
*   **OCR Engine:** `easyocr`
*   **Data Handling:** `json`, `pandas`

## 🏗️ System Architecture & Modularity
The codebase must be strictly modular. Separate the UI logic, the Computer Vision logic, and the OCR logic into different modules (e.g., `ui.py`, `alignment.py`, `extractor.py`).

### Module 1: The Gradio Configuration UI
*   **Task:** Build an intuitive interface using `gradio.Blocks()`.
*   **Flow:** 
    1. User uploads a reference image (e.g., a standard invoice).
    2. User utilizes Gradio's `ImageEditor` or equivalent component to define regions of interest (bounding boxes) and assigns a text label to each region (e.g., "Total_Amount").
    3. **Action:** Extract the coordinates `(x, y, width, height)` of these regions.
    4. **Output:** Save the reference image dimensions, the bounding box coordinates, and their labels into a `template.json` file.

### Module 2: Image Registration (Alignment Engine)
*   **Task:** Align incoming distorted/skewed images to match the exact perspective of the reference image used in Module 1.
*   **Flow:**
    1. Convert both the reference image and the new input image to grayscale. Apply `cv2.adaptiveThreshold` for illumination robustness.
    2. Detect features and compute descriptors using **ORB** (`cv2.ORB_create()`).
    3. Match features using `cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)`.
    4. Sort matches by distance and retain only the top N matches.
    5. Compute the transformation matrix using `cv2.findHomography` with the `cv2.RANSAC` method.
    6. Warp the input image using `cv2.warpPerspective` to match the exact dimension and alignment of the reference image.
*   **Output:** A perfectly aligned `numpy` array image.

### Module 3: OCR Extraction Pipeline
*   **Task:** Extract text from the aligned image using the saved template.
*   **Flow:**
    1. Load the corresponding `template.json`.
    2. Iterate through the `extraction_fields`.
    3. Crop the aligned image using the saved `(x, y, width, height)` coordinates.
    4. Pass the cropped segment to `easyocr.Reader().readtext()`.
    5. Store the extracted text alongside its label in a dictionary.
*   **Output:** Return a clean JSON or convert the dictionary to a Pandas DataFrame for CSV export.

## 🚀 Execution Steps for AI Agent
When requested to start coding, follow this implementation order:
1.  **Step 1:** Create the foundational utility functions for JSON read/write operations.
2.  **Step 2:** Implement the OpenCV Image Registration pipeline (`alignment.py`). Create unit tests with a dummy skewed image to ensure the homography works perfectly before moving to UI.
3.  **Step 3:** Implement the OCR extraction logic (`extractor.py`).
4.  **Step 4:** Build the Gradio UI (`app.py`), integrating the `ImageEditor` to capture coordinates, and wire it to the backend modules.

## ⚠️ Known Edge Cases (Handle if possible in MVP)
*   **Zero Good Matches:** If ORB fails to find enough good matches (e.g., totally different document), throw a specific error in the Gradio UI ("Document layout mismatch. Please select the correct template.").
*   **Coordinate Out of Bounds:** Ensure bounding boxes don't exceed the image dimensions after cropping.