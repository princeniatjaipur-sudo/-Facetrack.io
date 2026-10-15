# Face Detection Application

This project uses Python, OpenCV, and Haar Cascade classifiers to detect human faces in either a live webcam feed or an uploaded image.

## Features

- Start and stop webcam detection
- Upload JPG, JPEG, PNG, or BMP images
- Run face detection with OpenCV Haar cascades
- Draw bounding boxes around detected faces
- Display the total number of detected faces
- Handle common camera and file errors gracefully

## Project Structure

- `app.py` — main GUI application
- `face_detector.py` — face detection logic and helper functions
- `requirements.txt` — Python dependencies
- `README.md` — setup and usage instructions
- `assets/` — optional project assets

## Installation

1. Open a terminal in the project folder.
2. Create a virtual environment:

   Windows:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the App

```bash
python app.py
```

## How Haar Cascade Detection Works

A Haar Cascade is a machine-learning-based object detection method that uses a sequence of simple features to identify patterns that look like a face. OpenCV includes a pre-trained cascade file called `haarcascade_frontalface_default.xml`.

The detection process is:

1. Convert the image to grayscale.
2. Use `detectMultiScale` to search for face-like regions at different scales.
3. Adjust parameters such as `scaleFactor`, `minNeighbors`, and `minSize` to balance speed and accuracy.
4. Draw rectangles around the detected regions and count them.

This approach is lightweight and suitable for beginner-friendly projects without a deep-learning model.

## Testing Instructions

### Webcam mode

1. Run `python app.py`.
2. Click `Start Camera`.
3. Allow camera permission if your operating system asks for it.
4. Observe the live frame and face count.
5. Click `Stop Camera` when you are finished.

### Image mode

1. Run `python app.py`.
2. Click `Upload Image`.
3. Select a photo in JPG, JPEG, PNG, or BMP format.
4. Click `Detect Faces` if needed.
5. Review the original and processed images side by side.

## Troubleshooting

### Camera permission issues

- On Windows, allow camera access in the system privacy settings.
- On Linux, ensure your webcam is connected and not already in use by another app.
- If the camera still fails, close other video apps and try again.

### No faces detected

- Move closer to the camera.
- Increase the brightness or reduce shadows.
- Adjust parameters like `scaleFactor` and `minNeighbors` in `face_detector.py`.
- Try a higher-quality image with clear frontal lighting.

### Missing cascade file

- Reinstall OpenCV with `pip install -r requirements.txt`.
- Confirm that the `haarcascade_frontalface_default.xml` file is available in your OpenCV installation.

## Notes

This project is intentionally simple and easy to extend. You can tune detection parameters or add features such as eye detection, multi-face filtering, or a better GUI in a future version.
