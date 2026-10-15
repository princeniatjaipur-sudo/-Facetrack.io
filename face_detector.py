import os

import cv2


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_CASCADE_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "haarcascade_frontalface_default.xml",
)
DEFAULT_CASCADE_PATH = os.path.join(
    getattr(cv2.data, "haarcascades", ""),
    "haarcascade_frontalface_default.xml",
)


def _resolve_default_cascade_path():
    """Prefer the bundled XML in the project and fall back to OpenCV's packaged copy."""
    for candidate in (LOCAL_CASCADE_PATH, DEFAULT_CASCADE_PATH):
        if candidate and os.path.exists(candidate):
            return candidate
    return LOCAL_CASCADE_PATH


def load_cascade(cascade_path: str | None = None):
    """Load the Haar cascade file used for frontal face detection."""
    path = cascade_path or _resolve_default_cascade_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cascade file not found: {path}. "
            "Check that OpenCV is installed correctly."
        )

    classifier = cv2.CascadeClassifier(path)
    if classifier.empty():
        raise ValueError(
            f"Unable to load Haar cascade from {path}. The file may be damaged."
        )

    return classifier


def detect_faces(
    image,
    cascade,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size=(30, 30),
):
    """Return a list of face rectangles in an image."""
    if image is None:
        raise ValueError("Image data cannot be None.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return faces


def draw_face_boxes(image, faces, color=(0, 255, 0), thickness=2):
    """Draw rectangles around every detected face."""
    annotated = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
    return annotated


def process_frame(frame, cascade, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """Process a single video frame and return the annotated frame and face count."""
    faces = detect_faces(
        frame,
        cascade,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
    )
    annotated = draw_face_boxes(frame, faces)
    return annotated, len(faces)


def process_image_file(
    image_path,
    cascade,
    scale_factor=1.1,
    min_neighbors=5,
    min_size=(30, 30),
):
    """Read an uploaded image, detect faces, and return original + annotated image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file does not exist: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image file: {image_path}")

    annotated, count = process_frame(
        image,
        cascade,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
    )
    return image, annotated, count
