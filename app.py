import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from face_detector import load_cascade, process_frame, process_image_file


class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("1200x850")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#eef2f7")

        self.cascade = load_cascade()
        self.cap = None
        self.is_camera_running = False
        self.camera_after_id = None
        self.current_image_path = None

        self._build_ui()
        self._show_placeholder_images()

    def _build_ui(self):
        header = tk.Label(
            self.root,
            text="Real-Time Face Detection System",
            font=("Segoe UI", 26, "bold"),
            bg="#eef2f7",
            fg="#1f2937",
            pady=15,
        )
        header.pack()

        subtitle = tk.Label(
            self.root,
            text="Detect faces from your webcam or an uploaded image using OpenCV Haar Cascade classifiers.",
            font=("Segoe UI", 12),
            bg="#eef2f7",
            fg="#475569",
            wraplength=900,
        )
        subtitle.pack(pady=(0, 12))

        control_frame = tk.Frame(self.root, bg="#eef2f7")
        control_frame.pack(fill="x", padx=20, pady=8)

        button_style = {"padx": 12, "pady": 8, "font": ("Segoe UI", 10, "bold")}

        self.start_button = tk.Button(control_frame, text="Start Camera", command=self.start_camera, **button_style)
        self.stop_button = tk.Button(control_frame, text="Stop Camera", command=self.stop_camera, **button_style)
        self.upload_button = tk.Button(control_frame, text="Upload Image", command=self.upload_image, **button_style)
        self.detect_button = tk.Button(control_frame, text="Detect Faces", command=self.detect_uploaded_image, **button_style)

        self.start_button.pack(side="left", padx=(0, 10))
        self.stop_button.pack(side="left", padx=(0, 10))
        self.upload_button.pack(side="left", padx=(0, 10))
        self.detect_button.pack(side="left")

        info_frame = tk.Frame(self.root, bg="#ffffff", bd=1, relief="solid", padx=18, pady=10)
        info_frame.pack(fill="x", padx=20, pady=10)

        self.status_var = tk.StringVar(value="System ready")
        self.face_count_var = tk.StringVar(value="0 faces detected")

        tk.Label(info_frame, text="Status:", font=("Segoe UI", 11, "bold"), bg="#ffffff").grid(row=0, column=0, sticky="w")
        tk.Label(info_frame, textvariable=self.status_var, font=("Segoe UI", 11), bg="#ffffff", fg="#0f766e").grid(row=0, column=1, sticky="w", padx=(10, 20))
        tk.Label(info_frame, text="Face Count:", font=("Segoe UI", 11, "bold"), bg="#ffffff").grid(row=0, column=2, sticky="w")
        tk.Label(info_frame, textvariable=self.face_count_var, font=("Segoe UI", 11), bg="#ffffff", fg="#0f172a").grid(row=0, column=3, sticky="w", padx=(10, 0))

        image_panel = tk.Frame(self.root, bg="#eef2f7")
        image_panel.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.original_label = tk.Label(image_panel, text="Original Image", font=("Segoe UI", 12, "bold"), bg="#ffffff", relief="solid", bd=1, width=48, height=24)
        self.processed_label = tk.Label(image_panel, text="Processed Image", font=("Segoe UI", 12, "bold"), bg="#ffffff", relief="solid", bd=1, width=48, height=24)

        self.original_label.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.processed_label.pack(side="left", fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _show_placeholder_images(self):
        placeholder = cv2.imread("assets/placeholder.png") if os.path.exists("assets/placeholder.png") else None
        if placeholder is None:
            placeholder = self._create_placeholder_frame("No image loaded yet")
        self._display_cv_image(self.original_label, placeholder)
        self._display_cv_image(self.processed_label, placeholder.copy())

    def _create_placeholder_frame(self, text):
        frame = np.full((420, 640, 3), 245, dtype=np.uint8)
        cv2.putText(
            frame,
            text,
            (140, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (80, 80, 80),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _display_cv_image(self, label, frame):
        frame = frame.copy()
        height, width = frame.shape[:2]
        max_width = 540
        max_height = 360
        scale = min(max_width / width, max_height / height, 1.0)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        resized = cv2.resize(frame, (new_width, new_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo

    def start_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.status_var.set("Camera is already running.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Unable to open webcam.")
            messagebox.showerror("Camera Error", "No webcam was detected or camera permissions were denied.")
            self.cap = None
            return

        self.is_camera_running = True
        self.status_var.set("Camera running")
        self.update_camera_frame()

    def stop_camera(self):
        self.is_camera_running = False
        if self.camera_after_id is not None:
            self.root.after_cancel(self.camera_after_id)
            self.camera_after_id = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.status_var.set("Camera stopped")
        self.face_count_var.set("0 faces detected")
        self._show_placeholder_images()

    def update_camera_frame(self):
        if not self.is_camera_running or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Could not read a frame from the camera.")
            self.stop_camera()
            return

        processed_frame, count = process_frame(frame, self.cascade)
        self._display_cv_image(self.original_label, frame)
        self._display_cv_image(self.processed_label, processed_frame)
        self.face_count_var.set(f"{count} face(s) detected")
        self.status_var.set("Detecting faces in live camera feed")
        self.camera_after_id = self.root.after(30, self.update_camera_frame)

    def upload_image(self):
        self.stop_camera()
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")],
        )
        if not file_path:
            self.status_var.set("No image selected.")
            return

        self.current_image_path = file_path
        self.detect_uploaded_image()

    def detect_uploaded_image(self):
        if self.current_image_path is None:
            messagebox.showinfo("Upload an image", "Please upload an image to detect faces.")
            self.status_var.set("Waiting for image upload")
            return

        try:
            original, processed, count = process_image_file(self.current_image_path, self.cascade)
            self._display_cv_image(self.original_label, original)
            self._display_cv_image(self.processed_label, processed)
            self.face_count_var.set(f"{count} face(s) detected")
            if count == 0:
                self.status_var.set("No faces detected in the uploaded image")
                messagebox.showinfo("Detection Notice", "No faces were detected in the selected image.")
            else:
                self.status_var.set("Face detection complete")
        except (FileNotFoundError, ValueError) as exc:
            self.status_var.set("Image processing failed")
            messagebox.showerror("Image Error", str(exc))

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
