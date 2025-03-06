import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QPushButton, QFileDialog, QMenuBar, QMenu, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

class ImageProcessingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Image Processing GUI")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.original_image = None
        self.processed_image = None

        # Create menu bar
        self.create_menu_bar()

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Image display area
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label, 2)

        # Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        layout.addWidget(control_panel, 1)

        # Sliders for image processing functions
        self.create_sliders(control_layout)

        # Fourier Transform button
        fourier_button = QPushButton("Apply 2D Fourier Transform", self)
        fourier_button.clicked.connect(self.apply_fourier_transform)
        control_layout.addWidget(fourier_button)

        # Reset button
        reset_button = QPushButton("Reset Image", self)
        reset_button.clicked.connect(self.reset_image)
        control_layout.addWidget(reset_button)

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        # Load image action
        load_action = file_menu.addAction("Load Image")
        load_action.triggered.connect(self.load_image)

        # Save image action
        save_action = file_menu.addAction("Save Image")
        save_action.triggered.connect(self.save_image)

        # Exit action
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

    def create_sliders(self, layout):
        # Brightness slider
        self.brightness_slider = self.create_slider("Brightness", -100, 100, 0, self.update_brightness)
        layout.addWidget(QLabel("Brightness"))
        layout.addWidget(self.brightness_slider)

        # Contrast slider
        self.contrast_slider = self.create_slider("Contrast", 0, 200, 100, self.update_contrast)
        layout.addWidget(QLabel("Contrast"))
        layout.addWidget(self.contrast_slider)

        # Blur slider
        self.blur_slider = self.create_slider("Blur", 1, 21, 1, self.update_blur, single_step=2)
        layout.addWidget(QLabel("Blur (Kernel Size)"))
        layout.addWidget(self.blur_slider)

        # Sharpen slider
        self.sharpen_slider = self.create_slider("Sharpen", 0, 100, 0, self.update_sharpen)
        layout.addWidget(QLabel("Sharpen"))
        layout.addWidget(self.sharpen_slider)

        # Threshold slider
        self.threshold_slider = self.create_slider("Threshold", 0, 255, 127, self.update_threshold)
        layout.addWidget(QLabel("Threshold"))
        layout.addWidget(self.threshold_slider)

    def create_slider(self, name, min_val, max_val, default_val, callback, single_step=1):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setSingleStep(single_step)
        slider.valueChanged.connect(callback)
        return slider

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.original_image = cv2.imread(file_name)
            if self.original_image is not None:
                self.processed_image = self.original_image.copy()
                self.display_image(self.processed_image)
            else:
                QMessageBox.critical(self, "Error", "Failed to load image.")

    def save_image(self):
        if self.processed_image is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.bmp)")
            if file_name:
                cv2.imwrite(file_name, self.processed_image)
        else:
            QMessageBox.warning(self, "Warning", "No image to save.")

    def display_image(self, image):
        if image is not None:
            # Convert BGR to RGB for display
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.display_image(self.processed_image)
            self.reset_sliders()

    def reset_sliders(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.blur_slider.setValue(1)
        self.sharpen_slider.setValue(0)
        self.threshold_slider.setValue(127)

    def update_brightness(self):
        if self.original_image is not None:
            value = self.brightness_slider.value()
            self.processed_image = cv2.convertScaleAbs(self.original_image, beta=value)
            self.display_image(self.processed_image)

    def update_contrast(self):
        if self.original_image is not None:
            value = self.contrast_slider.value() / 100.0
            self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=value)
            self.display_image(self.processed_image)

    def update_blur(self):
        if self.original_image is not None:
            value = self.blur_slider.value()
            if value % 2 == 0:  # Ensure odd kernel size
                value += 1
            self.processed_image = cv2.GaussianBlur(self.original_image, (value, value), 0)
            self.display_image(self.processed_image)

    def update_sharpen(self):
        if self.original_image is not None:
            value = self.sharpen_slider.value() / 100.0
            kernel = np.array([[-1, -1, -1],
                               [-1,  9 + value, -1],
                               [-1, -1, -1]])
            self.processed_image = cv2.filter2D(self.original_image, -1, kernel)
            self.display_image(self.processed_image)

    def update_threshold(self):
        if self.original_image is not None:
            value = self.threshold_slider.value()
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, self.processed_image = cv2.threshold(gray_image, value, 255, cv2.THRESH_BINARY)
            self.display_image(self.processed_image)

    def apply_fourier_transform(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(gray_image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            self.processed_image = np.uint8(magnitude_spectrum)
            self.display_image(self.processed_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingWindow()
    window.show()
    sys.exit(app.exec())
