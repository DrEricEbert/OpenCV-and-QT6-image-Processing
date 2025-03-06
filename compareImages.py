import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QScrollArea, QFileDialog, QMessageBox,
                             QPushButton, QHBoxLayout, QVBoxLayout)
from PyQt6.QtGui import QImage, QPixmap, QAction
from PyQt6.QtCore import Qt

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Image Processor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize images
        self.image1 = None
        self.image2 = None
        self.result_image = None
        
        self.init_ui()
        
    def init_ui(self):
        # Create main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create image containers with scroll areas
        self.image1_scroll = QScrollArea()
        self.image2_scroll = QScrollArea()
        self.result_scroll = QScrollArea()
        
        self.image1_label = QLabel()
        self.image2_label = QLabel()
        self.result_label = QLabel()
        
        # Configure scroll areas
        for scroll in [self.image1_scroll, self.image2_scroll, self.result_scroll]:
            scroll.setWidgetResizable(True)
            scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.image1_scroll.setWidget(self.image1_label)
        self.image2_scroll.setWidget(self.image2_label)
        self.result_scroll.setWidget(self.result_label)
        
        # Create buttons
        button_layout = QVBoxLayout()
        self.register_btn = QPushButton("Register Images")
        self.compare_btn = QPushButton("Compare Images")
        self.register_btn.clicked.connect(self.register_images)
        self.compare_btn.clicked.connect(self.compare_images)
        button_layout.addWidget(self.register_btn)
        button_layout.addWidget(self.compare_btn)
        button_layout.addStretch()
        
        # Add widgets to main layout
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image1_scroll)
        image_layout.addWidget(self.image2_scroll)
        
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.result_scroll)
        main_layout.addLayout(button_layout)
        
        # Create menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        # Load actions
        load_image1_action = QAction("Load Image 1", self)
        load_image1_action.triggered.connect(lambda: self.load_image(1))
        load_image2_action = QAction("Load Image 2", self)
        load_image2_action.triggered.connect(lambda: self.load_image(2))
        
        # Save action
        save_result_action = QAction("Save Result", self)
        save_result_action.triggered.connect(self.save_result)
        
        file_menu.addAction(load_image1_action)
        file_menu.addAction(load_image2_action)
        file_menu.addSeparator()
        file_menu.addAction(save_result_action)
    
    def load_image(self, image_num):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            image = cv2.imread(filename)
            if image is None:
                QMessageBox.critical(self, "Error", "Could not open image!")
                return
            
            if image_num == 1:
                self.image1 = image
                self.display_image(image, self.image1_label)
            else:
                self.image2 = image
                self.display_image(image, self.image2_label)
    
    def display_image(self, image, label):
        if len(image.shape) == 2:
            h, w = image.shape
            bytes_per_line = w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)
        label.adjustSize()
    
    def register_images(self):
        if self.image1 is None or self.image2 is None:
            QMessageBox.warning(self, "Warning", "Please load both images first!")
            return
        
        # Convert images to grayscale
        img1_gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Warp image2 to align with image1
            h, w = img1_gray.shape
            aligned_img = cv2.warpPerspective(self.image2, M, (w, h))
            self.result_image = aligned_img
            self.display_image(aligned_img, self.result_label)
        else:
            QMessageBox.warning(self, "Error", "Not enough matches found for registration!")
    
    def compare_images(self):
        if self.image1 is None or self.image2 is None:
            QMessageBox.warning(self, "Warning", "Please load both images first!")
            return
        
        # Ensure images are the same size
        if self.image1.shape != self.image2.shape:
            QMessageBox.warning(self, "Error", "Images must be the same size for comparison!")
            return
        
        # Compute absolute difference
        diff = cv2.absdiff(self.image1, self.image2)
        self.result_image = diff
        self.display_image(diff, self.result_label)
    
    def save_result(self):
        if self.result_image is None:
            QMessageBox.warning(self, "Warning", "No result to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "Save Result", "",
                                                "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)")
        if filename:
            cv2.imwrite(filename, self.result_image)
            QMessageBox.information(self, "Success", "Image saved successfully!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec())