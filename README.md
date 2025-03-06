# OpenCV-and-QT6-image-Processing
Creating a comprehensive GUI for OpenCV image processing using PyQt6 involves several steps, including setting up the GUI, integrating OpenCV functions, and handling file operations. Below is a detailed implementation that includes sliders for various image processing functions, a 2D Fourier Transform, and file handling capabilities.


## Explanation of the Code:

1. **Imports**:
   - The necessary modules from PyQt6, OpenCV (`cv2`), and NumPy are imported.
   - pip install opencv-python numpy
   - conda install pyqt qt qtpy

2. **Main Window Setup**:
   - The `ImageProcessingWindow` class inherits from `QMainWindow`.
   - A menu bar is created with options to load, save, and exit.
   - The main layout is divided into an image display area and a control panel.

3. **Image Display**:
   - The `display_image` method converts OpenCV images (BGR) to RGB and displays them using a `QLabel`.

4. **Sliders**:
   - Sliders are created for brightness, contrast, blur, sharpen, and threshold adjustments.
   - Each slider is connected to a corresponding image processing function.

5. **Image Processing Functions**:
   - `update_brightness`: Adjusts image brightness.
   - `update_contrast`: Adjusts image contrast.
   - `update_blur`: Applies Gaussian blur with adjustable kernel size.
   - `update_sharpen`: Applies a sharpening filter.
   - `update_threshold`: Applies binary thresholding to a grayscale image.
   - `apply_fourier_transform`: Computes and displays the magnitude
