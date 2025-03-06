#!/usr/bin/env python3
import sys
import cv2
import json
import numpy as np
from PyQt6 import QtCore, QtWidgets, QtGui

# -------------------------
# Helper: Rainbow Color Map
# -------------------------
def get_rainbow_color(frame_idx, total_frames):
    """
    Map the frame index to a rainbow color.
    We'll use HSV where hue changes with time, then convert to BGR.
    """
    ratio = frame_idx / max(total_frames, 1)
    hue = int(ratio * 179)  # OpenCV hue in [0,179]
    hsv_color = np.uint8([[[hue, 255, 255]]])
    # Convert HSV to BGR (OpenCV uses BGR)
    bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
    # Return as a tuple in integer format.
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

# --------------------------
# Particle Tracking Class
# --------------------------
class ParticleTracker:
    def __init__(self, parameters):
        """
        parameters is a dict with keys such as:
          - "threshold": int
          - "min_area": float
          - "max_move": float  (max allowed movement between frames)
        """
        self.params = parameters
        self.tracks = {}  # Dictionary: {particle_id: [(frame_idx, (x, y)), ...]}
        self.next_id = 0
        self.tracking_points = {}  # {particle_id: (x, y)} of last detected position

    def process_frame(self, frame, frame_idx):
        """
        Process one frame to detect particles and update tracks.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply a simple threshold to detect bright particles.
        ret, thresh = cv2.threshold(
            gray, self.params['threshold'], 255, cv2.THRESH_BINARY
        )
        # Find contours from thresholded image.
        contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Compatibility for OpenCV 4.x
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.params['min_area']:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detections.append((cx, cy))

        # Associate detections with previously tracked particles
        assigned = set()
        new_tracking = {}

        # For each existing track, try to find a close detection.
        for pid, last_pos in self.tracking_points.items():
            best_det = None
            best_dist = float('inf')
            for det in detections:
                if det in assigned:
                    continue
                dist = np.hypot(last_pos[0] - det[0], last_pos[1] - det[1])
                if dist < best_dist and dist < self.params['max_move']:
                    best_det = det
                    best_dist = dist
            if best_det is not None:
                new_tracking[pid] = best_det
                self.tracks.setdefault(pid, []).append((frame_idx, best_det))
                assigned.add(best_det)

        # Any detection not assigned is considered a new particle.
        for det in detections:
            if det in assigned:
                continue
            pid = self.next_id
            self.next_id += 1
            new_tracking[pid] = det
            self.tracks[pid] = [(frame_idx, det)]
        self.tracking_points = new_tracking

    def to_json(self):
        """
        Export the tracks dictionary to a JSON string.
        """
        return json.dumps(self.tracks, indent=2)

    def get_tracks(self):
        return self.tracks

# --------------------------
# Main Window (Qt6 GUI)
# --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sophisticated Particle Tracker")
        self.resize(1000, 600)

        # Video and Tracking data
        self.video_frames = []  # List of frames (numpy arrays, BGR)
        self.total_frames = 0
        self.current_frame_index = 0
        # Default tracking parameters:
        self.tracker_params = {
            'threshold': 127,
            'min_area': 50.0,
            'max_move': 30.0,
        }
        self.play_speed = 1.0  # frames per timer tick; can be negative for reverse play.

        self.tracker = None  # ParticleTracker, will be created when video loads

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.advance_frame)

        self.setup_ui()
        self.create_menu()

    # --------------------------
    # UI Setup
    # --------------------------
    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left: Video Display and Timeline slider
        left_layout = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel("Video Display")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        left_layout.addWidget(self.video_label)

        # Timeline slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setValue(0)
        self.slider.sliderMoved.connect(self.slider_moved)
        left_layout.addWidget(self.slider)

        # Playback control buttons
        playback_layout = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.start_playback)
        playback_layout.addWidget(self.play_button)
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_playback)
        playback_layout.addWidget(self.pause_button)
        left_layout.addLayout(playback_layout)

        main_layout.addLayout(left_layout, stretch=3)

        # Right: Property grid for tracking parameters and play speed.
        prop_widget = QtWidgets.QWidget()
        prop_layout = QtWidgets.QFormLayout(prop_widget)

        # Tracking threshold
        self.threshold_spin = QtWidgets.QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(self.tracker_params['threshold'])
        self.threshold_spin.valueChanged.connect(self.update_parameters)
        prop_layout.addRow("Threshold", self.threshold_spin)

        # Minimum Area
        self.min_area_spin = QtWidgets.QDoubleSpinBox()
        self.min_area_spin.setRange(1, 10000)
        self.min_area_spin.setValue(self.tracker_params['min_area'])
        self.min_area_spin.valueChanged.connect(self.update_parameters)
        prop_layout.addRow("Min Area", self.min_area_spin)

        # Maximum allowed movement (max_move)
        self.max_move_spin = QtWidgets.QDoubleSpinBox()
        self.max_move_spin.setRange(1, 500)
        self.max_move_spin.setValue(self.tracker_params['max_move'])
        self.max_move_spin.valueChanged.connect(self.update_parameters)
        prop_layout.addRow("Max Move", self.max_move_spin)

        # Play Speed
        self.play_speed_spin = QtWidgets.QDoubleSpinBox()
        self.play_speed_spin.setRange(-10, 10)
        self.play_speed_spin.setSingleStep(0.5)
        self.play_speed_spin.setValue(self.play_speed)
        self.play_speed_spin.valueChanged.connect(self.update_play_speed)
        prop_layout.addRow("Play Speed", self.play_speed_spin)

        main_layout.addWidget(prop_widget, stretch=1)

    def create_menu(self):
        # Create a menu bar with File menu.
        file_menu = self.menuBar().addMenu("File")

        load_action = QtGui.QAction("Load Video", self)
        load_action.triggered.connect(self.load_video)
        file_menu.addAction(load_action)

        save_video_action = QtGui.QAction("Save Annotated Video", self)
        save_video_action.triggered.connect(self.save_annotated_video)
        file_menu.addAction(save_video_action)

        save_json_action = QtGui.QAction("Save Tracking JSON", self)
        save_json_action.triggered.connect(self.save_tracking_json)
        file_menu.addAction(save_json_action)

    # --------------------------
    # Parameter Updates
    # --------------------------
    def update_parameters(self):
        # Update the tracking parameters from property grid. If a video is loaded, re-run processing.
        self.tracker_params['threshold'] = self.threshold_spin.value()
        self.tracker_params['min_area'] = self.min_area_spin.value()
        self.tracker_params['max_move'] = self.max_move_spin.value()
        if self.video_frames:
            self.process_all_frames()
            self.update_frame_display()

    def update_play_speed(self):
        self.play_speed = self.play_speed_spin.value()

    # --------------------------
    # Video Loading and Processing
    # --------------------------
    def load_video(self):
        # Open a file dialog to choose a video file.
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.avi *.mp4 *.mov *.mkv)"
        )
        if not filename:
            return

        # Read video frames using OpenCV.
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to open video file.")
            return

        self.video_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.video_frames.append(frame)
        cap.release()

        self.total_frames = len(self.video_frames)
        if self.total_frames == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "No frames found in the video.")
            return

        self.slider.setMaximum(self.total_frames - 1)
        self.current_frame_index = 0

        # Process all frames to perform particle tracking.
        self.process_all_frames()
        self.update_frame_display()

    def process_all_frames(self):
        # Initialize a new tracker.
        self.tracker = ParticleTracker(self.tracker_params)
        # Process each frame sequentially.
        for idx, frame in enumerate(self.video_frames):
            self.tracker.process_frame(frame, idx)

    # --------------------------
    # Frame display and drawing
    # --------------------------
    def update_frame_display(self):
        if not self.video_frames:
            return

        # Get a copy of the current frame.
        frame = self.video_frames[self.current_frame_index].copy()

        # Draw particle tracks up to the current frame.
        if self.tracker:
            tracks = self.tracker.get_tracks()
            for pid, points in tracks.items():
                # Filter points up to the current frame.
                pts = [ (f, pos) for f, pos in points if f <= self.current_frame_index ]
                if len(pts) < 1:
                    continue
                # Draw the track as connected circles/lines.
                for i, (f, pos) in enumerate(pts):
                    color = get_rainbow_color(f, self.total_frames)
                    cv2.circle(frame, pos, 3, color, -1)
                    if i > 0:
                        prev_pos = pts[i-1][1]
                        cv2.line(frame, prev_pos, pos, color, 2)

        # Convert to RGB for Qt display.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_index)
        self.slider.blockSignals(False)

    def slider_moved(self, position):
        self.current_frame_index = position
        self.update_frame_display()

    # --------------------------
    # Playback control
    # --------------------------
    def start_playback(self):
        self.timer.start(30)  # timer ticks in milliseconds

    def pause_playback(self):
        self.timer.stop()

    def advance_frame(self):
        # Advance (or rewind, if play_speed negative) the current frame.
        self.current_frame_index += int(self.play_speed)
        if self.current_frame_index >= self.total_frames:
            self.current_frame_index = self.total_frames - 1
            self.timer.stop()
        elif self.current_frame_index < 0:
            self.current_frame_index = 0
            self.timer.stop()
        self.update_frame_display()

    # --------------------------
    # Video / JSON Saving
    # --------------------------
    def save_annotated_video(self):
        if not self.video_frames:
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Annotated Video", "", "AVI Video (*.avi)"
        )
        if not filename:
            return

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        height, width, _ = self.video_frames[0].shape
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        for idx, orig_frame in enumerate(self.video_frames):
            frame = orig_frame.copy()
            if self.tracker:
                tracks = self.tracker.get_tracks()
                for pid, points in tracks.items():
                    pts = [ (f, pos) for f, pos in points if f <= idx ]
                    if len(pts) < 1:
                        continue
                    for i, (f, pos) in enumerate(pts):
                        color = get_rainbow_color(f, self.total_frames)
                        cv2.circle(frame, pos, 3, color, -1)
                        if i > 0:
                            prev_pos = pts[i-1][1]
                            cv2.line(frame, prev_pos, pos, color, 2)
            out.write(frame)
        out.release()
        QtWidgets.QMessageBox.information(self, "Info", "Annotated video saved successfully.")

    def save_tracking_json(self):
        if not self.tracker:
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Tracking JSON", "", "JSON Files (*.json)"
        )
        if not filename:
            return
        json_str = self.tracker.to_json()
        with open(filename, 'w') as f:
            f.write(json_str)
        QtWidgets.QMessageBox.information(self, "Info", "Tracking JSON saved successfully.")

# --------------------------
# Main Entry Point
# --------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()