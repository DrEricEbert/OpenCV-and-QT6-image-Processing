import sys
import json
import numpy as np
import cv2
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QPen
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QPushButton, QLabel, QFileDialog, QComboBox,
                             QTableWidget, QTableWidgetItem, QDockWidget, QFormLayout,
                             QDoubleSpinBox, QSpinBox)

class RainbowColor:
    @staticmethod
    def get_color(age, max_age=100):
        hue = (age % max_age) * 360 / max_age
        return QColor.fromHsvF(hue / 360, 1.0, 1.0)

class ParticleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.params = {
            'threshold': 25,
            'min_distance': 15,
            'max_points': 200,
            'track_length': 20
        }

    def process_frame(self, frame_gray):
        # Detect features using goodFeaturesToTrack
        features = cv2.goodFeaturesToTrack(
            frame_gray,
            maxCorners=self.params['max_points'],
            qualityLevel=0.01,
            minDistance=self.params['min_distance']
        )

        new_tracks = {}
        if features is not None:
            for x, y in features.reshape(-1, 2):
                # Check distance to existing tracks
                too_close = any(
                    np.hypot(x - t['x'], y - t['y']) < self.params['min_distance']
                    for t in self.tracks.values()
                )
                if not too_close:
                    new_tracks[self.next_id] = {'x': x, 'y': y, 'age': 0, 'history': []}
                    self.next_id += 1

        # Update existing tracks with optical flow
        if len(self.tracks) > 0:
            old_points = np.array([[t['x'], t['y']] for t in self.tracks.values()]).reshape(-1, 1, 2)
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, old_points, None
            )
            
            for (id, track), (x, y), valid in zip(self.tracks.items(), new_points.reshape(-1, 2), status):
                if valid:
                    track['x'] = x
                    track['y'] = y
                    track['age'] += 1
                    track['history'].append((x, y))
                    if len(track['history']) > self.params['track_length']:
                        track['history'].pop(0)
                    new_tracks[id] = track

        self.tracks = new_tracks
        self.prev_gray = frame_gray.copy()
        return self.tracks

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.tracks = {}
        self.current_frame = 0

    def set_frame(self, frame, tracks):
        self.tracks = tracks
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        self.image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)#.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
        for track_id, track in self.tracks.items():
            color = RainbowColor.get_color(track['age'])
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            # Draw track history
            history = track['history']
            for i in range(1, len(history)):
                x1, y1 = history[i-1]
                x2, y2 = history[i]
                painter.drawLine(int(x1), int(y1), int(x2),int(y2))
            
            # Draw current position
            painter.drawEllipse(int(track['x']), int(track['y']), 5, 5)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Particle Tracker")
        self.setGeometry(100, 100, 1280, 720)
        
        # Video and tracker setup
        self.cap = None
        self.tracker = ParticleTracker()
        self.current_frame = 0
        self.playing = False
        self.play_speed = 1
        
        # Create GUI elements
        self.create_menu()
        self.create_main_widgets()
        self.create_dock_widgets()
        
        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def create_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        
        open_action = file_menu.addAction("Open Video")
        open_action.triggered.connect(self.open_video)
        
        save_action = file_menu.addAction("Save Tracks")
        save_action.triggered.connect(self.save_tracks)
        
        load_action = file_menu.addAction("Load Tracks")
        load_action.triggered.connect(self.load_tracks)

    def create_main_widgets(self):
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        # Video display
        self.video_widget = VideoWidget()
        layout.addWidget(self.video_widget)
        
        # Timeline slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.set_frame_position)
        layout.addWidget(self.slider)
        
        # Controls
        control_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_button)
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "2x", "4x"])
        self.speed_combo.currentIndexChanged.connect(self.update_speed)
        control_layout.addWidget(self.speed_combo)
        
        layout.addLayout(control_layout)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def create_dock_widgets(self):
        # Parameters dock
        dock = QDockWidget("Tracking Parameters", self)
        table = QTableWidget()
        table.setColumnCount(2)
        table.setRowCount(4)
        
        params = [
            ("Threshold", "threshold", 0, 255),
            ("Min Distance", "min_distance", 1, 100),
            ("Max Points", "max_points", 1, 500),
            ("Track Length", "track_length", 1, 100)
        ]
        
        for i, (name, key, min_val, max_val) in enumerate(params):
            table.setItem(i, 0, QTableWidgetItem(name))
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(self.tracker.params[key])
            spin.valueChanged.connect(lambda v, k=key: self.update_param(k, v))
            table.setCellWidget(i, 1, spin)
        
        dock.setWidget(table)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def update_param(self, key, value):
        self.tracker.params[key] = value

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if filename:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(filename)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(self.total_frames - 1)
            self.current_frame = 0
            self.tracker = ParticleTracker()
            self.update_frame()

    def update_frame(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if hasattr(self.tracker, 'prev_gray'):
                    tracks = self.tracker.process_frame(frame_gray)
                else:
                    self.tracker.prev_gray = frame_gray.copy()
                    tracks = {}
                
                # Draw tracks on the frame
                display_frame = frame.copy()
                self.video_widget.set_frame(display_frame, tracks)
                self.slider.setValue(self.current_frame)
                
                if self.playing:
                    self.current_frame += self.play_speed
                    if self.current_frame >= self.total_frames:
                        self.current_frame = 0
            else:
                self.timer.stop()
                self.playing = False
                self.play_button.setText("Play")

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start(100 // self.play_speed)
        else:
            self.timer.stop()

    def update_speed(self, index):
        speeds = [0.5, 1, 2, 4]
        self.play_speed = speeds[index]
        if self.playing:
            self.timer.setInterval(100 // self.play_speed)

    def set_frame_position(self, position):
        self.current_frame = position
        self.update_frame()

    def save_tracks(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Tracks", "", "JSON Files (*.json)"
        )
        if filename:
            data = {
                'tracks': {
                    str(k): {
                        'x': int(v['x']),
                        'y': int(v['y']),
                        'age': int(v['age']),
                        'history':[(int(p[0]), int(p[1])) for p in v['history']]
                    } for k, v in self.tracker.tracks.items()
                },
                'next_id': self.tracker.next_id,
                'params': self.tracker.params
            }
            
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    return super().default(obj)
            
            with open(filename, 'w') as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)

    def load_tracks(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Tracks", "", "JSON Files (*.json)"
        )
        if filename:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.tracker.tracks = {
                    int(k): {
                        'x': v['x'],
                        'y': v['y'],
                        'age': v['age'],
                        'history': [(p[0], p[1]) for p in v['history']]
                    } for k, v in data['tracks'].items()
                }
                self.tracker.next_id = data['next_id']
                self.tracker.params.update(data['params'])
               # self.update_frame()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
