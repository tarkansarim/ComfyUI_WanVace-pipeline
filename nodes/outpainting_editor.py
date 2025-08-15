"""
Outpainting Editor - Standalone window for advanced outpainting canvas manipulation
"""
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                            QGroupBox, QRadioButton, QButtonGroup, QSpinBox,
                            QSlider, QComboBox, QCheckBox, QFileDialog,
                            QMessageBox, QWidget, QDialogButtonBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSettings
from PyQt5.QtGui import QIcon
try:
    from .outpainting_preview_widget import OutpaintingPreviewWidget
except ImportError:
    # When running as standalone script, use absolute import
    from outpainting_preview_widget import OutpaintingPreviewWidget


class OutpaintingEditor(QDialog):
    """Advanced outpainting editor dialog"""
    
    def __init__(self, parent=None, video_frames=None, video_path=None, 
                 canvas_settings=None, scale_canvas=False):
        super().__init__(parent)
        self.video_frames = video_frames if video_frames is not None else []
        self.video_path = video_path
        self.canvas_settings = canvas_settings
        self.preview_widget = None
        self.result_settings = None
        
        # Setup UI
        self.setWindowTitle("Outpainting Editor")
        self.setModal(True)
        self.setMinimumSize(1200, 800)
        
        # Initialize UI components
        self.init_ui()
        
        # Load frames if provided
        if self.video_frames:
            self.preview_widget.set_frames(self.video_frames)
            self.setup_playback_controls()
            
        # Apply canvas settings if provided
        if self.canvas_settings:
            self.apply_canvas_settings(self.canvas_settings)
            
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Aspect ratio controls
        self.aspect_ratio_group = QButtonGroup()
        self.aspect_locked_cb = QRadioButton("Aspect Locked")
        self.aspect_locked_cb.setChecked(True)
        self.aspect_locked_cb.toggled.connect(self.on_aspect_lock_changed)
        self.aspect_ratio_group.addButton(self.aspect_locked_cb)
        controls_layout.addWidget(self.aspect_locked_cb)
        
        self.aspect_free_cb = QRadioButton("Free")
        self.aspect_free_cb.toggled.connect(self.on_aspect_lock_changed)
        self.aspect_ratio_group.addButton(self.aspect_free_cb)
        controls_layout.addWidget(self.aspect_free_cb)
        
        controls_layout.addWidget(QLabel("  |  "))
        
        # View mode toggle
        self.view_mode_group = QButtonGroup()
        self.video_view_rb = QRadioButton("Video")
        self.video_view_rb.setChecked(True)
        self.video_view_rb.toggled.connect(self.on_view_mode_changed)
        self.view_mode_group.addButton(self.video_view_rb)
        controls_layout.addWidget(self.video_view_rb)
        
        self.mask_view_rb = QRadioButton("Mask")
        self.mask_view_rb.toggled.connect(self.on_view_mode_changed)
        self.view_mode_group.addButton(self.mask_view_rb)
        controls_layout.addWidget(self.mask_view_rb)
        
        controls_layout.addWidget(QLabel("  |  "))
        
        # Allow cropping checkbox
        self.allow_cropping_cb = QCheckBox("Allow Cropping")
        self.allow_cropping_cb.setChecked(False)
        self.allow_cropping_cb.toggled.connect(self.on_allow_cropping_changed)
        self.allow_cropping_cb.setToolTip("Enable cropping the original video with border handles")
        controls_layout.addWidget(self.allow_cropping_cb)
        
        controls_layout.addWidget(QLabel("  |  "))
        
        # Feather amount
        controls_layout.addWidget(QLabel("Feather:"))
        self.feather_spin = QSpinBox()
        self.feather_spin.setMinimum(0)
        self.feather_spin.setMaximum(100)
        self.feather_spin.setValue(0)
        self.feather_spin.setSuffix(" px")
        self.feather_spin.valueChanged.connect(self.on_feather_changed)
        controls_layout.addWidget(self.feather_spin)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Aspect ratio presets
        aspect_ratio_layout = QHBoxLayout()
        aspect_ratio_layout.addWidget(QLabel("Aspect Ratio Presets:"))
        
        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItem("Custom", None)
        self.aspect_ratio_combo.addItem("16:9 (1920×1080)", (16, 9))
        self.aspect_ratio_combo.addItem("16:9 (3840×2160)", (16, 9, 3840, 2160))
        self.aspect_ratio_combo.addItem("21:9 (2560×1080)", (21, 9))
        self.aspect_ratio_combo.addItem("21:9 (3440×1440)", (21, 9, 3440, 1440))
        self.aspect_ratio_combo.addItem("4:3 (1024×768)", (4, 3))
        self.aspect_ratio_combo.addItem("4:3 (1600×1200)", (4, 3, 1600, 1200))
        self.aspect_ratio_combo.addItem("1:1 Square", (1, 1))
        self.aspect_ratio_combo.addItem("9:16 Vertical (1080×1920)", (9, 16))
        self.aspect_ratio_combo.addItem("3:4 Vertical (768×1024)", (3, 4))
        self.aspect_ratio_combo.addItem("2.35:1 Cinema", (2.35, 1))
        self.aspect_ratio_combo.addItem("2.39:1 Anamorphic", (2.39, 1))
        self.aspect_ratio_combo.addItem("1.85:1 Widescreen", (1.85, 1))
        self.aspect_ratio_combo.currentIndexChanged.connect(self.on_aspect_ratio_preset_changed)
        self.aspect_ratio_combo.setMinimumWidth(200)
        aspect_ratio_layout.addWidget(self.aspect_ratio_combo)
        
        # Center video button
        self.center_video_btn = QPushButton("Center Video")
        self.center_video_btn.clicked.connect(self.center_video)
        self.center_video_btn.setToolTip("Center the video in the canvas")
        aspect_ratio_layout.addWidget(self.center_video_btn)
        
        # Reset canvas button
        self.reset_canvas_btn = QPushButton("Reset Canvas")
        self.reset_canvas_btn.clicked.connect(self.reset_canvas)
        self.reset_canvas_btn.setToolTip("Reset canvas to original video size")
        aspect_ratio_layout.addWidget(self.reset_canvas_btn)
        
        aspect_ratio_layout.addStretch()
        layout.addLayout(aspect_ratio_layout)
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶")
        self.play_btn.setMaximumWidth(40)
        self.play_btn.setToolTip("Play/Pause")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        playback_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("■")
        self.stop_btn.setMaximumWidth(40)
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)
        playback_layout.addWidget(self.stop_btn)
        
        playback_layout.addWidget(QLabel("Frame:"))
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        self.frame_slider.setEnabled(False)
        playback_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(60)
        playback_layout.addWidget(self.frame_label)
        
        layout.addLayout(playback_layout)
        
        # Main preview widget
        self.preview_widget = OutpaintingPreviewWidget()
        self.preview_widget.setShowMask(False)
        self.preview_widget.frameChanged.connect(self.on_frame_auto_changed)
        layout.addWidget(self.preview_widget)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("QLabel { color: #666; padding: 5px; }")
        layout.addWidget(self.status_label)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def setup_playback_controls(self):
        """Setup playback controls based on loaded frames"""
        if self.video_frames:
            self.frame_slider.setMaximum(len(self.video_frames) - 1)
            self.frame_slider.setEnabled(True)
            self.frame_label.setText(f"1 / {len(self.video_frames)}")
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.update_status(f"Loaded {len(self.video_frames)} frames")
            
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
        
    def apply_canvas_settings(self, settings):
        """Apply provided canvas settings"""
        if 'canvas_width' in settings and 'canvas_height' in settings:
            self.preview_widget.set_canvas_size(settings['canvas_width'], settings['canvas_height'])
        if 'video_x' in settings and 'video_y' in settings:
            self.preview_widget.video_x = settings['video_x']
            self.preview_widget.video_y = settings['video_y']
        if 'feather_amount' in settings:
            self.feather_spin.setValue(settings['feather_amount'])
            
    def on_aspect_lock_changed(self):
        """Handle aspect lock toggle"""
        self.preview_widget.set_aspect_locked(self.aspect_locked_cb.isChecked())
        if not self.aspect_locked_cb.isChecked():
            self.aspect_ratio_combo.setCurrentIndex(0)
            
    def on_view_mode_changed(self):
        """Handle view mode toggle"""
        self.preview_widget.setShowMask(self.mask_view_rb.isChecked())
        
    def on_allow_cropping_changed(self):
        """Handle allow cropping toggle"""
        self.preview_widget.set_allow_cropping(self.allow_cropping_cb.isChecked())
        
    def on_feather_changed(self, value):
        """Handle feather amount change"""
        self.preview_widget.set_feather_amount(value)
        
    def on_aspect_ratio_preset_changed(self, index):
        """Handle aspect ratio preset selection"""
        preset_data = self.aspect_ratio_combo.itemData(index)
        
        if preset_data is None:
            return
            
        # Get current video dimensions
        if self.video_frames:
            video_height, video_width = self.video_frames[0].shape[:2]
        else:
            video_width, video_height = 1920, 1080
            
        if len(preset_data) >= 4:
            # Specific resolution provided
            target_width = preset_data[2]
            target_height = preset_data[3]
        else:
            # Calculate based on aspect ratio
            aspect_width = preset_data[0]
            aspect_height = preset_data[1]
            aspect_ratio = aspect_width / aspect_height
            
            # Scale to fit video while maintaining aspect ratio
            video_aspect = video_width / video_height
            
            if video_aspect > aspect_ratio:
                # Video is wider - fit to width
                target_width = video_width
                target_height = int(video_width / aspect_ratio)
            else:
                # Video is taller - fit to height
                target_height = video_height
                target_width = int(video_height * aspect_ratio)
                
        # Apply the new canvas size
        self.preview_widget.set_canvas_size(target_width, target_height)
        
        # Set the preset aspect ratio
        preset_aspect = target_width / target_height if target_height > 0 else 1.0
        self.preview_widget.setPresetAspect(preset_aspect)
        
        # Ensure aspect lock is checked
        if not self.aspect_locked_cb.isChecked():
            self.aspect_locked_cb.setChecked(True)
            
    def center_video(self):
        """Center the video in the current canvas"""
        if self.preview_widget.video_width > 0 and self.preview_widget.video_height > 0:
            self.preview_widget.video_x = (self.preview_widget.canvas_width - self.preview_widget.video_width) // 2
            self.preview_widget.video_y = (self.preview_widget.canvas_height - self.preview_widget.video_height) // 2
            self.preview_widget.update()
            self.update_status("Video centered in canvas")
            
    def reset_canvas(self):
        """Reset canvas to original video size"""
        if self.preview_widget.video_width > 0 and self.preview_widget.video_height > 0:
            self.preview_widget.set_canvas_size(self.preview_widget.video_width, 
                                              self.preview_widget.video_height)
            self.preview_widget.video_x = 0
            self.preview_widget.video_y = 0
            self.preview_widget.update()
            self.aspect_ratio_combo.setCurrentIndex(0)  # Reset to Custom
            self.update_status("Canvas reset to video size")
            
    def toggle_playback(self):
        """Toggle video playback"""
        if self.preview_widget.is_playing:
            self.preview_widget.pause()
            self.play_btn.setText("▶")
        else:
            self.preview_widget.play()
            self.play_btn.setText("⏸")
            
    def stop_playback(self):
        """Stop video playback"""
        self.preview_widget.stop()
        self.play_btn.setText("▶")
        self.frame_slider.setValue(0)
        
    def on_frame_changed(self, value):
        """Handle manual frame slider change"""
        self.preview_widget.set_frame(value)
        if self.video_frames:
            self.frame_label.setText(f"{value + 1} / {len(self.video_frames)}")
            
    def on_frame_auto_changed(self, frame_idx):
        """Handle automatic frame change from playback"""
        self.frame_slider.setValue(frame_idx)
        if self.video_frames:
            self.frame_label.setText(f"{frame_idx + 1} / {len(self.video_frames)}")
            
    def get_canvas_settings(self):
        """Get the final canvas settings"""
        return {
            'canvas_width': self.preview_widget.canvas_width,
            'canvas_height': self.preview_widget.canvas_height,
            'video_x': self.preview_widget.video_x,
            'video_y': self.preview_widget.video_y,
            'background_color': [0, 0, 0],  # Currently always black
            'feather_amount': self.feather_spin.value()
        }
        
    def accept(self):
        """Override accept to save results"""
        self.result_settings = self.get_canvas_settings()
        super().accept()
        
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Space:
            self.toggle_playback()
        elif event.key() == Qt.Key_C and event.modifiers() & Qt.ControlModifier:
            self.center_video()
        elif event.key() == Qt.Key_R and event.modifiers() & Qt.ControlModifier:
            self.reset_canvas()
        else:
            super().keyPressEvent(event)