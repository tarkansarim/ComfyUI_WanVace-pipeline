# Extracted mask editor components from mask_editor_standalone.py
# This file contains only the mask editing widgets without video processing

import sys
import os
import json
import re
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QSpinBox, QDoubleSpinBox, QFileDialog, QProgressBar, QTextEdit,
                             QGroupBox, QMessageBox, QRadioButton, QButtonGroup, QSlider,
                             QMenu, QAction, QComboBox, QCheckBox, QScrollArea, QDialog,
                             QGridLayout, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QRect, QRectF, QTimer, QPoint, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush, QPixmap, QImage, QLinearGradient, QRadialGradient, QPainterPath, QCursor, QIcon
import cv2
import numpy as np
from collections import deque
import time

def natural_sort_key(text):
    """Generate a key for natural sorting that handles numbers in filenames correctly"""
    parts = []
    for part in re.split(r'(\d+)', text):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part.lower())
    return parts

# Classic light theme color palette (from earlier version)
MODERN_COLORS = {
    'primary': '#0066cc',       # Classic blue
    'primary_light': '#3399ff', # Light blue
    'primary_dark': '#004499',  # Dark blue
    'secondary': '#66b3ff',     # Sky blue
    'background': '#f0f0f0',    # Light gray background
    'surface': '#ffffff',       # White
    'surface_light': '#fafafa', # Very light gray
    'border': '#cccccc',        # Gray border
    'text': '#000000',          # Black text
    'text_secondary': '#666666', # Gray text
    'error': '#cc0000',         # Red
    'success': '#008800',       # Green
    'warning': '#ff8800',       # Orange
}

# Create a base64 encoded checkmark image
def create_checkmark_base64():
    """Create a base64 encoded white checkmark image"""
    # Create a 16x16 image with transparent background
    img = QImage(16, 16, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    
    # Draw checkmark
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor("white"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    
    # Draw checkmark path
    path = QPainterPath()
    path.moveTo(3, 8)
    path.lineTo(6, 11)
    path.lineTo(13, 4)
    
    painter.drawPath(path)
    painter.end()
    
    # Convert to base64
    import io
    import base64
    buffer = io.BytesIO()
    img.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# Try to create checkmark, fallback to empty if it fails
try:
    CHECKMARK_BASE64 = create_checkmark_base64()
except:
    # Fallback: simple white checkmark PNG base64
    CHECKMARK_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAA7AAAAOwBeShxvQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAC5SURBVDiN1ZMxDoJAEEXfgIgFC7FoNLGz8QJcwAtwAS7ABbiAF6CzsZFGLCxIXIsFuyywCjHxJ5PJzPz3dzKZDXxL2kCS1AOGQLcGvJK0A9a+7y9qtQsh7oVheBZCXBuF0zS9KKXu0WvTNNlaRBiGp1qtVozj+NIsHIbhuVarFY7jeGkF/zqiOI7npQddlFrAEJgBTn6lAK1W66SUepbv4j0d13U92z3fVxiGV3lQXIAOMJGnELz5AL7UOu3s7teRAAAAAElFTkSuQmCC"

# Modern stylesheet
MODERN_STYLE = f"""
QWidget {{
    background-color: {MODERN_COLORS['background']};
    color: {MODERN_COLORS['text']};
    font-family: 'Segoe UI', 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
}}

QMainWindow {{
    background-color: {MODERN_COLORS['background']};
}}

QGroupBox {{
    background-color: {MODERN_COLORS['surface']};
    border: 1px solid {MODERN_COLORS['border']};
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
    padding-bottom: 4px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px 0 8px;
    color: {MODERN_COLORS['text']};
    background-color: {MODERN_COLORS['background']};
}}

QPushButton {{
    background-color: {MODERN_COLORS['primary']};
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 500;
    color: white;
    min-height: 18px;
}}

QPushButton:hover {{
    background-color: {MODERN_COLORS['primary_light']};
}}

QPushButton:pressed {{
    background-color: {MODERN_COLORS['primary_dark']};
}}

QPushButton:disabled {{
    background-color: {MODERN_COLORS['border']};
    color: {MODERN_COLORS['text_secondary']};
}}

QPushButton#process_btn {{
    background-color: {MODERN_COLORS['success']};
    font-size: 16px;
    font-weight: 600;
    padding: 12px 24px;
}}

QPushButton#process_btn:hover {{
    background-color: #2ecc71;
}}

QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {MODERN_COLORS['surface_light']};
    border: 1px solid {MODERN_COLORS['border']};
    border-radius: 4px;
    padding: 6px 10px;
    color: {MODERN_COLORS['text']};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {MODERN_COLORS['primary']};
    outline: none;
}}

QRadioButton {{
    color: {MODERN_COLORS['text']};
}}

QTextEdit {{
    background-color: {MODERN_COLORS['surface']};
    border: 1px solid {MODERN_COLORS['border']};
    border-radius: 6px;
    padding: 8px;
    color: {MODERN_COLORS['text']};
}}

QProgressBar {{
    background-color: {MODERN_COLORS['border']};
    border: 1px solid {MODERN_COLORS['border']};
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {MODERN_COLORS['primary']};
    border-radius: 4px;
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: {MODERN_COLORS['surface_light']};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {MODERN_COLORS['primary']};
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
}}

QSlider::handle:horizontal:hover {{
    background: {MODERN_COLORS['primary_light']};
}}

QLabel {{
    color: {MODERN_COLORS['text']};
}}

QMenu {{
    background-color: {MODERN_COLORS['surface']};
    border: 1px solid {MODERN_COLORS['border']};
    border-radius: 6px;
    padding: 4px;
}}

QMenu::item {{
    padding: 8px 24px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {MODERN_COLORS['primary']};
    color: white;
}}

QComboBox {{
    background-color: {MODERN_COLORS['surface']};
    border: 1px solid {MODERN_COLORS['border']};
    border-radius: 4px;
    padding: 6px;
    color: {MODERN_COLORS['text']};
    min-width: 200px;
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {MODERN_COLORS['text']};
    margin-right: 5px;
}}

QComboBox:hover {{
    border: 1px solid {MODERN_COLORS['primary']};
}}

QComboBox QAbstractItemView {{
    background-color: {MODERN_COLORS['surface']};
    border: 1px solid {MODERN_COLORS['border']};
    selection-background-color: {MODERN_COLORS['primary']};
    selection-color: white;
}}

QSpinBox, QDoubleSpinBox {{
    background-color: {MODERN_COLORS['surface']};
    border: 1px solid {MODERN_COLORS['border']};
    border-radius: 4px;
    padding: 4px;
    color: {MODERN_COLORS['text']};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid {MODERN_COLORS['primary']};
}}

QLineEdit {{
    background-color: {MODERN_COLORS['surface']};
    border: 1px solid {MODERN_COLORS['border']};
    border-radius: 4px;
    padding: 6px;
    color: {MODERN_COLORS['text']};
}}

QLineEdit:focus {{
    border: 2px solid {MODERN_COLORS['primary']};
}}

QCheckBox {{
    color: {MODERN_COLORS['text']};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {MODERN_COLORS['border']};
    border-radius: 3px;
    background-color: {MODERN_COLORS['surface']};
}}

QCheckBox::indicator:checked {{
    background-color: {MODERN_COLORS['primary']};
    border-color: {MODERN_COLORS['primary']};
    image: url(data:image/png;base64,{CHECKMARK_BASE64});
}}

QCheckBox::indicator:hover {{
    border-color: {MODERN_COLORS['primary']};
}}
"""

class ModernButton(QPushButton):
    """Custom button with hover animations"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMouseTracking(True)
        self._animation = QPropertyAnimation(self, b"_hover_progress")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._hover_progress = 0.0
        
    def enterEvent(self, event):
        self._animation.setStartValue(self._hover_progress)
        self._animation.setEndValue(1.0)
        self._animation.start()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self._animation.setStartValue(self._hover_progress)
        self._animation.setEndValue(0.0)
        self._animation.start()
        super().leaveEvent(event)
    
    @pyqtProperty(float)
    def _hover_progress(self):
        return self.__hover_progress
    
    @_hover_progress.setter
    def _hover_progress(self, value):
        self.__hover_progress = value
        self.update()
        
    def paintEvent(self, event):
        # Let the stylesheet handle most styling, just add subtle effects
        super().paintEvent(event)
        
        if self._hover_progress > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Add subtle glow on hover
            glow_color = QColor(MODERN_COLORS['primary'])
            glow_color.setAlpha(int(30 * self._hover_progress))
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(glow_color)
            
            rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
            painter.drawRoundedRect(rect, 6, 6)

class ModernCheckBox(QCheckBox):
    """Custom checkbox that uses standard QCheckBox with proper checkmark image"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)


class MaskTimelineWidget(QWidget):
    """Custom timeline widget that shows mask indicators and supports scrubbing"""
    frame_changed = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.total_frames = 0
        self.current_frame = 0
        self.mask_frames = set()
        self.setMouseTracking(True)
        self.hover_frame = -1
        self.is_scrubbing = False
        self.setCursor(Qt.PointingHandCursor)
        
    def paintEvent(self, event):
        if self.total_frames == 0:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        # Calculate dimensions
        width = self.width()
        height = self.height()
        frame_width = width / self.total_frames
        
        # Draw timeline track
        track_top = 20
        track_height = height - 25
        painter.fillRect(0, track_top, width, track_height, QColor(60, 60, 60))
        
        # Draw frame ticks for every 10th frame
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        for i in range(0, self.total_frames, 10):
            x = int(i * frame_width)
            painter.drawLine(x, height - 5, x, height)
        
        # Draw keyframe indicators (red lines)
        painter.setPen(QPen(QColor(255, 100, 100), 3))
        for frame in self.mask_frames:
            x = int(frame * frame_width + frame_width / 2)
            painter.drawLine(x, track_top + 5, x, height - 5)
        
        # Draw current frame indicator (wider and more visible)
        current_x = int(self.current_frame * frame_width + frame_width / 2)
        
        # Draw a triangle indicator at the bottom
        indicator_path = QPainterPath()
        indicator_path.moveTo(current_x - 6, height)
        indicator_path.lineTo(current_x + 6, height)
        indicator_path.lineTo(current_x, height - 8)
        indicator_path.closeSubpath()
        painter.fillPath(indicator_path, QColor(255, 255, 255))
        
        # Draw current frame line
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(current_x, track_top, current_x, height - 8)
        
        # Draw frame number in top left corner
        frame_text = f"Frame {self.current_frame + 1} / {self.total_frames}"
        font = painter.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        
        # Draw text in top area
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(5, 15, frame_text)
        
        # Draw hover indicator and tooltip
        if self.hover_frame >= 0 and self.hover_frame != self.current_frame:
            hover_x = int(self.hover_frame * frame_width + frame_width / 2)
            painter.setPen(QPen(QColor(150, 150, 150), 1, Qt.DashLine))
            painter.drawLine(hover_x, track_top, hover_x, height - 5)
            
            # Show hover frame number as tooltip
            if self.hover_frame in self.mask_frames:
                hover_text = f"Frame {self.hover_frame + 1} (has mask)"
                painter.setPen(QColor(255, 100, 100))
            else:
                hover_text = f"Frame {self.hover_frame + 1}"
                painter.setPen(QColor(200, 200, 200))
            
            font.setPointSize(10)
            font.setBold(False)
            painter.setFont(font)
            text_rect = painter.fontMetrics().boundingRect(hover_text)
            tooltip_x = min(hover_x + 5, width - text_rect.width() - 5)
            painter.drawText(tooltip_x, track_top - 5, hover_text)
    
    def mouseMoveEvent(self, event):
        if self.total_frames > 0:
            frame_width = self.width() / self.total_frames
            frame = int(event.x() / frame_width)
            frame = max(0, min(frame, self.total_frames - 1))
            
            if self.is_scrubbing and frame != self.current_frame:
                self.current_frame = frame
                self.frame_changed.emit(frame)
                self.update()
            else:
                self.hover_frame = frame
                self.update()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.total_frames > 0:
            self.is_scrubbing = True
            frame_width = self.width() / self.total_frames
            frame = int(event.x() / frame_width)
            frame = max(0, min(frame, self.total_frames - 1))
            if frame != self.current_frame:
                self.current_frame = frame
                self.frame_changed.emit(frame)
                self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_scrubbing = False
            # Restore focus to mask widget if parent editor exists
            if hasattr(self, 'parent_editor') and hasattr(self.parent_editor, 'mask_widget'):
                self.parent_editor.mask_widget.setFocus()
    
    def leaveEvent(self, event):
        self.hover_frame = -1
        self.is_scrubbing = False
        self.update()
        
    def setCurrentFrame(self, frame):
        """Set current frame from external source"""
        if 0 <= frame < self.total_frames:
            self.current_frame = frame
            self.update()


class InpaintingMaskEditor(QDialog):
    """Advanced mask editor window for inpainting with painting and shape animation"""
    
    def __init__(self, video_frames, parent=None, initial_mode=None):
        super().__init__(parent)
        self.video_frames = video_frames
        self.current_frame_index = 0
        self.mask_frames = [np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8) for frame in video_frames]
        
        # Load settings
        self.settings = QSettings('WAN', 'VideoFrameProcessor')
        
        # Use settings for mode if no initial_mode provided
        if initial_mode:
            self.initial_mode = initial_mode
        else:
            self.initial_mode = self.settings.value('mask_editor_mode', 'brush')
        
        # Drawing properties
        self.drawing_mode = None  # "brush", "shape", "eraser" - will be set later
        self.brush_size = 20
        self.is_drawing = False
        self.last_point = None
        
        # Shape animation properties
        self.shapes = {}  # Dict of frame -> shape data
        self.selected_shape = None
        self.shape_keyframes = {}  # Dict of frame -> list of shapes with vertices
        self.current_shape_stroke = []  # Current stroke being drawn for shape mode
        self.is_drawing_shape = False
        self.selected_vertex = None
        self.warp_mode = False
        self.relax_mode = False
        
        # Auto-save properties
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_project)
        self.auto_save_enabled = self.settings.value('mask_editor_auto_save', True, type=bool)
        self.auto_save_interval = self.settings.value('mask_editor_auto_save_interval', 300000, type=int)  # 5 minutes
        self.auto_save_path = None
        self.has_unsaved_changes = False
        self.is_discarding = False  # Flag to prevent auto-save during discard
        
        # Video playback properties
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.advance_frame)
        self.is_playing = False
        self.playback_fps = self.settings.value('playback_fps', 12, type=int)  # Load saved FPS
        self.playback_start_frame = 0  # Frame where playback started
        self._manual_navigation = False  # Flag to track if frame change is manual or automatic
        self._keyframe_navigation = False  # Flag to track if navigation is via Alt+Arrow
        
        self.init_ui()
        
        # Start auto-save timer if enabled
        if self.auto_save_enabled:
            self.auto_save_timer.start(self.auto_save_interval)
        
    def init_ui(self):
        self.setWindowTitle("Mask Editor - Arrow keys: navigate | B: brush | Z: zoom | X: toggle paint/erase | Space+drag: pan")
        
        # Get screen geometry and center window at 70% of screen size
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.7)
        height = int(screen.height() * 0.7)
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        self.setGeometry(x, y, width, height)
        
        # Enable keyboard shortcuts
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Top toolbar (compact)
        top_toolbar = QWidget()
        top_toolbar.setMaximumHeight(40)
        toolbar_layout = QHBoxLayout(top_toolbar)
        toolbar_layout.setContentsMargins(5, 2, 5, 2)
        toolbar_layout.setSpacing(5)
        
        # Create tool button style (dark theme)
        tool_button_style = """
            QPushButton {
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
                min-width: 32px;
                min-height: 32px;
                max-width: 32px;
                max-height: 32px;
                background-color: #3a3a3a;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #666;
            }
            QPushButton:checked {
                background-color: #4a90e2;
                border: 2px solid #2e6cb8;
            }
            QPushButton:checked:hover {
                background-color: #5ba0f2;
            }
        """
        
        # Add a label to show current brush mode
        mode_label_style = """
            QLabel {
                background-color: #333;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
            }
        """
        
        # Left side tool panel
        tools_frame = QWidget()
        tools_frame.setMaximumWidth(50)  # Slightly wider to accommodate mode label
        tools_frame.setStyleSheet("QWidget { background-color: #2d2d2d; border-right: 1px solid #555; }")
        tools_layout = QVBoxLayout(tools_frame)
        tools_layout.setContentsMargins(4, 4, 4, 4)
        tools_layout.setSpacing(2)
        
        # Tool buttons in single column
        # REMOVED BUTTON GROUP - IT'S CAUSING ISSUES
        # tool_group = QButtonGroup()
        # tool_group.setExclusive(True)  # Ensure exclusive selection
        
        # Create brush icon
        def create_brush_icon():
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw brush
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QBrush(Qt.black))
            painter.drawEllipse(QRect(10, 10, 12, 12))
            painter.drawLine(16, 16, 24, 24)
            painter.setPen(QPen(Qt.black, 3))
            painter.drawLine(22, 22, 26, 26)
            painter.end()
            return QIcon(pixmap)
        
        # Create eraser icon
        def create_eraser_icon():
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw eraser
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QBrush(Qt.white))
            painter.drawRect(QRect(8, 12, 16, 10))
            painter.setBrush(QBrush(Qt.black))
            painter.drawRect(QRect(8, 8, 16, 4))
            painter.end()
            return QIcon(pixmap)
        
        # Create shape icon
        def create_shape_icon():
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw bezier shape
            painter.setPen(QPen(Qt.black, 2))
            path = QPainterPath()
            path.moveTo(8, 24)
            path.cubicTo(8, 8, 24, 8, 24, 24)
            painter.drawPath(path)
            # Draw control points
            painter.setBrush(QBrush(Qt.black))
            painter.drawEllipse(QRect(6, 22, 4, 4))
            painter.drawEllipse(QRect(22, 22, 4, 4))
            painter.end()
            return QIcon(pixmap)
        
        # Create zoom icon
        def create_zoom_icon():
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw magnifying glass
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRect(6, 6, 14, 14))
            painter.drawLine(18, 18, 24, 24)
            # Draw plus sign
            painter.drawLine(13, 10, 13, 16)
            painter.drawLine(10, 13, 16, 13)
            painter.end()
            return QIcon(pixmap)
        
        # Create liquify icon
        def create_liquify_icon():
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw a wavy grid pattern
            painter.setPen(QPen(Qt.black, 2))
            # Draw wavy vertical lines
            for x in range(8, 25, 8):
                path = QPainterPath()
                path.moveTo(x, 4)
                for y in range(4, 29, 4):
                    offset = 2 * np.sin(y * 0.3)
                    path.lineTo(x + offset, y)
                painter.drawPath(path)
            # Draw wavy horizontal lines
            for y in range(8, 25, 8):
                path = QPainterPath()
                path.moveTo(4, y)
                for x in range(4, 29, 4):
                    offset = 2 * np.sin(x * 0.3)
                    path.lineTo(x, y + offset)
                painter.drawPath(path)
            painter.end()
            return QIcon(pixmap)
        
        # Store icon references for later use
        self.brush_icon = create_brush_icon()
        self.shape_icon = create_shape_icon()
        
        # Combined brush tool (pixel/shape toggle)
        self.brush_btn = QPushButton()
        self.brush_btn.setIcon(self.brush_icon)  # Start with pixel brush icon
        self.brush_btn.setCheckable(True)
        self.brush_btn.setChecked(False)  # Don't check by default
        self.brush_btn.clicked.connect(lambda: self.select_brush_mode())
        self.brush_btn.setToolTip("Brush tool - Pixel brush (B to activate, Shift+B to toggle pixel/shape modes)")
        self.brush_btn.setStyleSheet(tool_button_style)
        # tool_group.addButton(self.brush_btn)  # REMOVED
        tools_layout.addWidget(self.brush_btn)
        
        # Eraser tool
        self.eraser_btn = QPushButton()
        self.eraser_btn.setIcon(create_eraser_icon())
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(lambda: self.select_eraser_mode())
        self.eraser_btn.setToolTip("Erase mask (E)")
        self.eraser_btn.setStyleSheet(tool_button_style)
        # tool_group.addButton(self.eraser_btn)  # REMOVED
        tools_layout.addWidget(self.eraser_btn)
        
        # Zoom tool
        self.zoom_tool_btn = QPushButton()
        self.zoom_tool_btn.setIcon(create_zoom_icon())
        self.zoom_tool_btn.setCheckable(True)
        self.zoom_tool_btn.clicked.connect(lambda: self.select_zoom_mode())
        self.zoom_tool_btn.setToolTip("Zoom tool (Z) - Drag to zoom, Space+drag to pan")
        self.zoom_tool_btn.setStyleSheet(tool_button_style)
        # tool_group.addButton(self.zoom_tool_btn)  # REMOVED
        tools_layout.addWidget(self.zoom_tool_btn)
        
        # Liquify tool
        self.liquify_btn = QPushButton()
        self.liquify_btn.setIcon(create_liquify_icon())
        self.liquify_btn.setCheckable(True)
        self.liquify_btn.clicked.connect(lambda: self.select_liquify_mode())
        self.liquify_btn.setToolTip("Liquify tool (W) - Deform shapes with brush")
        self.liquify_btn.setStyleSheet(tool_button_style)
        self.liquify_btn.setVisible(False)  # Only visible in shape mode
        tools_layout.addWidget(self.liquify_btn)
        
        # Store mode group reference - REMOVED
        # self.mode_group = tool_group
        
        # Small separator
        tools_layout.addSpacing(10)
        
        # Mode indicator
        self.mode_indicator = QLabel("PIXEL")
        self.mode_indicator.setAlignment(Qt.AlignCenter)
        self.mode_indicator.setStyleSheet(mode_label_style)
        tools_layout.addWidget(self.mode_indicator)
        
        # Warp button removed - was displaying as "./ar" due to font issue
        
        tools_layout.addStretch()
        
        # Compact top toolbar items
        toolbar_layout.addWidget(QLabel("Size:"))
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 500)
        # Load saved brush sizes for each tool or use defaults
        self.brush_sizes = {
            'pixel': self.settings.value('mask_editor_brush_size_pixel', 20, type=int),
            'shape': self.settings.value('mask_editor_brush_size_shape', 50, type=int),
            'eraser': self.settings.value('mask_editor_brush_size_eraser', 30, type=int),
            'liquify': self.settings.value('mask_editor_brush_size_liquify', 40, type=int)
        }
        
        # Start with pixel brush size
        saved_brush_size = self.brush_sizes['pixel']
        self.brush_size_slider.setValue(saved_brush_size)
        self.brush_size = saved_brush_size
        self.brush_size_slider.setMaximumWidth(80)
        self.brush_size_slider.valueChanged.connect(self.on_brush_size_changed)
        self.brush_size_slider.setToolTip("Brush size (Alt+Right-click drag)")
        toolbar_layout.addWidget(self.brush_size_slider)
        
        self.brush_size_label = QLabel(str(saved_brush_size))
        self.brush_size_label.setMinimumWidth(25)
        toolbar_layout.addWidget(self.brush_size_label)
        
        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        toolbar_layout.addWidget(sep1)
        
        # Vertex count controls for shape mode
        self.vertex_count_label_text = QLabel("Vertices:")
        self.vertex_count_label_text.setVisible(False)  # Hidden by default
        toolbar_layout.addWidget(self.vertex_count_label_text)
        
        self.vertex_count_slider = QSlider(Qt.Horizontal)
        self.vertex_count_slider.setRange(8, 512)  # 8 to 512 vertices
        # Load saved vertex count or use default
        saved_vertex_count = self.settings.value('mask_editor_vertex_count', 32, type=int)
        self.vertex_count_slider.setValue(saved_vertex_count)
        self.vertex_count_slider.setMaximumWidth(150)
        self.vertex_count_slider.valueChanged.connect(self.on_vertex_count_changed)
        self.vertex_count_slider.sliderPressed.connect(self.on_vertex_slider_pressed)
        self.vertex_count_slider.sliderReleased.connect(self.on_vertex_slider_released)
        self.vertex_count_slider.setVisible(False)  # Hidden by default
        self.vertex_count_slider.setToolTip("Number of vertices for shape complexity")
        toolbar_layout.addWidget(self.vertex_count_slider)
        
        self.vertex_count_label = QLabel(str(saved_vertex_count))
        self.vertex_count_label.setMinimumWidth(30)
        self.vertex_count_label.setVisible(False)  # Hidden by default
        toolbar_layout.addWidget(self.vertex_count_label)
        
        # Separator for vertex controls
        self.vertex_sep = QFrame()
        self.vertex_sep.setFrameShape(QFrame.VLine)
        self.vertex_sep.setFrameShadow(QFrame.Sunken)
        self.vertex_sep.setVisible(False)  # Hidden by default
        toolbar_layout.addWidget(self.vertex_sep)
        
        # Fill holes checkbox (more compact)
        self.fill_holes_check = QCheckBox("Fill holes")
        self.fill_holes_check.setToolTip("Auto-fill closed shapes")
        # Load saved fill holes state
        saved_fill_holes = self.settings.value('mask_editor_fill_holes', False, type=bool)
        self.fill_holes_check.setChecked(saved_fill_holes)
        self.fill_holes_check.stateChanged.connect(self.on_fill_holes_changed)
        toolbar_layout.addWidget(self.fill_holes_check)
        
        # Smooth interpolation checkbox (spline motion between keyframes)
        self.spline_interpolation_check = QCheckBox("Smooth interpolation")
        self.spline_interpolation_check.setToolTip("Use spline curves for smooth motion between keyframes (3+ keyframes)")
        # Load saved spline interpolation state (enabled by default)
        saved_spline_interpolation = self.settings.value('mask_editor_spline_interpolation', True, type=bool)
        self.spline_interpolation_check.setChecked(saved_spline_interpolation)
        self.spline_interpolation_check.stateChanged.connect(self.on_spline_interpolation_changed)
        toolbar_layout.addWidget(self.spline_interpolation_check)
        
        # Smooth shapes checkbox (spline curves for individual shapes)
        self.spline_shapes_check = QCheckBox("Smooth shapes")
        self.spline_shapes_check.setToolTip("Use spline curves for smooth shape rendering with fewer vertices")
        # Load saved spline shapes state (enabled by default)
        saved_spline_shapes = self.settings.value('mask_editor_spline_shapes', True, type=bool)
        self.spline_shapes_check.setChecked(saved_spline_shapes)
        self.spline_shapes_check.stateChanged.connect(self.on_spline_shapes_changed)
        toolbar_layout.addWidget(self.spline_shapes_check)
        
        # Show lattice checkbox for liquify mode
        self.show_lattice_check = QCheckBox("Show lattice")
        self.show_lattice_check.setToolTip("Show deformation lattice grid")
        self.show_lattice_check.setVisible(False)  # Hidden by default
        # Load saved lattice state
        saved_show_lattice = self.settings.value('mask_editor_show_lattice', True, type=bool)
        self.show_lattice_check.setChecked(saved_show_lattice)
        self.show_lattice_check.stateChanged.connect(self.on_show_lattice_changed)
        toolbar_layout.addWidget(self.show_lattice_check)
        
        # Lattice resolution controls for liquify mode
        self.lattice_size_label_text = QLabel("Grid size:")
        self.lattice_size_label_text.setVisible(False)  # Hidden by default
        toolbar_layout.addWidget(self.lattice_size_label_text)
        
        self.lattice_size_slider = QSlider(Qt.Horizontal)
        self.lattice_size_slider.setRange(10, 50)  # 10 to 50 pixels
        # Load saved lattice size or use default
        saved_lattice_size = self.settings.value('mask_editor_lattice_size', 20, type=int)
        self.lattice_size_slider.setValue(saved_lattice_size)
        self.lattice_size_slider.setMaximumWidth(100)
        self.lattice_size_slider.valueChanged.connect(self.on_lattice_size_changed)
        self.lattice_size_slider.setVisible(False)  # Hidden by default
        self.lattice_size_slider.setToolTip("Lattice grid resolution")
        toolbar_layout.addWidget(self.lattice_size_slider)
        
        self.lattice_size_label = QLabel(str(saved_lattice_size))
        self.lattice_size_label.setMinimumWidth(25)
        self.lattice_size_label.setVisible(False)  # Hidden by default
        toolbar_layout.addWidget(self.lattice_size_label)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        toolbar_layout.addWidget(sep2)
        
        # Clear button (compact)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMaximumWidth(50)
        self.clear_btn.clicked.connect(self.clear_current_frame)
        toolbar_layout.addWidget(self.clear_btn)
        
        # Clear all button
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setMaximumWidth(70)
        self.clear_all_btn.clicked.connect(self.clear_all_frames)
        self.clear_all_btn.setToolTip("Clear masks for all frames")
        toolbar_layout.addWidget(self.clear_all_btn)
        
        # Bake liquify button (only visible in liquify mode)
        self.bake_liquify_btn = QPushButton("Bake Liquify")
        self.bake_liquify_btn.setMaximumWidth(100)
        self.bake_liquify_btn.clicked.connect(self.bake_liquify)
        self.bake_liquify_btn.setToolTip("Apply liquify deformation permanently and reset the lattice")
        self.bake_liquify_btn.setVisible(False)  # Hidden by default
        toolbar_layout.addWidget(self.bake_liquify_btn)
        
        toolbar_layout.addStretch()
        
        # Apply to current frame button
        self.apply_current_btn = QPushButton("Apply to Current")
        self.apply_current_btn.clicked.connect(self.apply_to_current_frame)
        self.apply_current_btn.setToolTip("Apply the current mask to the current frame in the main mask buffer")
        toolbar_layout.addWidget(self.apply_current_btn)
        
        # Apply to all frames
        self.apply_all_btn = QPushButton("Apply to All")
        self.apply_all_btn.clicked.connect(self.apply_to_all_frames)
        toolbar_layout.addWidget(self.apply_all_btn)
        
        main_layout.addWidget(top_toolbar)
        
        # Main content area with tools and video views
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(5)
        
        # Add tools frame to left side
        content_layout.addWidget(tools_frame)
        
        # Mask view only (no video view)
        views_layout = QHBoxLayout()
        views_layout.setSpacing(5)
        
        # Mask view (drawing area) - takes full space
        mask_group = QGroupBox("Mask Editor")
        mask_layout = QVBoxLayout()
        mask_layout.setContentsMargins(5, 5, 5, 5)
        self.mask_widget = MaskDrawingWidget()
        self.mask_widget.setMinimumSize(800, 600)  # Larger minimum size since it's the only view
        self.mask_widget.parent_editor = self  # Set parent reference for UI updates
        
        # Initialize mask widget with saved settings
        self.mask_widget.set_brush_size(saved_brush_size)
        self.mask_widget.target_vertex_count = saved_vertex_count
        self.mask_widget.show_lattice = saved_show_lattice
        self.mask_widget.liquify_grid_size = saved_lattice_size
        
        mask_layout.addWidget(self.mask_widget)
        mask_group.setLayout(mask_layout)
        views_layout.addWidget(mask_group, 1)  # Stretch factor 1 - takes all available space
        
        content_layout.addLayout(views_layout, 1)  # Views take all available space
        
        main_layout.addWidget(content_widget, 1)  # Content takes all available space
        
        # Timeline controls with custom widget
        timeline_container = QVBoxLayout()
        
        # Custom timeline widget that shows mask indicators and supports scrubbing
        self.timeline_widget = MaskTimelineWidget()
        self.timeline_widget.setFixedHeight(60)  # Slightly taller for better visibility
        self.timeline_widget.total_frames = len(self.video_frames)
        self.timeline_widget.current_frame = 0
        self.timeline_widget.mask_frames = set()  # Frames that have masks
        self.timeline_widget.parent_editor = self  # Add reference to parent editor
        self.timeline_widget.frame_changed.connect(self.on_timeline_frame_changed)
        timeline_container.addWidget(self.timeline_widget)
        
        # Timeline controls (simplified - no frame slider needed)
        timeline_layout = QHBoxLayout()
        
        # Playback controls
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.play_pause_btn.setToolTip("Play/Pause video playback (Spacebar)")
        timeline_layout.addWidget(self.play_pause_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setToolTip("Stop playback and return to start")
        timeline_layout.addWidget(self.stop_btn)
        
        # FPS control
        self.fps_label = QLabel("FPS:")
        timeline_layout.addWidget(self.fps_label)
        
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setRange(1, 60)
        
        # Load saved FPS value from settings
        saved_fps = self.settings.value('playback_fps', 12, type=int)
        self.fps_spinner.setValue(saved_fps)
        self.playback_fps = saved_fps  # Update the internal FPS value
        
        self.fps_spinner.setMaximumWidth(60)
        self.fps_spinner.setToolTip("Playback frames per second")
        self.fps_spinner.valueChanged.connect(self.on_fps_changed)
        timeline_layout.addWidget(self.fps_spinner)
        
        # Loop checkbox
        self.loop_check = QCheckBox("Loop")
        self.loop_check.setChecked(True)
        self.loop_check.setToolTip("Loop playback when reaching end")
        timeline_layout.addWidget(self.loop_check)
        
        # Separator
        sep_playback = QFrame()
        sep_playback.setFrameShape(QFrame.VLine)
        sep_playback.setFrameShadow(QFrame.Sunken)
        timeline_layout.addWidget(sep_playback)
        
        # Navigation buttons
        
        timeline_layout.addStretch()  # Push copy/paste buttons to the right
        
        # Copy/paste buttons
        self.copy_btn = QPushButton("Copy Mask")
        self.copy_btn.clicked.connect(self.copy_mask)
        timeline_layout.addWidget(self.copy_btn)
        
        self.paste_btn = QPushButton("Paste Mask")
        self.paste_btn.clicked.connect(self.paste_mask)
        self.paste_btn.setEnabled(False)
        timeline_layout.addWidget(self.paste_btn)
        
        timeline_container.addLayout(timeline_layout)
        main_layout.addLayout(timeline_container)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        button_layout.addStretch()
        
        self.save_btn = QPushButton("Save Masks")
        self.save_btn.clicked.connect(self.save_masks)
        button_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_without_prompts)
        button_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(button_layout)
        
        # Initialize display
        self.update_display()
        self.update_navigation_buttons()
        
        # Apply initial mode after a delay to ensure UI is ready
        if hasattr(self, 'initial_mode') and self.initial_mode:
            QTimer.singleShot(200, lambda: self._apply_initial_mode(self.initial_mode))
        else:
            # Default to shape brush mode
            QTimer.singleShot(200, self.select_brush_mode)
        
    def _apply_initial_mode(self, mode):
        """Apply the initial mode with proper button state handling"""
        
        # Force uncheck all buttons first
        self.brush_btn.setChecked(False)
        self.eraser_btn.setChecked(False)
        self.zoom_tool_btn.setChecked(False)
        self.liquify_btn.setChecked(False)
        
        # Force UI update
        QApplication.processEvents()
        
        # Apply the correct mode by calling the selection methods directly
        if mode == "shape":
            # Handle shape mode via brush button now
            self.mask_widget._last_brush_mode = "shape"
            self.select_brush_mode()
            # Force button state update
            self.brush_btn.setChecked(True)
            self.update_brush_button_icon()
            self.brush_btn.update()
            self.brush_btn.repaint()
        elif mode == "eraser":
            self.select_eraser_mode()
            # Force button state update
            self.eraser_btn.setChecked(True)
            self.eraser_btn.update()
            self.eraser_btn.repaint()
        elif mode == "liquify":
            self.select_liquify_mode()
            # Force button state update
            self.liquify_btn.setChecked(True)
            self.liquify_btn.update()
            self.liquify_btn.repaint()
        else:  # brush
            self.select_brush_mode()
            # Force button state update
            self.brush_btn.setChecked(True)
            self.brush_btn.update()
            self.brush_btn.repaint()
        
        # Process events again to ensure UI is updated
        QApplication.processEvents()
        
        # Set initial focus to mask widget for keyboard shortcuts
        self.mask_widget.setFocus()
    
    def update_brush_button_icon(self):
        """Update the brush button icon based on current brush mode"""
        if not hasattr(self, 'brush_btn'):
            return
            
        current_mode = getattr(self.mask_widget, '_last_brush_mode', 'brush')
        if current_mode == "shape":
            self.brush_btn.setIcon(self.shape_icon)
            self.brush_btn.setToolTip("Brush tool - Shape brush (B to activate, Shift+B to toggle pixel/shape modes)")
        else:
            self.brush_btn.setIcon(self.brush_icon)
            self.brush_btn.setToolTip("Brush tool - Pixel brush (B to activate, Shift+B to toggle pixel/shape modes)")
        
    def set_drawing_mode(self, mode):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[InpaintingMaskEditor.set_drawing_mode] Called with mode={mode}, current mode={self.drawing_mode}")
        sys.stdout.flush()  # Force flush stdout
        
        # Store the previous mode for later checks
        previous_mode = self.drawing_mode
        
        # When leaving liquify mode, bake the changes BEFORE changing modes
        if previous_mode == "liquify" and mode != "liquify":
            logger.info(f"[InpaintingMaskEditor.set_drawing_mode] Leaving liquify mode to {mode}, checking if we need to bake")
            # Bake any liquify changes before switching modes
            if hasattr(self.mask_widget, 'bake_liquify_deformation'):
                if (hasattr(self.mask_widget, 'liquify_deformation_field') and 
                    self.mask_widget.liquify_deformation_field is not None and
                    hasattr(self.mask_widget, 'liquify_original_shapes') and
                    self.mask_widget.liquify_original_shapes is not None):
                    # Check if there are any actual deformations to preserve
                    max_deform = np.max(np.abs(self.mask_widget.liquify_deformation_field))
                    logger.info(f"[InpaintingMaskEditor.set_drawing_mode] Max deformation: {max_deform}")
                    if max_deform > 0.01:  # If there's significant deformation
                        logger.info(f"[InpaintingMaskEditor.set_drawing_mode] Baking liquify changes before switching to {mode}")
                        self.mask_widget.bake_liquify_deformation()
                    else:
                        logger.info(f"[InpaintingMaskEditor.set_drawing_mode] No significant deformation to bake")
        
        self.drawing_mode = mode
        self.mask_widget.drawing_mode = mode  # Set directly on the widget too
        self.mask_widget.set_drawing_mode(mode)
        
        # Save mode to settings
        self.settings.setValue('mask_editor_mode', mode)
        logger.info(f"[InpaintingMaskEditor.set_drawing_mode] Saved mode '{mode}' to settings")
        
        # Update navigation buttons to show correct labels
        self.update_navigation_buttons()
        
        # Restore the appropriate brush size for this mode
        if mode == "brush":
            new_size = self.brush_sizes.get('pixel', 20)
        elif mode == "shape":
            new_size = self.brush_sizes.get('shape', 50)
        elif mode == "eraser":
            new_size = self.brush_sizes.get('eraser', 30)
        elif mode == "liquify":
            new_size = self.brush_sizes.get('liquify', 40)
        else:
            new_size = self.brush_size  # Keep current size for other modes
        
        # Update the slider and brush size
        self.brush_size_slider.setValue(new_size)
        self.brush_size = new_size
        self.brush_size_label.setText(f"{new_size}")
        self.mask_widget.set_brush_size(new_size)
        
        # Initialize liquify deformation field when entering liquify mode
        if mode == "liquify":
            if hasattr(self.mask_widget, 'liquify_deformation_field'):
                if self.mask_widget.liquify_deformation_field is None and self.mask_widget.mask is not None:
                    h, w = self.mask_widget.mask.shape[:2]
                    self.mask_widget.liquify_deformation_field = np.zeros((h, w, 2), dtype=np.float32)
            
        
        # When leaving liquify mode, reset the liquify state
        if previous_mode == "liquify" and mode != "liquify":
            # Reset liquify state after baking (baking already happened above)
            if hasattr(self.mask_widget, 'liquify_deformation_field'):
                self.mask_widget.liquify_deformation_field = None
            if hasattr(self.mask_widget, 'liquify_original_shapes'):
                self.mask_widget.liquify_original_shapes = None
            if hasattr(self.mask_widget, '_temp_deformed_shapes'):
                self.mask_widget._temp_deformed_shapes = None
            logger.info(f"[InpaintingMaskEditor.set_drawing_mode] Reset liquify state after switching to {mode}")
        
        # Update mode indicator
        if mode == "shape":
            self.mode_indicator.setText("SHAPE")
            self.mode_indicator.setStyleSheet("""
                QLabel {
                    background-color: #8b4513;
                    color: white;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }
            """)
        elif mode == "brush":
            self.mode_indicator.setText("PIXEL")
            self.mode_indicator.setStyleSheet("""
                QLabel {
                    background-color: #333;
                    color: white;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }
            """)
        elif mode == "eraser":
            self.mode_indicator.setText("ERASE")
            self.mode_indicator.setStyleSheet("""
                QLabel {
                    background-color: #cc0000;
                    color: white;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }
            """)
        elif mode == "liquify":
            self.mode_indicator.setText("LIQUIFY")
            self.mode_indicator.setStyleSheet("""
                QLabel {
                    background-color: #9400d3;
                    color: white;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }
            """)
        
        # Show/hide controls based on mode
        self.fill_holes_check.setVisible(mode == "brush")
        self.show_lattice_check.setVisible(mode == "liquify")
        self.lattice_size_label_text.setVisible(mode == "liquify")
        self.lattice_size_slider.setVisible(mode == "liquify")
        self.lattice_size_label.setVisible(mode == "liquify")
        self.bake_liquify_btn.setVisible(mode == "liquify")
        
        # Show/hide shape-specific controls
        is_shape_mode = mode == "shape" or mode == "liquify"
        # Warp button removed - was displaying as "./ar"
        self.liquify_btn.setVisible(is_shape_mode)
        self.vertex_count_label_text.setVisible(mode == "shape")  # Only for shape mode
        self.vertex_count_slider.setVisible(mode == "shape")  # Only for shape mode
        self.vertex_count_label.setVisible(mode == "shape")  # Only for shape mode
        self.vertex_sep.setVisible(mode == "shape")  # Only for shape mode
        
        # Reset warp mode when switching modes
        if mode != "shape":
            self.warp_mode = False
            # Warp button removed
            self.mask_widget.warp_mode = False
            
        # If switching to eraser, check if we need shape mode
        if mode == "eraser":
            # Check if there are any shapes in the current frame
            if hasattr(self.mask_widget, 'shape_keyframes'):
                frame = self.current_frame_index
                shapes = self.mask_widget.get_shapes_for_frame(frame)
                if shapes:
                    # We have shapes, eraser will work on shapes
                    pass
        
    def set_tool(self, tool):
        """Set the current active tool"""
        self.mask_widget.set_current_tool(tool)
        
    def select_brush_mode(self):
        """Select brush tool with current brush mode (pixel or shape)"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[InpaintingMaskEditor.select_brush_mode] Called")
        sys.stdout.flush()
        # Manually handle exclusive selection
        self.brush_btn.setChecked(True)
        self.eraser_btn.setChecked(False)
        self.zoom_tool_btn.setChecked(False)
        self.liquify_btn.setChecked(False)
        self.set_tool("brush")
        
        # Use the last brush mode (pixel or shape)
        last_mode = getattr(self.mask_widget, '_last_brush_mode', 'brush')
        logger.info(f"[InpaintingMaskEditor.select_brush_mode] Using last_mode={last_mode}")
        self.set_drawing_mode(last_mode)
        
        # Update the button icon to match the current mode
        self.update_brush_button_icon()
    
    def toggle_brush_mode(self):
        """Toggle between pixel and shape brush modes (called by Shift+B)"""
        current_mode = getattr(self.mask_widget, '_last_brush_mode', 'brush')
        
        # Toggle the mode
        if current_mode == "brush":
            new_mode = "shape"
        else:
            new_mode = "brush"
        
        # Update the drawing mode
        self.set_drawing_mode(new_mode)
        
        # If currently on brush tool, update the display
        if self.brush_btn.isChecked():
            self.update_brush_button_icon()
            
        
    def select_eraser_mode(self):
        """Select brush tool and eraser mode"""
        # Manually handle exclusive selection
        self.brush_btn.setChecked(False)
        self.eraser_btn.setChecked(True)
        self.zoom_tool_btn.setChecked(False)
        self.liquify_btn.setChecked(False)
        self.set_tool("brush")
        self.set_drawing_mode("eraser")
        
        # Restore focus to mask widget
        self.mask_widget.setFocus()
        
    def select_shape_mode(self):
        """Select brush tool and shape mode"""
        # Manually handle exclusive selection
        self.brush_btn.setChecked(False)
        self.eraser_btn.setChecked(False)
        self.shape_btn.setChecked(True)
        self.zoom_tool_btn.setChecked(False)
        self.liquify_btn.setChecked(False)
        self.set_tool("brush")
        self.set_drawing_mode("shape")
        
    def select_zoom_mode(self):
        """Select zoom tool"""
        # Manually handle exclusive selection
        self.brush_btn.setChecked(False)
        self.eraser_btn.setChecked(False)
        self.zoom_tool_btn.setChecked(True)
        self.liquify_btn.setChecked(False)
        self.set_tool("zoom")
        
        # Restore focus to mask widget
        self.mask_widget.setFocus()
        
    def select_liquify_mode(self):
        """Select liquify tool"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[InpaintingMaskEditor.select_liquify_mode] Called")
        sys.stdout.flush()
        # Manually handle exclusive selection
        self.brush_btn.setChecked(False)
        self.eraser_btn.setChecked(False)
        self.zoom_tool_btn.setChecked(False)
        self.liquify_btn.setChecked(True)
        self.set_tool("liquify")
        # Set drawing mode to liquify
        self.set_drawing_mode("liquify")
        
        # Restore focus to mask widget
        self.mask_widget.setFocus()
            
    def set_warp_mode(self, enabled):
        """Enable/disable warp mode for shape editing"""
        self.warp_mode = enabled
        self.mask_widget.warp_mode = enabled
        
    def on_brush_size_changed(self, value):
        self.brush_size = value
        self.brush_size_label.setText(f"{value}")
        self.mask_widget.set_brush_size(value)
        
        # Save to settings based on current mode
        if self.drawing_mode == "brush":
            self.brush_sizes['pixel'] = value
            self.settings.setValue('mask_editor_brush_size_pixel', value)
        elif self.drawing_mode == "shape":
            self.brush_sizes['shape'] = value
            self.settings.setValue('mask_editor_brush_size_shape', value)
        elif self.drawing_mode == "eraser":
            self.brush_sizes['eraser'] = value
            self.settings.setValue('mask_editor_brush_size_eraser', value)
        elif self.drawing_mode == "liquify":
            self.brush_sizes['liquify'] = value
            self.settings.setValue('mask_editor_brush_size_liquify', value)
        
    def on_vertex_count_changed(self, value):
        """Handle vertex count slider change"""
        self.vertex_count_label.setText(f"{value}")
        
        # Update mask widget vertex count
        self.mask_widget.target_vertex_count = value
        
        # Save to settings
        self.settings.setValue('mask_editor_vertex_count', value)
        
        # If slider is being dragged, update display with resampled preview
        if hasattr(self, '_vertex_slider_dragging') and self._vertex_slider_dragging:
            # Update the mask widget to show preview
            self.mask_widget._preview_vertex_count = value
            self.mask_widget.update()
        else:
            # Not dragging - update actual shapes
            if hasattr(self.mask_widget, 'shape_keyframes'):
                for frame in self.mask_widget.shape_keyframes:
                    for shape in self.mask_widget.shape_keyframes[frame]:
                        if 'vertices' in shape and len(shape['vertices']) > 0:
                            # Resample vertices to new count
                            shape['vertices'] = self.mask_widget.resample_vertices(shape['vertices'], value)
                            shape['vertex_count'] = value
                
                # Update display
                self.mask_widget.update_mask_from_shapes()
        
        # Restore focus to mask widget
        self.mask_widget.setFocus()
    
    def on_vertex_slider_pressed(self):
        """Handle when vertex slider is pressed - store original shapes"""
        self._vertex_slider_dragging = True
        
        # Store original shapes before any modification
        if hasattr(self.mask_widget, 'shape_keyframes'):
            import copy
            self._original_shape_keyframes = copy.deepcopy(self.mask_widget.shape_keyframes)
            
        # Set preview mode in mask widget
        self.mask_widget._vertex_preview_mode = True
        self.mask_widget._preview_vertex_count = self.vertex_count_slider.value()
    
    def on_vertex_slider_released(self):
        """Handle when vertex slider is released - apply resampling"""
        self._vertex_slider_dragging = False
        
        # Clear preview mode
        self.mask_widget._vertex_preview_mode = False
        if hasattr(self.mask_widget, '_preview_vertex_count'):
            del self.mask_widget._preview_vertex_count
        
        # Apply resampling to shapes from original
        value = self.vertex_count_slider.value()
        if hasattr(self, '_original_shape_keyframes'):
            import copy
            # Restore original shapes and resample to final value
            self.mask_widget.shape_keyframes = copy.deepcopy(self._original_shape_keyframes)
            
            for frame in self.mask_widget.shape_keyframes:
                for shape in self.mask_widget.shape_keyframes[frame]:
                    if 'vertices' in shape and len(shape['vertices']) > 0:
                        # Resample vertices to new count
                        shape['vertices'] = self.mask_widget.resample_vertices(shape['vertices'], value)
                        shape['vertex_count'] = value
            
            # Invalidate shape cache since vertex counts changed
            self.mask_widget.invalidate_shape_cache()
            # Update display
            self.mask_widget.update_mask_from_shapes()
        
        # Clear stored original shapes
        if hasattr(self, '_original_shape_keyframes'):
            del self._original_shape_keyframes
        
        # Restore focus to mask widget
        self.mask_widget.setFocus()
        
    def on_fill_holes_changed(self, state):
        """Handle fill holes checkbox state change"""
        # Save to settings
        self.settings.setValue('mask_editor_fill_holes', state == Qt.Checked)
        
    def on_show_lattice_changed(self, state):
        """Handle show lattice checkbox state change"""
        # Save to settings
        self.settings.setValue('mask_editor_show_lattice', state == Qt.Checked)
        # Update the mask widget
        if hasattr(self.mask_widget, 'show_lattice'):
            self.mask_widget.show_lattice = (state == Qt.Checked)
            self.mask_widget.update()
        
    def on_lattice_size_changed(self, value):
        """Handle lattice size slider change"""
        self.lattice_size_label.setText(f"{value}")
        # Save to settings
        self.settings.setValue('mask_editor_lattice_size', value)
        # Update the mask widget
        if hasattr(self.mask_widget, 'liquify_grid_size'):
            old_grid_size = self.mask_widget.liquify_grid_size
            self.mask_widget.liquify_grid_size = value
            
            # If we have an existing deformation field and the grid size changed,
            # we need to resample the deformation field
            if (hasattr(self.mask_widget, 'liquify_deformation_field') and 
                self.mask_widget.liquify_deformation_field is not None and 
                old_grid_size != value):
                # The deformation field size doesn't need to change, we just update
                # the grid visualization size. The deformation field is per-pixel.
                pass
            
            self.mask_widget.update()
    
    def on_spline_interpolation_changed(self, state):
        """Handle smooth interpolation checkbox state change"""
        # Save to settings
        self.settings.setValue('mask_editor_spline_interpolation', state == Qt.Checked)
        # Invalidate shape cache to trigger re-interpolation with new method
        if hasattr(self.mask_widget, 'invalidate_shape_cache'):
            self.mask_widget.invalidate_shape_cache()
        # Force immediate recalculation of masks from shapes
        self.mask_widget.update_mask_from_shapes()
        # Update display immediately
        self.mask_widget.update()
        # Restore focus to mask widget for keyboard shortcuts
        self.mask_widget.setFocus()
        
    def on_spline_shapes_changed(self, state):
        """Handle smooth shapes checkbox state change"""
        # Save to settings
        self.settings.setValue('mask_editor_spline_shapes', state == Qt.Checked)
        # Force immediate recalculation of masks from shapes with new rendering method
        self.mask_widget.update_mask_from_shapes()
        # Trigger immediate redraw of current shapes
        self.mask_widget.update()
        # Restore focus to mask widget for keyboard shortcuts
        self.mask_widget.setFocus()
        
    def on_timeline_frame_changed(self, index):
        """Handle frame change from timeline widget clicks - mark as manual navigation"""
        self._manual_navigation = True
        self.on_frame_changed(index)
    
    def on_frame_changed(self, index):
        # Stop playback if user manually navigates to different frame (but not during automatic playback)
        if self.is_playing and hasattr(self, '_manual_navigation') and self._manual_navigation:
            self.pause_playback()
        
        # Before changing frames, check for temporary interpolated shapes
        # Skip this check if we're doing keyframe navigation (Alt+Arrow)
        if (hasattr(self.mask_widget, '_temp_interpolated_shapes') and 
            self.current_frame_index in self.mask_widget._temp_interpolated_shapes and
            not getattr(self, '_keyframe_navigation', False)):  # Skip if keyframe navigation
            response = QMessageBox.question(self, "Unsaved Shape", 
                "You have an unsaved shape on this frame.\n\n" +
                "Do you want to apply it before switching frames?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            
            if response == QMessageBox.Yes:
                # Apply the temporary shape
                self.apply_to_current_frame()
            elif response == QMessageBox.No:
                # Clear the temporary shape
                del self.mask_widget._temp_interpolated_shapes[self.current_frame_index]
            else:
                # Cancel - don't change frames
                return
        
        # Before changing frames, preserve any liquify changes
        if (self.drawing_mode == "liquify" and hasattr(self.mask_widget, 'liquify_deformation_field') and
            self.mask_widget.liquify_deformation_field is not None and
            self.mask_widget.liquify_original_shapes is not None):
            
            # Check if there are any actual deformations to preserve
            max_deform = np.max(np.abs(self.mask_widget.liquify_deformation_field))
            if max_deform > 0.01:  # If there's significant deformation
                print(f"Preserving liquify changes on frame {self.current_frame_index} before switching to frame {index}")
                # Bake the current liquify deformations into the keyframe
                self.mask_widget.bake_liquify_deformation()
        
        self.current_frame_index = index
        # Frame change - shapes will be recalculated
        # Update timeline widget current frame
        self.timeline_widget.setCurrentFrame(index)
        
        # If in liquify mode, temporarily switch to shape mode for proper interpolation
        original_drawing_mode = None
        if self.drawing_mode == "liquify":
            original_drawing_mode = "liquify"
            # Temporarily switch to shape mode internally
            self.mask_widget.set_drawing_mode("shape")
        
        # Reset liquify deformation for the new frame
        if original_drawing_mode == "liquify" and hasattr(self.mask_widget, 'liquify_deformation_field'):
            # Reset the deformation field for the new frame
            if self.mask_widget.mask is not None:
                h, w = self.mask_widget.mask.shape[:2]
                self.mask_widget.liquify_deformation_field = np.zeros((h, w, 2), dtype=np.float32)
            else:
                self.mask_widget.liquify_deformation_field = None
            
            # Reset the original shapes for the new frame
            self.mask_widget.liquify_original_shapes = None
            
            # Reset the liquifying state
            self.mask_widget.is_liquifying = False
            
            # Clear temporary deformed shapes to restore proper interpolation
            self.mask_widget._temp_deformed_shapes = None
        
        self.update_display()
        
        # Restore liquify mode after display update
        if original_drawing_mode == "liquify":
            self.mask_widget.set_drawing_mode("liquify")
        
        # Restore focus to mask widget for keyboard shortcuts
        self.mask_widget.setFocus()
        
        # Update navigation button states
        self.update_navigation_buttons()
        
    def update_display(self):
        """Update mask display for current frame"""
        # Get current frame for mask widget background
        frame = self.video_frames[self.current_frame_index]
        
        # Update mask display
        mask = self.mask_frames[self.current_frame_index]
        self.mask_widget.set_mask(mask, frame)
        
        # Update mask from shapes if in shape mode or liquify mode (for interpolation)
        if self.drawing_mode in ["shape", "liquify"]:
            self.mask_widget.update_mask_from_shapes()
        
        # Update Apply to Current button state
        self.update_apply_button_state()
    
    def update_apply_button_state(self):
        """Update the Apply to Current button based on temporary shapes"""
        has_temp_shapes = (hasattr(self.mask_widget, '_temp_interpolated_shapes') and 
                          self.current_frame_index in self.mask_widget._temp_interpolated_shapes)
        
        if has_temp_shapes:
            # Make button more prominent when there are temporary shapes
            self.apply_current_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff8800;
                    color: white;
                    font-weight: bold;
                    border: 2px solid #ff6600;
                }
                QPushButton:hover {
                    background-color: #ff9922;
                }
            """)
            self.apply_current_btn.setText("Apply to Current (Unsaved Shape)")
        else:
            # Reset to normal style
            self.apply_current_btn.setStyleSheet("")
            self.apply_current_btn.setText("Apply to Current")
        
    def clear_current_frame(self):
        """Clear the mask for current frame"""
        # Save undo state before clearing
        self.mask_widget.save_undo_state("Clear frame")
        
        if self.drawing_mode in ["shape", "liquify"]:
            # In shape mode, remove the keyframe entirely so interpolation takes over
            if hasattr(self.mask_widget, 'shape_keyframes'):
                if self.current_frame_index in self.mask_widget.shape_keyframes:
                    # Check if this is the reference keyframe
                    was_reference_frame = (hasattr(self.mask_widget, '_sequence_reference_signature') and 
                                         len(self.mask_widget.shape_keyframes) > 0 and
                                         self.current_frame_index == min(self.mask_widget.shape_keyframes.keys()))
                    
                    # Remove the keyframe completely
                    del self.mask_widget.shape_keyframes[self.current_frame_index]
                    
                    # Reset reference if we deleted the reference keyframe
                    if was_reference_frame:
                        self.mask_widget.reset_sequence_reference_signature()
                    
                    # Invalidate the shape interpolation cache since we removed a keyframe
                    self.mask_widget.invalidate_shape_cache()
                # Shapes will be recalculated for this frame
                # Force the mask widget to update its current frame
                self.mask_widget.current_frame = self.current_frame_index
                # Update the mask from shapes - this will now interpolate from surrounding keyframes
                self.mask_widget.update_mask_from_shapes()
                # Update timeline to remove the keyframe indicator
                self.update_mask_frame_tracking()
        else:
            # In pixel mode, clear the pixel data
            self.mask_frames[self.current_frame_index] = np.zeros_like(self.mask_frames[self.current_frame_index])
        self.update_display()
        # Restore focus to mask widget for keyboard shortcuts
        self.mask_widget.setFocus()
        
    def fill_current_frame(self):
        """Fill the mask for current frame"""
        # Save undo state before filling
        self.mask_widget.save_undo_state("Fill frame")
        
        if self.drawing_mode == "shape":
            # In shape mode, create a shape that covers the entire frame
            if hasattr(self.mask_widget, 'shape_keyframes'):
                h, w = self.video_frames[self.current_frame_index].shape[:2]
                # Create a rectangle shape that covers the entire frame
                full_frame_shape = {
                    'vertices': [
                        (0, 0),
                        (w, 0),
                        (w, h),
                        (0, h)
                    ],
                    'vertex_count': 4,
                    'filled': True
                }
                # Set this as the only shape for this frame
                self.mask_widget.shape_keyframes[self.current_frame_index] = [full_frame_shape]
                # Invalidate the shape interpolation cache since we added/modified a keyframe
                self.mask_widget.invalidate_shape_cache()
                # Update the mask from shapes
                self.mask_widget.update_mask_from_shapes()
                # Update timeline
                self.update_mask_frame_tracking()
        else:
            # In pixel mode, fill the pixel data
            self.mask_frames[self.current_frame_index] = np.ones_like(self.mask_frames[self.current_frame_index]) * 255
        self.update_display()
        # Restore focus to mask widget for keyboard shortcuts
        self.mask_widget.setFocus()
        
    def clear_all_frames(self):
        """Clear masks for all frames"""
        # Save undo state before clearing
        self.mask_widget.save_undo_state("Clear all frames")
        
        # Clear any active liquify deformation first
        if self.drawing_mode == "liquify":
            # Reset liquify state
            if hasattr(self.mask_widget, 'liquify_deformation_field'):
                self.mask_widget.liquify_deformation_field = None
            if hasattr(self.mask_widget, 'liquify_original_shapes'):
                self.mask_widget.liquify_original_shapes = None
            if hasattr(self.mask_widget, '_temp_deformed_shapes'):
                self.mask_widget._temp_deformed_shapes = None
            self.mask_widget.is_liquifying = False
        
        # Clear shape keyframes regardless of mode (shape/liquify both use shapes)
        if self.drawing_mode in ["shape", "liquify"]:
            # In shape/liquify mode, clear all shape keyframes
            if hasattr(self.mask_widget, 'shape_keyframes'):
                self.mask_widget.shape_keyframes.clear()
                # Reset reference signature when clearing all keyframes
                self.mask_widget.reset_sequence_reference_signature()
            # Clear shape cache if it exists
            if hasattr(self.mask_widget, '_shape_cache'):
                self.mask_widget._shape_cache.clear()
            # Clear any temporary shapes
            if hasattr(self.mask_widget, '_temp_shapes'):
                self.mask_widget._temp_shapes = None
            if hasattr(self.mask_widget, '_temp_deformed_shapes'):
                self.mask_widget._temp_deformed_shapes = None
            # Update mask from shapes for current frame
            self.mask_widget.update_mask_from_shapes()
            # Update timeline to remove all keyframe indicators
            self.update_mask_frame_tracking()
        else:
            # In pixel/brush/eraser mode, clear all pixel data
            for i in range(len(self.mask_frames)):
                self.mask_frames[i] = np.zeros_like(self.mask_frames[i])
        
        # Clear the current mask display
        if hasattr(self.mask_widget, 'mask') and self.mask_widget.mask is not None:
            self.mask_widget.mask.fill(0)
        
        self.update_display()
        # Restore focus to mask widget for keyboard shortcuts
        self.mask_widget.setFocus()
        QMessageBox.information(self, "Success", "All masks cleared")
        
    def bake_liquify(self):
        """Bake the liquify deformation permanently"""
        if self.drawing_mode == "liquify" and hasattr(self.mask_widget, 'bake_liquify_deformation'):
            self.mask_widget.bake_liquify_deformation()
            # Update the display
            self.update_display()
            # Restore focus to mask widget for keyboard shortcuts
            self.mask_widget.setFocus()
            QMessageBox.information(self, "Liquify Baked", "The liquify deformation has been applied permanently and the lattice has been reset.")
        
    def apply_to_current_frame(self):
        """Apply the current mask to the current frame in the mask buffer"""
        
        # Get the current mask from the drawing widget
        if self.mask_widget.mask is None:
            QMessageBox.warning(self, "Warning", "No mask to apply")
            return
        
        # Ensure mask widget mode matches editor mode
        if self.mask_widget.drawing_mode != self.drawing_mode:
            self.mask_widget.drawing_mode = self.drawing_mode
            
        # Store the current mask
        self.mask_frames[self.current_frame_index] = self.mask_widget.mask.copy()
        
        # If in shape mode, we need to handle keyframes
        if self.drawing_mode == "shape":
            if not hasattr(self.mask_widget, 'shape_keyframes'):
                return
            
            # COMPLETELY NEW APPROACH - Just copy whatever shapes are visible
            # Don't use any interpolation or get_shapes_for_frame bullshit
            
            # Check if we already have a keyframe at this exact frame
            if self.current_frame_index in self.mask_widget.shape_keyframes:
                # Silently return without showing a message
                return
            
            # NEVER use interpolated cache for Apply to Current - it's unreliable
            # Instead, directly check if this is a keyframe or needs interpolation
            current_shapes = None
            
            # First check if this frame already has shapes (is a keyframe)
            if self.current_frame_index in self.mask_widget.shape_keyframes:
                # This shouldn't happen as we check above, but just in case
                current_shapes = self.mask_widget.shape_keyframes[self.current_frame_index]
                print(f"Found existing keyframe with {len(current_shapes)} shapes")
            else:
                # Not a keyframe - need to get interpolated shapes WITHOUT caching
                keyframes = sorted(self.mask_widget.shape_keyframes.keys())
                if keyframes:
                    # Find surrounding keyframes
                    prev_frame = None
                    next_frame = None
                    for kf in keyframes:
                        if kf < self.current_frame_index:
                            prev_frame = kf
                        elif kf > self.current_frame_index and next_frame is None:
                            next_frame = kf
                            break
                    
                    # Get interpolated shapes for JUST this frame
                    if prev_frame is not None and next_frame is not None and prev_frame != next_frame:
                        t = (self.current_frame_index - prev_frame) / (next_frame - prev_frame)
                        current_shapes = self.mask_widget.interpolate_shapes(
                            self.mask_widget.shape_keyframes[prev_frame],
                            self.mask_widget.shape_keyframes[next_frame],
                            t
                        )
                    elif prev_frame is not None:
                        # Deep copy shapes to avoid modifying originals
                        current_shapes = []
                        for shape in self.mask_widget.shape_keyframes[prev_frame]:
                            current_shapes.append(shape.copy())
                    elif next_frame is not None:
                        # Deep copy shapes to avoid modifying originals
                        current_shapes = []
                        for shape in self.mask_widget.shape_keyframes[next_frame]:
                            current_shapes.append(shape.copy())
            
            # Check if we have temporary shapes drawn on this interpolated frame
            has_temp_shapes = (hasattr(self.mask_widget, '_temp_interpolated_shapes') and 
                             self.current_frame_index in self.mask_widget._temp_interpolated_shapes)
            
            if has_temp_shapes:
                # Get the temporary mask
                temp_mask = self.mask_widget._temp_interpolated_shapes[self.current_frame_index]
                
                # Merge temporary mask with any existing interpolated shapes
                combined_mask = temp_mask.copy()
                preserved_start_point = None  # Track the start point to preserve
                if current_shapes:
                    # Draw existing interpolated shapes to mask
                    existing_mask = np.zeros_like(temp_mask)
                    for shape in current_shapes:
                        vertices = np.array(shape['vertices'], dtype=np.int32)
                        cv2.fillPoly(existing_mask, [vertices], 255)
                        
                        # Preserve the start point from the first shape
                        if preserved_start_point is None and len(vertices) > 0:
                            preserved_start_point = vertices[0].tolist()
                    combined_mask = cv2.bitwise_or(combined_mask, existing_mask)
                
                # Convert combined mask to shapes
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Always use the target vertex count from the slider
                # This ensures all shapes respect the user's vertex count setting
                vertex_count_to_use = self.vertex_count_slider.value()
                print(f"Using vertex count {vertex_count_to_use} from slider (ignoring existing shape counts)")
                
                # Create shapes from contours
                new_shapes = []
                for contour in contours:
                    # Skip very small contours
                    if cv2.contourArea(contour) < 100:
                        continue
                    
                    # Resample contour to fixed vertex count
                    vertices = self.mask_widget.resample_contour_fixed_vertices(contour, 
                                                                              vertex_count_to_use,
                                                                              preserve_start_point=preserved_start_point)
                    
                    shape = {
                        'vertices': vertices,
                        'closed': True,
                        'visible': True,
                        'vertex_count': vertex_count_to_use
                    }
                    new_shapes.append(shape)
                
                current_shapes = new_shapes
                
                # Clear the temporary shape for this frame
                del self.mask_widget._temp_interpolated_shapes[self.current_frame_index]
            
            # If no interpolated shapes and no temp shapes, try to create from the mask
            elif not current_shapes and np.any(self.mask_widget.mask > 0):
                # Convert mask to shapes using contour detection
                contours, _ = cv2.findContours(self.mask_widget.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                current_shapes = []
                target_vertices = self.vertex_count_slider.value()
                
                for contour in contours:
                    # Skip very small contours
                    if cv2.contourArea(contour) < 100:
                        continue
                    
                    # Simplify contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Convert to vertex list
                    vertices = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
                    
                    # Resample to target vertex count
                    if len(vertices) > 2:
                        vertices = self.mask_widget.resample_vertices(vertices, target_vertices)
                        
                        shape = {
                            'vertices': vertices,
                            'visible': True,
                            'is_shape': True,
                            'vertex_count': target_vertices
                        }
                        current_shapes.append(shape)
                
                print(f"Created {len(current_shapes)} shapes from mask contours")
            
            if current_shapes:
                # CRITICAL: Only add to THIS frame
                
                # Make a copy of current shapes preserving their vertex parameterization
                shapes_copy = []
                
                for shape in current_shapes:
                    shape_copy = shape.copy()
                    # Ensure vertices match the slider setting
                    if 'vertices' in shape_copy:
                        # Resample to match slider setting if needed
                        target_vertex_count = self.vertex_count_slider.value()
                        current_vertices = shape_copy['vertices']
                        if len(current_vertices) != target_vertex_count:
                            shape_copy['vertices'] = self.mask_widget.resample_vertices(current_vertices, target_vertex_count)
                        else:
                            shape_copy['vertices'] = [list(v) for v in current_vertices]
                        shape_copy['vertex_count'] = target_vertex_count
                    shapes_copy.append(shape_copy)
                
                # Add ONLY to current frame
                self.mask_widget.shape_keyframes[self.current_frame_index] = shapes_copy
                
                
                # Clear interpolation cache only for affected frames
                # Find neighboring keyframes to determine affected range
                keyframes = sorted(self.mask_widget.shape_keyframes.keys())
                prev_kf = None
                next_kf = None
                for kf in keyframes:
                    if kf < self.current_frame_index:
                        prev_kf = kf
                    elif kf > self.current_frame_index and next_kf is None:
                        next_kf = kf
                
                # Clear cache for frames between previous and next keyframes
                start_frame = prev_kf if prev_kf is not None else 0
                end_frame = next_kf if next_kf is not None else len(self.video_frames)
                
                # Shapes will be recalculated for the affected frames
                
                print(f"Will recalculate shapes for frames {start_frame} to {end_frame}")
                
                # Update tracking
                self.update_mask_frame_tracking()
                
                # Silently succeed - no message box
                print(f"Created 1 keyframe at frame {self.current_frame_index + 1}")
            else:
                # Silently handle - no warning
                print("No shapes to create keyframe from")
        else:
            # Pixel mode
            print("Pixel mode - just saving mask")
            # Silently succeed - no message box
            print(f"Mask applied to frame {self.current_frame_index + 1}")
            self.update_mask_frame_tracking()
        
        
        # Update the Apply button state after applying
        self.update_apply_button_state()
    
    def apply_to_all_frames(self):
        """Apply current mask to all frames"""
        current_mask = self.mask_frames[self.current_frame_index].copy()
        for i in range(len(self.mask_frames)):
            self.mask_frames[i] = current_mask.copy()
        QMessageBox.information(self, "Success", "Mask applied to all frames")
        
    def copy_mask(self):
        """Copy current mask"""
        self.copied_mask = self.mask_frames[self.current_frame_index].copy()
        self.paste_btn.setEnabled(True)
        
    def paste_mask(self):
        """Paste copied mask"""
        if hasattr(self, 'copied_mask'):
            self.mask_frames[self.current_frame_index] = self.copied_mask.copy()
            self.update_display()
            
    def save_masks(self):
        """Return the mask frames to parent"""
        self.accept()
        
    def get_masks(self):
        """Get the edited mask frames"""
        # If in shape mode, ensure masks are updated from shapes
        if self.drawing_mode == "shape":
            # Store current frame index
            original_index = self.current_frame_index
            
            
            for i in range(len(self.mask_frames)):
                # Change to each frame
                self.current_frame_index = i
                # Set the mask for that frame
                self.mask_widget.set_mask(self.mask_frames[i], self.video_frames[i])
                # Update mask from shapes
                self.mask_widget.update_mask_from_shapes()
                # Store the updated mask
                self.mask_frames[i] = self.mask_widget.mask.copy()
                
                # Debug: check if mask has content
                if i % 100 == 0 or i < 10:  # Log first 10 and every 100th frame
                    has_content = np.any(self.mask_frames[i] > 0)
                    print(f"Frame {i}: has_content={has_content}, max_value={np.max(self.mask_frames[i])}")
            
            # Restore original frame
            self.current_frame_index = original_index
            self.mask_widget.set_mask(self.mask_frames[original_index], self.video_frames[original_index])
            self.mask_widget.update_mask_from_shapes()
            
        return self.mask_frames
    
    def get_editor_state(self):
        """Get the editor state including shapes and mode"""
        state = {
            'drawing_mode': self.drawing_mode,
            'shape_keyframes': self.mask_widget.shape_keyframes.copy() if hasattr(self.mask_widget, 'shape_keyframes') else {},
            'brush_size': self.brush_size,
            'vertex_count': self.vertex_count_slider.value()
        }
        # print(f"DEBUG: Saving editor state with drawing_mode: {self.drawing_mode}")
        return state
    
    def save_project(self, filepath=None):
        """Save the entire project state to a file"""
        # If no filepath provided, show file dialog
        if filepath is None or filepath == '' or not isinstance(filepath, str):
            # Get last used directory from settings
            settings = QSettings('WAN', 'MaskEditor')
            last_dir = settings.value('last_project_dir', '')
            default_name = os.path.join(last_dir, "mask_project.wmp") if last_dir else "mask_project.wmp"
            
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save Mask Project",
                default_name,  # WAN Mask Project
                "WAN Mask Project (*.wmp);;All Files (*.*)"
            )
            
            # Save the directory for next time
            if filepath:
                settings.setValue('last_project_dir', os.path.dirname(filepath))
            # User cancelled the dialog
            if not filepath:
                return False
        
        # Final validation
        if not isinstance(filepath, str) or filepath == '':
            return False
        
        try:
            # Prepare project data
            project_data = {
                'version': '1.0',
                'drawing_mode': self.drawing_mode,
                'current_frame': self.current_frame_index,
                'brush_size': self.brush_size,
                'vertex_count': self.vertex_count_slider.value(),
                'shape_keyframes': {},
                'mask_frames': {},
                'video_info': {
                    'frame_count': len(self.video_frames),
                    'width': self.video_frames[0].shape[1],
                    'height': self.video_frames[0].shape[0]
                }
            }
            
            # Convert shape keyframes to serializable format
            if hasattr(self.mask_widget, 'shape_keyframes'):
                for frame, shapes in self.mask_widget.shape_keyframes.items():
                    serializable_shapes = []
                    for shape in shapes:
                        serializable_shape = {
                            'vertices': [[float(v[0]), float(v[1])] for v in shape['vertices']],
                            'closed': shape.get('closed', True),
                            'visible': shape.get('visible', True),
                            'vertex_count': shape.get('vertex_count', len(shape['vertices'])),
                            'is_shape': shape.get('is_shape', True)
                        }
                        serializable_shapes.append(serializable_shape)
                    project_data['shape_keyframes'][str(frame)] = serializable_shapes
            
            # Save non-empty mask frames as PNG base64
            for i in range(len(self.mask_frames)):
                mask = self.mask_frames[i]
                if mask is not None and isinstance(mask, np.ndarray) and np.any(mask > 0):  # Only save non-empty masks
                    # Encode mask as PNG
                    success, buffer = cv2.imencode('.png', mask)
                    if success:
                        encoded = base64.b64encode(buffer).decode('utf-8')
                        project_data['mask_frames'][str(i)] = encoded
            
            # Write to file using Path for better file handling
            from pathlib import Path
            save_path = Path(str(filepath))  # Ensure filepath is a string
            json_text = json.dumps(project_data, indent=2)
            save_path.write_text(json_text, encoding='utf-8')
            
            # Clear unsaved changes flag
            self.clear_unsaved_changes()
            
            # Only show message box if not auto-saving
            if isinstance(filepath, str) and not filepath.endswith("autosave_") and "autosave_" not in filepath:
                QMessageBox.information(self, "Success", f"Project saved to {filepath}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
            return False
    
    def load_project(self, filepath=None):
        """Load a project state from a file with a fake progress bar."""
        import tempfile, datetime
        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import QTimer

        # File dialog if needed
        if filepath is None or filepath is False or not isinstance(filepath, str):
            settings = QSettings('WAN', 'MaskEditor')
            last_dir = settings.value('last_project_dir', '')
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Load Mask Project", last_dir, "WAN Mask Project (*.wmp);;All Files (*.*)"
            )
            if filepath:
                settings.setValue('last_project_dir', os.path.dirname(filepath))
            if not filepath:
                return False

        # Progress dialog
        progress = QProgressDialog("Loading project...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Loading Project")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        # Fake progress timer
        progress_value = [0]
        def update_fake_progress():
            if progress_value[0] < 90 and not progress.wasCanceled():
                progress_value[0] += 1
                progress.setValue(progress_value[0])
                QApplication.processEvents()
        fake_timer = QTimer()
        fake_timer.timeout.connect(update_fake_progress)
        fake_timer.start(40)

        try:
            # Load JSON
            with open(filepath, 'r') as f:
                project_data = json.load(f)
            progress.setValue(max(progress_value[0], 10))
            QApplication.processEvents()

            # Version check
            if project_data.get('version', '0.0') != '1.0':
                QMessageBox.warning(self, "Warning", "Project file version mismatch. Some data may not load correctly.")

            # Dimension check
            video_info = project_data.get('video_info', {})
            if (video_info.get('frame_count') != len(self.video_frames) or
                video_info.get('width') != self.video_frames[0].shape[1] or
                video_info.get('height') != self.video_frames[0].shape[0]):
                reply = QMessageBox.question(
                    self, "Dimension Mismatch",
                    "The project dimensions don't match the current video. Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    fake_timer.stop()
                    progress.close()
                    return False

            # Clear data
            self.mask_frames = [np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8) for frame in self.video_frames]
            self.mask_widget.shape_keyframes.clear()
            progress.setValue(max(progress_value[0], 20))
            QApplication.processEvents()

            # Load mask frames
            mask_frames_data = project_data.get('mask_frames', {})
            total_masks = len(mask_frames_data)
            if total_masks > 0:
                for i, (frame_str, encoded_mask) in enumerate(mask_frames_data.items()):
                    if progress.wasCanceled():
                        fake_timer.stop()
                        progress.close()
                        return False
                    frame_idx = int(frame_str)
                    if 0 <= frame_idx < len(self.mask_frames):
                        decoded = base64.b64decode(encoded_mask)
                        nparr = np.frombuffer(decoded, np.uint8)
                        mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            self.mask_frames[frame_idx] = mask
                    if i % 5 == 0:
                        mask_progress = 20 + int((i + 1) / total_masks * 40)
                        progress_value[0] = max(progress_value[0], mask_progress)
                        progress.setValue(progress_value[0])
                        QApplication.processEvents()
            else:
                progress_value[0] = max(progress_value[0], 60)
                progress.setValue(60)
                QApplication.processEvents()

            # Load shape keyframes
            shape_keyframes_data = project_data.get('shape_keyframes', {})
            total_keyframes = len(shape_keyframes_data)
            if total_keyframes > 0:
                for i, (frame_str, shapes) in enumerate(shape_keyframes_data.items()):
                    if progress.wasCanceled():
                        fake_timer.stop()
                        progress.close()
                        return False
                    frame_idx = int(frame_str)
                    deserialized_shapes = []
                    for shape in shapes:
                        deserialized_shape = {
                            'vertices': shape['vertices'],
                            'closed': shape.get('closed', True),
                            'visible': shape.get('visible', True),
                            'vertex_count': shape.get('vertex_count', len(shape['vertices'])),
                            'is_shape': shape.get('is_shape', True)
                        }
                        deserialized_shapes.append(deserialized_shape)
                    self.mask_widget.shape_keyframes[frame_idx] = deserialized_shapes
                    if i % 3 == 0:
                        shape_progress = 60 + int((i + 1) / total_keyframes * 20)
                        progress_value[0] = max(progress_value[0], shape_progress)
                        progress.setValue(progress_value[0])
                        QApplication.processEvents()
            else:
                progress_value[0] = max(progress_value[0], 80)
                progress.setValue(80)
                QApplication.processEvents()

            # Restore UI state
            self.vertex_count_slider.setValue(project_data.get('vertex_count', 32))
            self.set_drawing_mode(project_data.get('drawing_mode', 'brush'))
            self.brush_size = project_data.get('brush_size', 20)
            self.brush_size_slider.setValue(self.brush_size)
            progress_value[0] = max(progress_value[0], 85)
            progress.setValue(progress_value[0])
            QApplication.processEvents()

            # Go to saved frame
            saved_frame = project_data.get('current_frame', 0)
            if 0 <= saved_frame < len(self.video_frames):
                self._manual_navigation = False  # Project loading is automatic
                self.on_frame_changed(saved_frame)
            progress_value[0] = max(progress_value[0], 90)
            progress.setValue(progress_value[0])
            QApplication.processEvents()

            # Update timeline
            original_mode = self.drawing_mode
            if self.mask_widget.shape_keyframes and self.drawing_mode not in ["shape", "liquify"]:
                self.drawing_mode = "shape"
            self.update_mask_frame_tracking()
            if self.drawing_mode != original_mode:
                self.drawing_mode = original_mode
            self.update_display()
            self.clear_unsaved_changes()

            # Snap to 100%
            fake_timer.stop()
            progress.setValue(100)
            QApplication.processEvents()
            progress.close()

            QMessageBox.information(self, "Success", f"Project loaded from {filepath}")
            return True

        except Exception as e:
            fake_timer.stop()
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def clean_up_session_data(self):
        """Clean up session data files when user discards changes"""
        try:
            if hasattr(self, 'output_dir') and self.output_dir:
                import os
                session_file = os.path.join(self.output_dir, "session_data.json")
                if os.path.exists(session_file):
                    os.remove(session_file)
                    print("Session data cleaned up after discard")
                
                # Also clean up any project data file that might exist
                project_file = os.path.join(self.output_dir, "project_data.json")
                if os.path.exists(project_file):
                    os.remove(project_file)
                    print("Project data cleaned up after discard")
                    
        except Exception as e:
            print(f"Error cleaning up session data: {e}")
    
    def auto_save_project(self):
        """Auto-save the project to a temporary file"""
        if not self.has_unsaved_changes or self.is_discarding:
            return
            
        try:
            # Create auto-save filename
            import tempfile
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_save_dir = Path(tempfile.gettempdir()) / "wan_mask_editor_autosave"
            auto_save_dir.mkdir(exist_ok=True)
            
            # Keep only the last 5 auto-saves
            existing_saves = sorted(auto_save_dir.glob("autosave_*.wmp"))
            if len(existing_saves) > 4:
                for old_save in existing_saves[:-4]:
                    old_save.unlink()
            
            auto_save_path = auto_save_dir / f"autosave_{timestamp}.wmp"
            
            # Save without showing dialog
            self.save_project(str(auto_save_path))
            self.auto_save_path = str(auto_save_path)
            # Avoid print in auto-save to prevent console buffer issues
            # print(f"Auto-saved to: {auto_save_path}")
            
        except Exception as e:
            # Silently fail auto-save to avoid console buffer issues
            pass
    
    def mark_unsaved_changes(self):
        """Mark that there are unsaved changes and update window title"""
        self.has_unsaved_changes = True
        current_title = self.windowTitle()
        if not current_title.endswith(" *"):
            self.setWindowTitle(current_title + " *")
    
    def clear_unsaved_changes(self):
        """Clear unsaved changes flag and update window title"""
        self.has_unsaved_changes = False
        current_title = self.windowTitle()
        if current_title.endswith(" *"):
            self.setWindowTitle(current_title[:-2])
    
    def set_editor_state(self, state):
        """Restore editor state including shapes and mode"""
        # This method is now deprecated - state restoration happens directly in open_inpainting_mask_editor
        pass
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Left:
            if event.modifiers() & Qt.AltModifier:
                # Alt+Left - Jump to previous keyframe/mask based on current mode
                self.go_to_prev_mask()  # This already handles shape vs regular mode
            elif not event.modifiers():
                # Plain Left arrow - Previous frame (backward compatibility)
                new_index = max(0, self.current_frame_index - 1)
                self._manual_navigation = True
                self.on_frame_changed(new_index)
        elif event.key() == Qt.Key_Right:
            if event.modifiers() & Qt.AltModifier:
                # Alt+Right - Jump to next keyframe/mask based on current mode
                self.go_to_next_mask()  # This already handles shape vs regular mode
            elif not event.modifiers():
                # Plain Right arrow - Next frame (backward compatibility)
                new_index = min(len(self.video_frames) - 1, self.current_frame_index + 1)
                self._manual_navigation = True
                self.on_frame_changed(new_index)
        elif event.key() == Qt.Key_E:
            # Pass to mask widget for temporary tool handling
            self.mask_widget.keyPressEvent(event)
        elif event.key() == Qt.Key_B:
            if event.modifiers() & Qt.ShiftModifier:
                # Shift+B - Pass to mask widget for cycling brush modes
                self.mask_widget.keyPressEvent(event)
            else:
                # B - Activate brush tool
                self.brush_btn.click()
        elif event.key() == Qt.Key_Z and not (event.modifiers() & Qt.ControlModifier):
            # Pass Z key to mask widget for temporary zoom handling
            self.mask_widget.keyPressEvent(event)
        elif event.key() == Qt.Key_W:
            # Activate liquify tool - automatically switch to shape mode if needed
            if self.drawing_mode != "shape" and self.drawing_mode != "liquify":
                # First switch to shape mode via brush button
                self.mask_widget._last_brush_mode = "shape"
                self.select_brush_mode()
            # Then activate liquify
            self.liquify_btn.click()
        elif event.key() == Qt.Key_0:
            # Pass to mask widget to reset zoom
            self.mask_widget.keyPressEvent(event)
        elif event.key() == Qt.Key_K:
            # Pass to mask widget for timeline scrub tool
            self.mask_widget.keyPressEvent(event)
        elif event.key() == Qt.Key_A:
            # Apply to current frame
            self.apply_to_current_frame()
        elif event.key() == Qt.Key_Left and event.modifiers() & Qt.ControlModifier:
            # Ctrl+Left - Previous mask
            self.go_to_prev_mask()
        elif event.key() == Qt.Key_Right and event.modifiers() & Qt.ControlModifier:
            # Ctrl+Right - Next mask
            self.go_to_next_mask()
        elif event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            # Ctrl+S - Save project
            self.save_project()
        elif event.key() == Qt.Key_O and event.modifiers() & Qt.ControlModifier:
            # Ctrl+O - Load project
            self.load_project()
        elif event.key() == Qt.Key_Space and not event.isAutoRepeat():
            # Spacebar - Toggle playback
            self.toggle_playback()
            event.accept()  # Prevent passing to mask widget
            return
        else:
            # Pass the event to mask widget for handling
            self.mask_widget.keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Handle key release events - pass to mask widget for temporary tool handling"""
        self.mask_widget.keyReleaseEvent(event)
    
    def closeEvent(self, event):
        """Handle window close event - prompt to save if there are unsaved changes"""
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save the project before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Save:
                if self.save_project():
                    # Stop timers only when actually closing
                    if hasattr(self, 'auto_save_timer'):
                        self.auto_save_timer.stop()
                    if hasattr(self, 'playback_timer'):
                        self.playback_timer.stop()
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.Discard:
                # Stop auto-save immediately when user chooses to discard
                self.is_discarding = True
                if hasattr(self, 'auto_save_timer'):
                    self.auto_save_timer.stop()
                
                # Stop session timer and clean up session data
                if hasattr(self, 'session_timer'):
                    self.session_timer.stop()
                
                # Delete session data file to prevent data persistence
                self.clean_up_session_data()
                
                # Stop playback timer too
                if hasattr(self, 'playback_timer'):
                    self.playback_timer.stop()
                
                event.accept()
            else:  # Cancel
                event.ignore()
                # Don't stop timers - window is staying open
        else:
            # No unsaved changes - stop timers and close
            if hasattr(self, 'auto_save_timer'):
                self.auto_save_timer.stop()
            if hasattr(self, 'playback_timer'):
                self.playback_timer.stop()
            event.accept()
    
    def cancel_without_prompts(self):
        """Close immediately without save prompts - for Cancel button"""
        # Stop all timers
        if hasattr(self, 'auto_save_timer'):
            self.auto_save_timer.stop()
        if hasattr(self, 'session_timer'):
            self.session_timer.stop()
        if hasattr(self, 'playback_timer'):
            self.playback_timer.stop()
        
        # Close immediately without prompts
        self.accept()
    
    def go_to_prev_mask(self):
        """Navigate to previous frame with a mask or shape keyframe"""
        current = self.current_frame_index
        
        # In shape/liquify mode, navigate between actual shape keyframes
        if self.drawing_mode in ["shape", "liquify"] and hasattr(self.mask_widget, 'shape_keyframes'):
            # Get sorted list of shape keyframes
            keyframes = sorted(self.mask_widget.shape_keyframes.keys(), reverse=True)
            for frame in keyframes:
                if frame < current and self.mask_widget.shape_keyframes[frame]:  # Has shapes
                    # First step one frame back to trigger proper liquify baking (if in liquify mode)
                    if self.drawing_mode == "liquify" and current > 0:
                        print(f"[Alt+Arrow] Stepping back one frame to trigger liquify bake before jumping to keyframe")
                        self.on_frame_changed(current - 1)
                    
                    self._manual_navigation = True
                    self._keyframe_navigation = True  # Flag to skip temp shape dialog
                    self.on_frame_changed(frame)
                    self._keyframe_navigation = False  # Reset flag
                    return
        else:
            # Regular mask navigation
            for frame in sorted(self.timeline_widget.mask_frames, reverse=True):
                if frame < current:
                    self._manual_navigation = True
                    self._keyframe_navigation = True  # Flag to skip temp shape dialog
                    self.on_frame_changed(frame)
                    self._keyframe_navigation = False  # Reset flag
                    return
    
    def go_to_next_mask(self):
        """Navigate to next frame with a mask or shape keyframe"""
        current = self.current_frame_index
        
        # In shape/liquify mode, navigate between actual shape keyframes
        if self.drawing_mode in ["shape", "liquify"] and hasattr(self.mask_widget, 'shape_keyframes'):
            # Get sorted list of shape keyframes
            keyframes = sorted(self.mask_widget.shape_keyframes.keys())
            for frame in keyframes:
                if frame > current and self.mask_widget.shape_keyframes[frame]:  # Has shapes
                    # First step one frame forward to trigger proper liquify baking (if in liquify mode)
                    if self.drawing_mode == "liquify" and current < len(self.video_frames) - 1:
                        print(f"[Alt+Arrow] Stepping forward one frame to trigger liquify bake before jumping to keyframe")
                        self.on_frame_changed(current + 1)
                    
                    self._manual_navigation = True
                    self._keyframe_navigation = True  # Flag to skip temp shape dialog
                    self.on_frame_changed(frame)
                    self._keyframe_navigation = False  # Reset flag
                    return
        else:
            # Regular mask navigation
            for frame in sorted(self.timeline_widget.mask_frames):
                if frame > current:
                    self._manual_navigation = True
                    self._keyframe_navigation = True  # Flag to skip temp shape dialog
                    self.on_frame_changed(frame)
                    self._keyframe_navigation = False  # Reset flag
                    return
    
    def go_to_prev_keyframe(self):
        """Go to previous shape keyframe (alias for go_to_prev_mask in shape mode)"""
        if self.drawing_mode in ["shape", "liquify"]:
            self.go_to_prev_mask()
    
    def go_to_next_keyframe(self):
        """Go to next shape keyframe (alias for go_to_next_mask in shape mode)"""
        if self.drawing_mode in ["shape", "liquify"]:
            self.go_to_next_mask()
    
    # Video playback methods
    def toggle_playback(self):
        """Toggle video playback on/off"""
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start video playback"""
        if not self.video_frames:
            print("[Playback] No video frames to play")
            return
            
        self.is_playing = True
        self.play_pause_btn.setText("⏸ Pause")
        self.playback_start_frame = self.current_frame_index
        
        # Calculate timer interval from FPS
        interval = int(1000 / self.playback_fps)  # Convert to milliseconds
        self.playback_timer.start(interval)
        
    
    def pause_playback(self):
        """Pause video playback"""
        self.is_playing = False
        self.play_pause_btn.setText("▶ Play")
        self.playback_timer.stop()
        
    
    def stop_playback(self):
        """Stop playback and return to start frame"""
        # Stop playback immediately
        was_playing = self.is_playing
        self.is_playing = False
        self.play_pause_btn.setText("▶ Play")
        self.playback_timer.stop()
        
        # Return to start frame if not already there
        if self.playback_start_frame != self.current_frame_index:
            # Set navigation flag
            self._manual_navigation = False  # This is automatic navigation
            
            # Try frame change
            try:
                self.on_frame_changed(self.playback_start_frame)
                
                # Update timeline widget
                self.timeline_widget.current_frame = self.playback_start_frame
                self.timeline_widget.update()
                self.timeline_widget.repaint()  # Force repaint
                
                # Force display update
                if hasattr(self, 'update_display'):
                    self.update_display()
                
            except Exception as e:
                print(f"Error during stop playback frame change: {e}")
    
    def advance_frame(self):
        """Advance to next frame during playback"""
        if not self.is_playing:
            return
            
        next_frame = self.current_frame_index + 1
        
        # Check if we've reached the end
        if next_frame >= len(self.video_frames):
            if self.loop_check.isChecked():
                # Loop back to start
                next_frame = 0
            else:
                # Stop playback
                self.pause_playback()
                return
        
        # Navigate to next frame (mark as automatic, not manual)
        self._manual_navigation = False
        self.on_frame_changed(next_frame)
        self.timeline_widget.current_frame = next_frame
        self.timeline_widget.update()
    
    def on_fps_changed(self, fps):
        """Handle FPS change during playback"""
        self.playback_fps = fps
        
        # Save FPS value to settings
        self.settings.setValue('playback_fps', fps)
        
        # If currently playing, restart the timer with new interval
        if self.is_playing:
            self.playback_timer.stop()
            interval = int(1000 / self.playback_fps)
            self.playback_timer.start(interval)
            
    
    def update_navigation_buttons(self):
        """Update the state of navigation buttons"""
        # Navigation buttons have been removed
        pass
    
    def update_mask_frame_tracking(self):
        """Scan all mask frames and update timeline tracking"""
        self.timeline_widget.mask_frames.clear()
        
        # If we're in shape mode OR liquify mode, ONLY track shape keyframes
        if (self.drawing_mode in ["shape", "liquify"]) and hasattr(self.mask_widget, 'shape_keyframes'):
            # Only add actual keyframes, not interpolated frames
            for frame in self.mask_widget.shape_keyframes:
                if self.mask_widget.shape_keyframes[frame]:  # Has shapes
                    self.timeline_widget.mask_frames.add(frame)
                    # print(f"DEBUG: Added shape keyframe at frame {frame}")
        else:
            # In pixel mode, check for pixel-painted masks
            for i in range(len(self.mask_frames)):
                # Only add if it has content
                if np.any(self.mask_frames[i] > 0):
                    self.timeline_widget.mask_frames.add(i)
        
        # print(f"DEBUG: Timeline now shows {len(self.timeline_widget.mask_frames)} frames")
        self.timeline_widget.update()



class MaskDrawingWidget(QWidget):
    """Widget for drawing masks"""
    
    def __init__(self):
        super().__init__()
        self.mask = None
        self.video_frame = None
        self.drawing_mode = None  # Will be set by parent editor
        self.brush_size = 20
        self.is_drawing = False
        self.last_point = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus
        
        # Brush size adjustment
        self.adjusting_brush_size = False
        self.brush_adjust_start_pos = None
        self.brush_adjust_start_size = None
        
        # Parent reference for updating UI
        self.parent_editor = None
        
        # Tool system
        self.current_tool = "brush"  # "brush", "zoom", "timeline_scrub"
        
        # View properties
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        
        # Timeline scrub properties
        self.is_timeline_scrubbing = False
        self.timeline_scrub_start_x = 0
        self.timeline_scrub_start_frame = 0
        self.is_panning = False
        self.pan_start = None
        self.pan_start_offset = None
        
        # Zoom adjustment
        self.is_zooming = False
        self.zoom_start_pos = None
        self.zoom_start_level = None
        
        # Display properties
        self.scaled_pixmap = None
        self.display_rect = QRect()
        
        # Space key state
        self.space_pressed = False
        self.shift_pressed = False  # For liquify smooth/relax mode
        
        # Temporary tool state
        self.temp_tool_keys = {
            Qt.Key_B: {'pressed': False, 'stroke_made': False, 'previous_tool': None, 'previous_mode': None},
            Qt.Key_E: {'pressed': False, 'stroke_made': False, 'previous_tool': None, 'previous_mode': None},
            Qt.Key_W: {'pressed': False, 'stroke_made': False, 'previous_tool': None, 'previous_mode': None},
            Qt.Key_Z: {'pressed': False, 'stroke_made': False, 'previous_tool': None, 'previous_mode': None},
            Qt.Key_K: {'pressed': False, 'stroke_made': False, 'previous_tool': None, 'previous_mode': None},
            Qt.Key_Control: {'pressed': False, 'stroke_made': False, 'previous_tool': None, 'previous_mode': None, 'previous_shape_eraser': False}
        }
        
        # Stroke tracking for fill holes feature
        self.current_stroke_points = []
        
        # Shape mode properties
        self.shape_keyframes = {}  # frame -> list of shapes
        self.target_vertex_count = 32  # Default vertex count for shapes
        self.selected_shape_index = None
        self.selected_vertex_index = None
        self.warp_mode = False
        self.relax_mode = False
        self.shape_eraser_mode = False  # Toggle for shape eraser
        self.drag_start_pos = None
        self.original_vertex_pos = None
        self.temp_shape_mask = None  # Temporary mask for shape painting
        self.shape_merge_mode = False  # Whether to merge with interpolated shapes (Shift+Click)
        self.is_drawing_new_shape = False  # Whether currently drawing new shape (for dimming interpolated shapes)
        self.show_shape_debug = False  # Toggle for debug visualization
        
        # Shape interpolation cache
        
        # Liquify properties
        self.liquify_grid_size = 20  # Grid cell size in pixels
        self.liquify_deformation_field = None  # Stores the deformation
        self.liquify_original_shapes = None  # Store original shapes before deformation
        self.is_liquifying = False
        self.show_lattice = True  # Whether to show the lattice grid
        self._last_liquify_keyframe_frame = None  # Prevent multiple keyframes
        self._updating_timeline = False  # Prevent keyframe creation during timeline updates
        self._temp_deformed_shapes = None  # Temporary storage for deformed shapes during liquify
        
        # Remember last brush mode for B key
        self._last_brush_mode = "shape"  # Default to shape brush
        
        # Undo/Redo system
        self.undo_stack = []  # Stack of previous states
        self.redo_stack = []  # Stack of undone states
        self.max_undo_steps = 20  # Maximum number of undo steps to keep
        
        # Performance optimization: defer expensive operations during initialization
        self._initializing = False
        
        # Shape interpolation cache to avoid recalculation
        self._shape_cache = {}  # (frame, cache_key) -> interpolated shapes
        self._cache_key = 0  # Incremented when keyframes change
        
    def set_mask(self, mask, video_frame):
        """Set the mask to edit"""
        self.mask = mask  # Don't copy - edit the original
        self.video_frame = video_frame
        self.update()
    
    def complete_initialization(self):
        """Complete initialization after UI is responsive"""
        self._initializing = False
        self.update()  # Trigger a repaint with full rendering
    
    def invalidate_shape_cache(self):
        """Invalidate the shape interpolation cache"""
        self._cache_key += 1
        self._shape_cache.clear()  # Clear old cache entries
        # Limit cache size to prevent memory issues
        if self._cache_key > 1000000:
            self._cache_key = 0
        
    def set_drawing_mode(self, mode):
        import logging
        logger = logging.getLogger(__name__)
        
        # Before switching away from liquify mode, always bake pending changes
        if self.drawing_mode == "liquify" and mode != "liquify":
            logger.info(f"[MaskDrawingWidget.set_drawing_mode] Switching from liquify to {mode}, baking changes")
            self.bake_liquify_deformation()
        
        self.drawing_mode = mode
        
        # Remember the last brush mode for B key shortcuts
        if mode in ["brush", "shape"]:
            self._last_brush_mode = mode
        
        # When entering liquify mode, we no longer reset liquify_original_shapes
        # It will be populated lazily when needed in bake_liquify_deformation
        # This prevents losing shapes if the user switches modes before moving the brush
        if mode == "liquify":
            logger.info(f"[MaskDrawingWidget.set_drawing_mode] Entering liquify mode - shapes will be captured on first use")
        
    def set_brush_size(self, size):
        self.brush_size = size
        self.update()
        
    def set_current_tool(self, tool):
        """Set the current active tool"""
        import logging
        logger = logging.getLogger(__name__)
        
        # ------------------------------------------------------------------
        # 2.  Before switching away from liquify, always bake pending changes
        # ------------------------------------------------------------------
        # Check both tool and drawing mode since liquify can be either
        if (self.current_tool == "liquify" and tool != "liquify") or \
           (self.drawing_mode == "liquify" and tool != "liquify"):
            logger.info(f"[MaskDrawingWidget.set_current_tool] Switching from liquify to {tool}, baking changes")
            self.bake_liquify_deformation()
        
        self.current_tool = tool
        if tool == "zoom":
            self.setCursor(Qt.CrossCursor)
        elif tool == "timeline_scrub":
            self.setCursor(Qt.SizeHorCursor)  # Horizontal resize cursor for timeline scrub
        else:
            self.setCursor(Qt.ArrowCursor)
            
    def widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to image coordinates accounting for zoom and pan"""
        if not self.display_rect.isValid() or self.mask is None or self.mask.size == 0:
            return None, None
            
        # Get position relative to displayed image
        x = widget_pos.x() - self.display_rect.x()
        y = widget_pos.y() - self.display_rect.y()
        
        # Check if click is within image bounds
        if x < 0 or y < 0 or x >= self.display_rect.width() or y >= self.display_rect.height():
            return None, None
            
        # Prevent division by zero
        if self.display_rect.width() == 0 or self.display_rect.height() == 0:
            return None, None
            
        # Convert to image coordinates with safe division
        img_x = int(x * self.mask.shape[1] / self.display_rect.width())
        img_y = int(y * self.mask.shape[0] / self.display_rect.height())
        
        # Clamp to image bounds
        img_x = max(0, min(self.mask.shape[1] - 1, img_x))
        img_y = max(0, min(self.mask.shape[0] - 1, img_y))
        
        return img_x, img_y
        
    def mark_stroke_made(self):
        """Mark that a stroke was made for any currently pressed temp tool keys"""
        for key, state in self.temp_tool_keys.items():
            if state['pressed']:
                state['stroke_made'] = True
    
    def clear_temp_tool_states(self):
        """Clear all temporary tool states - used when switching tools permanently"""
        for key, state in self.temp_tool_keys.items():
            state['pressed'] = False
            state['stroke_made'] = False
            state['previous_tool'] = None
            state['previous_mode'] = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check for Alt modifier first for timeline scrubbing
            if event.modifiers() & Qt.AltModifier:
                # Start timeline scrubbing with Alt+click
                self.is_timeline_scrubbing = True
                self.timeline_scrub_start_x = event.pos().x()
                if self.parent_editor:
                    self.timeline_scrub_start_frame = self.parent_editor.current_frame_index
                else:
                    self.timeline_scrub_start_frame = 0
                self.setCursor(Qt.SizeHorCursor)
                return  # Don't process other tools
            elif self.space_pressed:
                # Start panning
                self.is_panning = True
                self.pan_start = event.pos()
                self.pan_start_offset = self.pan_offset
                self.setCursor(Qt.ClosedHandCursor)
            elif self.current_tool == "zoom":
                # Start zooming
                self.is_zooming = True
                self.zoom_start_pos = event.pos()
                self.zoom_start_level = self.zoom_level
                self.zoom_start_pan = QPoint(self.pan_offset)  # Store initial pan offset
                # Mark stroke made for temporary zoom
                self.mark_stroke_made()
            elif self.current_tool == "liquify" and self.mask is not None:
                # Start liquify deformation
                if self.drawing_mode == "liquify":
                    self.is_liquifying = True
                    self.last_point = event.pos()
                    # Mark stroke made for temporary liquify
                    self.mark_stroke_made()
                    # Set up liquify immediately on mouse press to ensure it's ready
                    if self.liquify_original_shapes is None:
                        print("[MousePress] liquify_original_shapes is None, setting up liquify")
                        self.setup_liquify_for_current_frame()
            elif self.current_tool == "timeline_scrub":
                # Start timeline scrubbing
                self.is_timeline_scrubbing = True
                self.timeline_scrub_start_x = event.pos().x()
                if self.parent_editor:
                    self.timeline_scrub_start_frame = self.parent_editor.current_frame_index
                else:
                    self.timeline_scrub_start_frame = 0
                # Mark stroke made for temporary timeline scrub
                self.mark_stroke_made()
            elif self.current_tool == "brush" and self.mask is not None:
                # Mark stroke made for temporary brush/eraser
                self.mark_stroke_made()
                if self.drawing_mode == "shape":
                    if self.warp_mode:
                        # Check if clicking on a vertex
                        self.check_vertex_selection(event.pos())
                    elif self.shape_eraser_mode:
                        # Start shape eraser stroke - similar to shape drawing
                        self.is_drawing = True
                        self.current_stroke_points = []
                        self.last_point = event.pos()
                        
                        # Create temporary mask for shape erasing
                        self.temp_shape_mask = np.zeros_like(self.mask)
                        
                        # Draw first point
                        img_coords = self.widget_to_image_coords(event.pos())
                        if img_coords[0] is not None:
                            self.current_stroke_points.append(img_coords)
                            # Draw on temp mask
                            cv2.circle(self.temp_shape_mask, tuple(map(int, img_coords)), self.brush_size, 255, -1)
                    else:
                        # Start drawing a shape like regular brush
                        self.is_drawing = True
                        self.last_point = event.pos()
                        self.current_stroke_points = []
                        
                        # Store whether Shift is pressed for merge mode
                        self.shape_merge_mode = bool(event.modifiers() & Qt.ShiftModifier)
                        
                        # Track if drawing new shape (not merging) for visual dimming
                        self.is_drawing_new_shape = not self.shape_merge_mode
                        
                        # Create temporary mask for shape painting
                        self.temp_shape_mask = np.zeros_like(self.mask)
                        
                        # Draw first point
                        img_coords = self.widget_to_image_coords(event.pos())
                        if img_coords[0] is not None:
                            self.current_stroke_points.append(img_coords)
                            # Draw on temp mask
                            cv2.circle(self.temp_shape_mask, tuple(map(int, img_coords)), self.brush_size, 255, -1)
                else:
                    # Regular brush/eraser mode
                    if self.drawing_mode == "eraser":
                        # Start eraser stroke for boolean operation
                        self.is_drawing = True
                        self.current_stroke_points = []
                        self.last_point = event.pos()  # Track last point for preview
                        img_coords = self.widget_to_image_coords(event.pos())
                        if img_coords[0] is not None:
                            self.current_stroke_points.append(img_coords)
                    else:
                        # Regular brush mode
                        self.is_drawing = True
                        self.last_point = event.pos()
                        self.current_stroke_points = []  # Start new stroke
                        
                        # Add first point to stroke
                        img_coords = self.widget_to_image_coords(event.pos())
                        if img_coords[0] is not None:
                            self.current_stroke_points.append(img_coords)
                        
                        self.draw_at_point(event.pos())
        elif event.button() == Qt.RightButton and event.modifiers() & Qt.AltModifier and (self.current_tool == "brush" or self.current_tool == "liquify"):
            # Start brush size adjustment
            self.adjusting_brush_size = True
            self.brush_adjust_start_pos = event.pos()
            self.brush_adjust_start_size = self.brush_size
            self.setCursor(Qt.SizeHorCursor)
            
    def mouseMoveEvent(self, event):
        if self.is_panning:
            # Update pan offset
            delta = event.pos() - self.pan_start
            self.pan_offset = self.pan_start_offset + delta
            self.update()
        elif self.is_zooming:
            # Calculate zoom based on horizontal movement
            delta = event.pos().x() - self.zoom_start_pos.x()
            zoom_factor = 1.0 + delta / 200.0  # Adjust sensitivity
            new_zoom = self.zoom_start_level * zoom_factor
            new_zoom = max(0.1, min(10.0, new_zoom))  # Clamp between 0.1x and 10x
            
            if new_zoom != self.zoom_level:
                # Store the relative position of the zoom point at start
                if not hasattr(self, 'zoom_rel_x') or not hasattr(self, 'zoom_rel_y'):
                    # Calculate once at the beginning
                    img_width = self.video_frame.shape[1]
                    img_height = self.video_frame.shape[0]
                    
                    # Calculate display rect at start zoom
                    start_width = int(img_width * self.zoom_start_level)
                    start_height = int(img_height * self.zoom_start_level)
                    start_x = (self.width() - start_width) // 2 + self.zoom_start_pan.x()
                    start_y = (self.height() - start_height) // 2 + self.zoom_start_pan.y()
                    
                    # Calculate relative position (0-1) within the image
                    self.zoom_rel_x = (self.zoom_start_pos.x() - start_x) / float(start_width)
                    self.zoom_rel_y = (self.zoom_start_pos.y() - start_y) / float(start_height)
                
                # Apply new zoom
                self.zoom_level = new_zoom
                
                # Calculate new image dimensions
                img_width = self.video_frame.shape[1]
                img_height = self.video_frame.shape[0]
                new_width = int(img_width * self.zoom_level)
                new_height = int(img_height * self.zoom_level)
                
                # Calculate where the image would be centered
                new_x = (self.width() - new_width) // 2
                new_y = (self.height() - new_height) // 2
                
                # Find where the zoom point would be in the new image
                new_mouse_x = new_x + self.zoom_rel_x * new_width
                new_mouse_y = new_y + self.zoom_rel_y * new_height
                
                # Adjust pan to keep the zoom point in the same position
                self.pan_offset.setX(int(self.zoom_start_pos.x() - new_mouse_x))
                self.pan_offset.setY(int(self.zoom_start_pos.y() - new_mouse_y))
            
            self.update()
        elif self.adjusting_brush_size:
            # Calculate new brush size based on horizontal movement
            delta = event.pos().x() - self.brush_adjust_start_pos.x()
            new_size = self.brush_adjust_start_size + delta // 2
            new_size = max(1, min(500, new_size))  # Clamp between 1 and 500
            self.brush_size = new_size
            
            # Update parent UI if available
            if self.parent_editor:
                self.parent_editor.brush_size_slider.setValue(new_size)
                self.parent_editor.brush_size = new_size
        elif self.is_drawing and self.mask is not None and self.current_tool == "brush":
            if self.drawing_mode == "shape":
                if self.shape_eraser_mode:
                    # Paint on temporary mask like shape drawing for fill functionality
                    img_coords = self.widget_to_image_coords(event.pos())
                    if img_coords[0] is not None and hasattr(self, 'temp_shape_mask'):
                        last_coords = self.widget_to_image_coords(self.last_point)
                        if last_coords[0] is not None:
                            # Draw line on temp mask
                            cv2.line(self.temp_shape_mask, 
                                    tuple(map(int, last_coords)), 
                                    tuple(map(int, img_coords)), 
                                    255, self.brush_size * 2)
                        self.current_stroke_points.append(img_coords)
                    self.last_point = event.pos()
                    self.update()  # Show preview
                else:
                    # Paint on temporary mask like regular brush
                    img_coords = self.widget_to_image_coords(event.pos())
                    if img_coords[0] is not None and hasattr(self, 'temp_shape_mask'):
                        last_coords = self.widget_to_image_coords(self.last_point)
                        if last_coords[0] is not None:
                            # Draw line on temp mask
                            cv2.line(self.temp_shape_mask, 
                                    tuple(map(int, last_coords)), 
                                    tuple(map(int, img_coords)), 
                                    255, self.brush_size * 2)
                        self.current_stroke_points.append(img_coords)
                    self.last_point = event.pos()
                    self.update()  # Show preview
            elif self.drawing_mode == "eraser":
                # Handle regular eraser
                img_coords = self.widget_to_image_coords(event.pos())
                if img_coords[0] is not None:
                    self.current_stroke_points.append(img_coords)
                self.last_point = event.pos()
                self.update()  # Show preview
            else:
                # Regular brush drawing
                self.draw_line(self.last_point, event.pos())
                
                # Add point to stroke for fill holes feature
                img_coords = self.widget_to_image_coords(event.pos())
                if img_coords[0] is not None:
                    self.current_stroke_points.append(img_coords)
                
                self.last_point = event.pos()
        elif self.selected_vertex_index is not None and self.warp_mode:
            # Warp the selected vertex
            self.warp_vertex(event.pos())
        elif self.is_liquifying and self.current_tool == "liquify":
            # First time moving? Set up liquify
            if self.liquify_original_shapes is None:
                self.setup_liquify_for_current_frame()
            
            # Apply liquify deformation
            self.apply_liquify_deformation(event.pos())
            self.last_point = event.pos()
        elif self.is_timeline_scrubbing:
            # Calculate frame offset based on horizontal movement
            if self.parent_editor:
                # Calculate pixels per frame based on viewport width and total frames
                total_frames = len(self.parent_editor.video_frames)
                if total_frames > 0:
                    pixels_per_frame = 5.0  # Adjust sensitivity (5 pixels = 1 frame, very sensitive)
                    delta_x = event.pos().x() - self.timeline_scrub_start_x
                    frame_offset = int(delta_x / pixels_per_frame)
                    
                    # Calculate new frame index
                    new_frame = self.timeline_scrub_start_frame + frame_offset
                    new_frame = max(0, min(total_frames - 1, new_frame))
                    
                    # Update the frame if it changed
                    if new_frame != self.parent_editor.current_frame_index:
                        self.parent_editor.on_frame_changed(new_frame)
        self.update()  # Update to show brush cursor
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_panning:
                self.is_panning = False
                if self.current_tool == "zoom":
                    self.setCursor(Qt.CrossCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            elif self.is_zooming:
                self.is_zooming = False
                # Clear cached zoom relative position
                if hasattr(self, 'zoom_rel_x'):
                    delattr(self, 'zoom_rel_x')
                if hasattr(self, 'zoom_rel_y'):
                    delattr(self, 'zoom_rel_y')
            elif self.is_drawing:
                self.is_drawing = False
                
                # Save undo state before applying changes
                action_performed = False
                
                if self.drawing_mode == "shape":
                    if self.shape_eraser_mode and hasattr(self, 'temp_shape_mask'):
                        # Save state before erasing
                        self.save_undo_state("Shape erase")
                        # Apply shape eraser with filled shape
                        self.apply_shape_eraser_with_fill()
                        self.current_stroke_points = []  # Clear stroke points
                        self.temp_shape_mask = None  # Clear temp mask
                        action_performed = True
                    elif hasattr(self, 'temp_shape_mask'):
                        # Save state before creating shape
                        self.save_undo_state("Create shape")
                        # Create shape from painted mask
                        self.create_shape_from_painted_mask()
                        self.current_stroke_points = []  # Clear stroke points
                        # Clear shape mode flags
                        self.shape_merge_mode = False
                        self.is_drawing_new_shape = False
                        action_performed = True
                elif self.drawing_mode == "eraser" and len(self.current_stroke_points) > 2:
                    # Save state before erasing
                    self.save_undo_state("Erase")
                    # Apply boolean eraser operation to shapes
                    self.apply_eraser_to_shapes()
                    self.current_stroke_points = []  # Clear stroke points
                    action_performed = True
                elif (self.parent_editor and self.parent_editor.fill_holes_check.isChecked() and 
                    self.drawing_mode == "brush" and len(self.current_stroke_points) > 10):
                    # Save state before filling
                    self.save_undo_state("Fill shape")
                    self.fill_closed_shape()
                    self.current_stroke_points = []  # Clear stroke points
                    action_performed = True
                else:
                    # For regular brush strokes
                    if self.drawing_mode == "brush" and len(self.current_stroke_points) > 0:
                        self.save_undo_state("Brush stroke")
                        action_performed = True
                    # Clear stroke points for regular brush mode
                    self.current_stroke_points = []
            elif self.selected_vertex_index is not None:
                # Save state after warping vertex
                self.save_undo_state("Warp vertex")
                self.selected_vertex_index = None
                self.selected_shape_index = None
            elif self.is_liquifying:
                # Save state after liquify stroke
                self.save_undo_state("Liquify")
                self.is_liquifying = False
                # Don't reset the deformation - keep it accumulated
                
                # Keyframe has already been created at mouse press if needed
                # No need to create another one here
            elif self.is_timeline_scrubbing:
                self.is_timeline_scrubbing = False
                # Restore cursor based on current tool
                if self.current_tool == "zoom":
                    self.setCursor(Qt.CrossCursor)
                elif self.current_tool == "timeline_scrub":
                    self.setCursor(Qt.SizeHorCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.RightButton and self.adjusting_brush_size:
            self.adjusting_brush_size = False
            self.setCursor(Qt.ArrowCursor)
        
    def draw_at_point(self, pos):
        """Draw at a single point"""
        if self.mask is None:
            return
            
        # Convert widget coordinates to mask coordinates
        x, y = self.widget_to_image_coords(pos)
        if x is None or y is None:
            return
        
        try:
            # Draw circle with bounds checking
            color = 255 if self.drawing_mode == "brush" else 0
            # Ensure brush size is reasonable
            brush_size = min(self.brush_size, max(self.mask.shape[0], self.mask.shape[1]) // 2)
            cv2.circle(self.mask, (int(x), int(y)), brush_size, color, -1)
            self.update()
            
            # Update timeline mask tracking only for pixel painting
            if self.drawing_mode in ["brush", "eraser"] and self.parent_editor and hasattr(self.parent_editor, 'timeline_widget'):
                frame = self.parent_editor.current_frame_index
                has_mask = np.any(self.mask > 0)
                
                # Only track if there's no shape keyframe at this position
                if frame not in self.shape_keyframes:
                    if has_mask:
                        self.parent_editor.timeline_widget.mask_frames.add(frame)
                    else:
                        self.parent_editor.timeline_widget.mask_frames.discard(frame)
                    
                    self.parent_editor.timeline_widget.update()
                    self.parent_editor.update_navigation_buttons()
        except Exception as e:
            print(f"Error drawing at point: {e}")
            return
        
    def draw_line(self, start_pos, end_pos):
        """Draw a line between two points"""
        if self.mask is None:
            return
            
        # Convert widget coordinates to mask coordinates
        x1, y1 = self.widget_to_image_coords(start_pos)
        x2, y2 = self.widget_to_image_coords(end_pos)
        
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return
        
        try:
            # Draw line with bounds checking
            color = 255 if self.drawing_mode == "brush" else 0
            # Ensure brush size is reasonable
            brush_size = min(self.brush_size * 2, max(self.mask.shape[0], self.mask.shape[1]) // 2)
            cv2.line(self.mask, (int(x1), int(y1)), (int(x2), int(y2)), color, brush_size)
            self.update()
        except Exception as e:
            print(f"Error drawing line: {e}")
            return
    
    def fill_closed_shape(self):
        """Fill a closed shape if the stroke forms a loop"""
        if len(self.current_stroke_points) < 10 or self.mask is None:
            return
            
        try:
            # Check if the path is closed (first and last points are close)
            first_point = self.current_stroke_points[0]
            last_point = self.current_stroke_points[-1]
            
            # Calculate distance between first and last point
            dist = np.sqrt((first_point[0] - last_point[0])**2 + (first_point[1] - last_point[1])**2)
            
            # If the path is closed (distance is less than brush size)
            if dist < self.brush_size * 3:
                # Create a mask with just the stroke path
                stroke_mask = np.zeros_like(self.mask)
                
                # Draw the stroke path
                for i in range(1, len(self.current_stroke_points)):
                    pt1 = tuple(map(int, self.current_stroke_points[i-1]))
                    pt2 = tuple(map(int, self.current_stroke_points[i]))
                    cv2.line(stroke_mask, pt1, pt2, 255, self.brush_size * 2)
                
                # Close the path
                pt1 = tuple(map(int, self.current_stroke_points[-1]))
                pt2 = tuple(map(int, self.current_stroke_points[0]))
                cv2.line(stroke_mask, pt1, pt2, 255, self.brush_size * 2)
                
                # Find the center point of the shape for flood fill
                points = np.array(self.current_stroke_points, dtype=np.int32)
                M = cv2.moments(points)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Create a copy for flood fill
                    flood_mask = stroke_mask.copy()
                    
                    # Ensure center point is within bounds
                    cy = max(0, min(flood_mask.shape[0] - 1, cy))
                    cx = max(0, min(flood_mask.shape[1] - 1, cx))
                    
                    # Check if center point is inside the shape (not on the stroke)
                    if flood_mask[cy, cx] == 0:
                        # Flood fill from center
                        cv2.floodFill(flood_mask, None, (cx, cy), 255)
                    else:
                        # Center is on the stroke, try to find a point inside
                        # Use contour detection to find the inside
                        contours, _ = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Get the largest contour
                            largest_contour = max(contours, key=cv2.contourArea)
                            # Fill the contour
                            flood_mask = np.zeros_like(self.mask)
                            cv2.drawContours(flood_mask, [largest_contour], -1, 255, -1)
                    
                    # Apply to the actual mask using direct indexing
                    if self.drawing_mode == "brush":
                        # Set pixels where flood_mask is white
                        self.mask[flood_mask > 0] = 255
                    else:  # eraser mode
                        # Clear pixels where flood_mask is white
                        self.mask[flood_mask > 0] = 0
                    
                    self.update()
        except Exception as e:
            print(f"Error filling shape: {e}")
    
    def create_shape_from_painted_mask(self):
        """Create shape from painted pixels"""
        if not hasattr(self, 'temp_shape_mask'):
            return
            
        try:
            # First, ensure the painted mask is fully connected
            # Apply morphological closing to connect nearby regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            connected_mask = cv2.morphologyEx(self.temp_shape_mask, cv2.MORPH_CLOSE, kernel)
            
            # Fill any holes in the painted mask
            filled_mask = connected_mask.copy()
            h, w = filled_mask.shape
            flood_fill_mask = np.zeros((h+2, w+2), np.uint8)
            
            # Find all white regions and fill from edges to find holes
            inverted = cv2.bitwise_not(filled_mask)
            cv2.floodFill(inverted, flood_fill_mask, (0, 0), 255)
            holes = cv2.bitwise_not(inverted)
            filled_mask = cv2.bitwise_or(filled_mask, holes)
            
            # Get current frame
            frame = self.parent_editor.current_frame_index if self.parent_editor else 0
            
            # Check if we're creating a shape on an interpolated frame
            is_interpolated_frame = frame not in self.shape_keyframes
            current_shapes = self.get_shapes_for_frame(frame) if is_interpolated_frame else []
            
            # Check if Shift was held during shape creation (merge mode)
            merge_mode = hasattr(self, 'shape_merge_mode') and self.shape_merge_mode
            
            # Determine vertex count to use
            vertex_count_to_use = self.target_vertex_count
            if is_interpolated_frame and current_shapes:
                # Use the vertex count from the first interpolated shape
                vertex_count_to_use = len(current_shapes[0]['vertices'])
            
            # Handle interpolated frames differently based on merge mode
            if is_interpolated_frame and not merge_mode:
                # Default behavior: create completely new shape (no merge)
                # This will replace the interpolated shape with a new one
                # print(f"Creating new shape on interpolated frame {frame}, replacing interpolated shapes")
                
                # Create a keyframe immediately with just the new shape
                # This effectively replaces any interpolated shapes
                pass  # Fall through to normal shape creation
                
            elif is_interpolated_frame and merge_mode:
                # Shift+Click: merge with interpolated shape
                # print(f"Merging shape with interpolated shapes on frame {frame}")
                
                # Create a combined mask with interpolated shapes and new shape
                combined_mask = np.zeros_like(self.mask)
                preserved_start_point = None  # Track the start point to preserve
                
                # Add all interpolated shapes to the combined mask
                for shape in current_shapes:
                    if shape['visible']:
                        vertices = np.array(shape['vertices'], dtype=np.int32)
                        cv2.fillPoly(combined_mask, [vertices], 255)
                        
                        # Preserve the start point from the first visible shape
                        if preserved_start_point is None and len(vertices) > 0:
                            preserved_start_point = vertices[0].tolist()
                
                # Add the new painted shape
                combined_mask = cv2.bitwise_or(combined_mask, filled_mask)
                
                # Apply morphological operations to ensure smooth boundaries
                kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_smooth)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_smooth)
                
                # Create new keyframe with the merged shape
                self.shape_keyframes[frame] = []
                self.invalidate_shape_cache()
                
                # Find contours of the merged shape
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Skip very small contours
                    if cv2.contourArea(contour) < 50:
                        continue
                    
                    # Simplify contour to get vertices with fixed count using reference signature
                    reference_sig = self.get_sequence_reference_signature()
                    vertices = self.resample_contour_fixed_vertices(contour, 
                                                                  num_vertices=vertex_count_to_use,
                                                                  preserve_start_point=preserved_start_point,
                                                                  reference_signature=reference_sig)
                    
                    if len(vertices) < 3:
                        continue
                    
                    # Note: winding order is already handled inside resample_contour_fixed_vertices
                    
                    # Add the merged shape
                    shape = {
                        'vertices': vertices,
                        'closed': True,
                        'visible': True,
                        'vertex_count': vertex_count_to_use
                    }
                    
                    self.shape_keyframes[frame].append(shape)
                    
                    # If this is the first keyframe created, establish it as adaptive reference
                    if len(self.shape_keyframes) == 1 and not hasattr(self, '_sequence_reference_signature'):
                        self._sequence_reference_signature = self.create_adaptive_reference_signature(vertices)
                
                # Clear shape mode flags
                self.shape_merge_mode = False
                self.is_drawing_new_shape = False
                
                # Update everything and return early
                # Don't normalize - it destroys our preserved start points
                # self.normalize_all_shape_keyframes()
                self.update_mask_from_shapes()
                
                if self.parent_editor and hasattr(self.parent_editor, 'timeline_widget'):
                    if self.shape_keyframes[frame]:
                        self.parent_editor.timeline_widget.mask_frames.add(frame)
                    else:
                        self.parent_editor.timeline_widget.mask_frames.discard(frame)
                    self.parent_editor.timeline_widget.update()
                    self.parent_editor.update_navigation_buttons()
                
                self.temp_shape_mask = None
                return
            
            # Initialize frame shapes if needed (only for actual keyframes)
            if frame not in self.shape_keyframes:
                self.shape_keyframes[frame] = []
                self.invalidate_shape_cache()
            
            # Create a combined mask with all existing shapes
            combined_mask = np.zeros_like(self.mask)
            shapes_to_remove = []
            preserved_start_point = None  # Track the start point to preserve
            
            # Draw all existing shapes to check for overlaps
            for i, existing_shape in enumerate(self.shape_keyframes[frame]):
                if existing_shape['visible']:
                    existing_vertices = np.array(existing_shape['vertices'], dtype=np.int32)
                    shape_mask = np.zeros_like(self.mask)
                    cv2.fillPoly(shape_mask, [existing_vertices], 255)
                    
                    # Check if this shape overlaps with the new painted area
                    overlap = cv2.bitwise_and(filled_mask, shape_mask)
                    if np.sum(overlap) > 0:
                        # Mark this shape for removal as it will be merged
                        shapes_to_remove.append(i)
                        combined_mask = cv2.bitwise_or(combined_mask, shape_mask)
                        
                        # Don't preserve start point - let the algorithm find the natural one
                        # This prevents locking to wrong positions
                        # preserved_start_point = existing_vertices[0].tolist()
            
            # Combine with the new painted mask
            combined_mask = cv2.bitwise_or(combined_mask, filled_mask)
            
            # Apply morphological operations to ensure smooth boundaries
            kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_smooth)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_smooth)
            
            # Find only the external contour
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return
            
            # Remove shapes that were merged (in reverse order to maintain indices)
            for i in reversed(shapes_to_remove):
                del self.shape_keyframes[frame][i]
            
            # Process each external contour as a separate shape
            for contour in contours:
                # Skip very small contours
                if cv2.contourArea(contour) < 50:
                    continue
                
                # Simplify contour to get vertices with fixed count using reference signature
                # Note: winding order is already handled inside resample_contour_fixed_vertices
                reference_sig = self.get_sequence_reference_signature()
                vertices = self.resample_contour_fixed_vertices(contour, 
                                                              num_vertices=vertex_count_to_use,
                                                              preserve_start_point=preserved_start_point,
                                                              reference_signature=reference_sig)
                
                if len(vertices) < 3:
                    continue
                
                # Add the new shape
                shape = {
                    'vertices': vertices,
                    'closed': True,
                    'visible': True,
                    'vertex_count': vertex_count_to_use
                }
                
                self.shape_keyframes[frame].append(shape)
                
                # If this is the first keyframe created, establish it as adaptive reference
                if len(self.shape_keyframes) == 1 and not hasattr(self, '_sequence_reference_signature'):
                    self._sequence_reference_signature = self.create_adaptive_reference_signature(vertices)
            
            # Clear interpolation cache
            self.invalidate_shape_cache()
            
            # Don't normalize - it destroys our preserved start points
            # self.normalize_all_shape_keyframes()
            
            # Update the mask
            self.update_mask_from_shapes()
            
            # Update timeline to show this shape keyframe
            if self.parent_editor and hasattr(self.parent_editor, 'timeline_widget'):
                if self.shape_keyframes[frame]:  # Has shapes
                    self.parent_editor.timeline_widget.mask_frames.add(frame)
                else:
                    self.parent_editor.timeline_widget.mask_frames.discard(frame)
                self.parent_editor.timeline_widget.update()
                self.parent_editor.update_navigation_buttons()
            
            # Clear temp mask
            self.temp_shape_mask = None
            
        except Exception as e:
            print(f"Error creating shape from painted mask: {e}")
            import traceback
            traceback.print_exc()
    
    def resample_contour_fixed_vertices(self, contour, num_vertices=32, preserve_start_point=None, reference_signature=None):
        """Resample contour to have a fixed number of vertices preserving the curve shape
        
        Args:
            contour: OpenCV contour
            num_vertices: Target number of vertices
            preserve_start_point: Deprecated parameter (no longer used)
            reference_signature: Reference signature for consistent parametrization
        """
        # Convert contour to list of points (use original contour, not approximated)
        if len(contour.shape) == 3:
            # Standard OpenCV contour format
            original_points = [p[0].tolist() for p in contour]
        else:
            # Already flattened
            original_points = contour.tolist()
        
        if len(original_points) < 3:
            return original_points
        
        # IMPORTANT: First ensure consistent winding order on the raw points
        # This must happen before finding the start point
        original_points = self.ensure_consistent_winding(original_points)
        
        # Then resample with reference signature for consistent parametrization
        return self.resample_vertices(original_points, num_vertices, reference_signature=reference_signature)
    
    def resample_contour_distance_based(self, points, num_vertices):
        """Distance-based resampling as fallback method"""
        resampled = []
        
        # Calculate cumulative distances
        distances = [0]
        for i in range(1, len(points)):
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
            distances.append(distances[-1] + dist)
        
        # Add distance from last to first point for closed contour
        total_dist = distances[-1] + np.linalg.norm(np.array(points[0]) - np.array(points[-1]))
        
        # Sample points at regular intervals
        for i in range(num_vertices):
            target_dist = (i * total_dist) / num_vertices
            
            # Find the segment containing this distance
            for j in range(len(distances) - 1):
                if distances[j] <= target_dist <= distances[j + 1]:
                    # Interpolate between points[j] and points[j+1]
                    t = (target_dist - distances[j]) / (distances[j + 1] - distances[j])
                    pt = (
                        int(points[j][0] * (1 - t) + points[j + 1][0] * t),
                        int(points[j][1] * (1 - t) + points[j + 1][1] * t)
                    )
                    resampled.append(list(pt))
                    break
            else:
                # Handle wrap-around for closed contour
                if target_dist > distances[-1]:
                    remaining_dist = target_dist - distances[-1]
                    last_segment_dist = np.linalg.norm(np.array(points[0]) - np.array(points[-1]))
                    if last_segment_dist > 0:
                        t = remaining_dist / last_segment_dist
                        pt = (
                            int(points[-1][0] * (1 - t) + points[0][0] * t),
                            int(points[-1][1] * (1 - t) + points[0][1] * t)
                        )
                        resampled.append(list(pt))
        
        return resampled
    
    def simplify_stroke(self, points, tolerance=5.0):
        """Simplify stroke points using Douglas-Peucker algorithm"""
        if len(points) < 3:
            return points
            
        # Convert to numpy array
        points = np.array(points)
        
        # Simple distance-based simplification
        simplified = [points[0]]
        
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - simplified[-1])
            if dist > tolerance:
                simplified.append(points[i])
        
        # Always include the last point
        if not np.array_equal(simplified[-1], points[-1]):
            simplified.append(points[-1])
            
        return simplified
    
    def check_vertex_selection(self, pos):
        """Check if clicking on a vertex for warping"""
        img_coords = self.widget_to_image_coords(pos)
        if img_coords[0] is None:
            return
            
        frame = self.parent_editor.current_frame_index if self.parent_editor else 0
        shapes = self.get_shapes_for_frame(frame)
        
        # Check each shape's vertices
        for shape_idx, shape in enumerate(shapes):
            for vertex_idx, vertex in enumerate(shape['vertices']):
                dist = np.linalg.norm(np.array(vertex) - np.array(img_coords))
                if dist < 10:  # Within 10 pixels
                    self.selected_shape_index = shape_idx
                    self.selected_vertex_index = vertex_idx
                    self.drag_start_pos = pos
                    self.original_vertex_pos = vertex.copy()
                    return
    
    def warp_vertex(self, pos):
        """Warp the selected vertex"""
        if self.selected_shape_index is None or self.selected_vertex_index is None:
            return
            
        img_coords = self.widget_to_image_coords(pos)
        if img_coords[0] is None:
            return
            
        frame = self.parent_editor.current_frame_index if self.parent_editor else 0
        
        # Update vertex position
        if frame in self.shape_keyframes:
            shape = self.shape_keyframes[frame][self.selected_shape_index]
            shape['vertices'][self.selected_vertex_index] = list(img_coords)
            
            # Apply relaxation if shift is held
            if self.relax_mode:
                self.relax_vertex(frame, self.selected_shape_index, self.selected_vertex_index)
            
            # Clear cache and update
            # Shapes will be recalculated
            
            # Don't normalize - it destroys our preserved start points
            # self.normalize_all_shape_keyframes()
            
            self.update_mask_from_shapes()
    
    def relax_vertex(self, frame, shape_idx, vertex_idx):
        """Apply relaxation to smooth vertex positions"""
        if frame not in self.shape_keyframes:
            return
            
        shape = self.shape_keyframes[frame][shape_idx]
        vertices = shape['vertices']
        
        if vertex_idx < 1 or vertex_idx >= len(vertices) - 1:
            return
            
        # Average with neighbors
        prev_vertex = np.array(vertices[vertex_idx - 1])
        next_vertex = np.array(vertices[vertex_idx + 1])
        current_vertex = np.array(vertices[vertex_idx])
        
        # Blend towards the average
        avg_pos = (prev_vertex + next_vertex) / 2
        relaxed_pos = current_vertex * 0.5 + avg_pos * 0.5
        
        vertices[vertex_idx] = relaxed_pos.tolist()
    
    def get_current_shapes(self):
        """Get the current shapes being drawn/edited (not from keyframes)"""
        # This returns any actively drawn shapes that haven't been saved as keyframes yet
        # For now, return empty list as shapes are immediately saved to keyframes
        return []
    
    def get_shapes_for_frame(self, frame):
        """Get interpolated shapes for a specific frame"""
        # Include spline mode in cache key to prevent conflicts between spline and linear interpolation
        spline_enabled = (hasattr(self.parent_editor, 'spline_interpolation_check') and 
                         self.parent_editor.spline_interpolation_check.isChecked())
        cache_key = (frame, self._cache_key, spline_enabled)
        if cache_key in self._shape_cache:
            return self._shape_cache[cache_key]
        
        # If it's a keyframe, return the shapes as-is (don't re-normalize)
        if frame in self.shape_keyframes:
            # Return deep copies to avoid modifying originals
            shapes = [shape.copy() for shape in self.shape_keyframes[frame]]
            self._shape_cache[cache_key] = shapes
            return shapes
            
        # Find surrounding keyframes
        keyframes = sorted(self.shape_keyframes.keys())
        if not keyframes:
            return []
            
        # Find previous and next keyframes
        prev_frame = None
        next_frame = None
        
        for kf in keyframes:
            if kf <= frame:
                prev_frame = kf
            if kf >= frame and next_frame is None:
                next_frame = kf
                
        # Interpolate shapes
        shapes = []
        if prev_frame is not None and next_frame is not None and prev_frame != next_frame:
            # Use spline interpolation if enabled, otherwise linear
            if spline_enabled:
                # Try spline interpolation (if enough keyframes)
                spline_shapes = self.interpolate_shapes_spline(frame)
                if spline_shapes is not None:
                    shapes = spline_shapes
                else:
                    # Fall back to linear interpolation if spline isn't possible
                    t = (frame - prev_frame) / (next_frame - prev_frame)
                    shapes = self.interpolate_shapes(
                        self.shape_keyframes[prev_frame],
                        self.shape_keyframes[next_frame],
                        t
                    )
            else:
                # Use linear interpolation
                t = (frame - prev_frame) / (next_frame - prev_frame)
                shapes = self.interpolate_shapes(
                    self.shape_keyframes[prev_frame],
                    self.shape_keyframes[next_frame],
                    t
                )
        elif prev_frame is not None:
            # Return copy of keyframe shapes WITHOUT resampling (they're already normalized)
            if self.shape_keyframes[prev_frame]:
                shapes = [shape.copy() for shape in self.shape_keyframes[prev_frame]]
        elif next_frame is not None:
            # Return copy of keyframe shapes WITHOUT resampling (they're already normalized)
            if self.shape_keyframes[next_frame]:
                shapes = [shape.copy() for shape in self.shape_keyframes[next_frame]]
        
        # Cache the result
        self._shape_cache[cache_key] = shapes
        return shapes
    
    def interpolate_shapes(self, shapes1, shapes2, t):
        """Interpolate between two sets of shapes"""
        interpolated = []
        
        # Match shapes by index
        max_shapes = max(len(shapes1), len(shapes2))
        
        for i in range(max_shapes):
            # Handle shape appearance/disappearance
            if i >= len(shapes1):
                # Shape appears - fade in
                shape2 = shapes2[i]
                if t > 0.5:  # Appear in second half
                    interpolated.append(shape2)
            elif i >= len(shapes2):
                # Shape disappears - fade out
                shape1 = shapes1[i]
                if t < 0.5:  # Disappear in first half
                    interpolated.append(shape1)
            else:
                # Both shapes exist - interpolate
                shape1 = shapes1[i]
                shape2 = shapes2[i]
                
                # Get vertices directly - they should already be properly resampled
                # when they were created as keyframes
                vertices1 = shape1['vertices']
                vertices2 = shape2['vertices']
                
                # Ensure both shapes have the same vertex count for interpolation
                # If they don't, resample to match instead of skipping
                if len(vertices1) != len(vertices2):
                    target_count = max(len(vertices1), len(vertices2))
                    
                    # Resample the smaller shape to match the larger one
                    if len(vertices1) != target_count:
                        vertices1 = self.resample_vertices_simple(vertices1, target_count)
                    if len(vertices2) != target_count:
                        vertices2 = self.resample_vertices_simple(vertices2, target_count)
                
                # Interpolate vertices directly without resampling
                interp_vertices = []
                for v1, v2 in zip(vertices1, vertices2):
                    interp_v = [
                        v1[0] * (1 - t) + v2[0] * t,
                        v1[1] * (1 - t) + v2[1] * t
                    ]
                    interp_vertices.append(interp_v)
                
                # Use the interpolated vertices directly without resampling
                interpolated.append({
                    'vertices': interp_vertices,
                    'closed': shape1.get('closed', True),
                    'visible': True,
                    'vertex_count': len(interp_vertices)  # Preserve vertex count
                })
        
        return interpolated
    
    def catmull_rom_point(self, P0, P1, P2, P3, t):
        """Calculate point on Catmull-Rom spline curve
        
        Args:
            P0, P1, P2, P3: Control points [x, y]
            t: Parameter from 0 to 1
            
        Returns:
            Interpolated point [x, y] between P1 and P2
        """
        t2 = t * t
        t3 = t2 * t
        
        return [
            0.5 * ((2 * P1[0]) + (-P0[0] + P2[0]) * t + 
                   (2*P0[0] - 5*P1[0] + 4*P2[0] - P3[0]) * t2 + 
                   (-P0[0] + 3*P1[0] - 3*P2[0] + P3[0]) * t3),
            0.5 * ((2 * P1[1]) + (-P0[1] + P2[1]) * t + 
                   (2*P0[1] - 5*P1[1] + 4*P2[1] - P3[1]) * t2 + 
                   (-P0[1] + 3*P1[1] - 3*P2[1] + P3[1]) * t3)
        ]
    
    def get_surrounding_keyframes(self, target_frame):
        """Get the 4 keyframes needed for spline interpolation around target frame
        
        Returns:
            (prev_prev, prev, next, next_next) frame indices or None if not available
        """
        if not hasattr(self, 'shape_keyframes') or len(self.shape_keyframes) < 2:
            return None, None, None, None
            
        # Get sorted list of keyframe indices
        keyframe_indices = sorted(self.shape_keyframes.keys())
        
        # Find the two keyframes that bracket the target frame
        prev_frame = None
        next_frame = None
        
        for i, frame in enumerate(keyframe_indices):
            if frame <= target_frame:
                prev_frame = frame
            if frame > target_frame and next_frame is None:
                next_frame = frame
                break
        
        if prev_frame is None or next_frame is None:
            return None, None, None, None
            
        # Find the frame before prev and after next
        prev_index = keyframe_indices.index(prev_frame)
        next_index = keyframe_indices.index(next_frame)
        
        prev_prev = keyframe_indices[prev_index - 1] if prev_index > 0 else None
        next_next = keyframe_indices[next_index + 1] if next_index < len(keyframe_indices) - 1 else None
        
        return prev_prev, prev_frame, next_frame, next_next
    
    def extrapolate_keyframe_position(self, known1, known2, is_before=True):
        """Extrapolate a control point position for spline interpolation
        
        Args:
            known1, known2: Two known keyframe positions
            is_before: True if extrapolating before known1, False if after known2
            
        Returns:
            Extrapolated control point position
        """
        if is_before:
            # Extrapolate before known1 using the direction from known2 to known1
            direction = known1 - known2
            return known1 + direction
        else:
            # Extrapolate after known2 using the direction from known1 to known2
            direction = known2 - known1
            return known2 + direction
    
    def interpolate_shapes_spline(self, frame):
        """Interpolate shapes using Catmull-Rom splines for smooth motion
        
        Args:
            frame: Target frame for interpolation
            
        Returns:
            List of interpolated shapes or None if spline interpolation not possible
        """
        # Check if spline interpolation is enabled
        if (not hasattr(self.parent_editor, 'spline_interpolation_check') or 
            not self.parent_editor.spline_interpolation_check.isChecked()):
            return None
            
        # Get surrounding keyframes
        prev_prev, prev_frame, next_frame, next_next = self.get_surrounding_keyframes(frame)
        
        # Need at least 3 keyframes for meaningful spline interpolation
        if prev_frame is None or next_frame is None:
            return None
            
        # Get the shapes from the two main keyframes
        shapes1 = self.shape_keyframes[prev_frame]
        shapes2 = self.shape_keyframes[next_frame]
        
        # Calculate interpolation parameter
        t = (frame - prev_frame) / (next_frame - prev_frame)
        
        interpolated = []
        max_shapes = max(len(shapes1), len(shapes2))
        
        for i in range(max_shapes):
            # Handle shape appearance/disappearance (same as linear)
            if i >= len(shapes1):
                shape2 = shapes2[i]
                if t > 0.5:
                    interpolated.append(shape2)
            elif i >= len(shapes2):
                shape1 = shapes1[i]
                if t < 0.5:
                    interpolated.append(shape1)
            else:
                # Both shapes exist - apply spline interpolation
                shape1 = shapes1[i]
                shape2 = shapes2[i]
                
                vertices1 = shape1['vertices']
                vertices2 = shape2['vertices']
                
                if len(vertices1) != len(vertices2):
                    # Fall back to linear for mismatched vertex counts
                    interp_vertices = []
                    for v1, v2 in zip(vertices1, vertices2):
                        interp_v = [
                            v1[0] * (1 - t) + v2[0] * t,
                            v1[1] * (1 - t) + v2[1] * t
                        ]
                        interp_vertices.append(interp_v)
                else:
                    # Apply Catmull-Rom spline interpolation to each vertex
                    interp_vertices = []
                    
                    # Get control shapes for spline (if available)
                    shapes0 = None
                    shapes3 = None
                    
                    if prev_prev is not None and i < len(self.shape_keyframes[prev_prev]):
                        shapes0 = self.shape_keyframes[prev_prev][i]['vertices']
                    if next_next is not None and i < len(self.shape_keyframes[next_next]):
                        shapes3 = self.shape_keyframes[next_next][i]['vertices']
                    
                    # Interpolate each vertex pair with spline
                    for j, (v1, v2) in enumerate(zip(vertices1, vertices2)):
                        # Get control points for this vertex
                        if shapes0 is not None and j < len(shapes0):
                            P0 = shapes0[j]
                        else:
                            # Extrapolate P0
                            P0 = [v1[0] + (v1[0] - v2[0]), v1[1] + (v1[1] - v2[1])]
                        
                        P1 = v1
                        P2 = v2
                        
                        if shapes3 is not None and j < len(shapes3):
                            P3 = shapes3[j]
                        else:
                            # Extrapolate P3
                            P3 = [v2[0] + (v2[0] - v1[0]), v2[1] + (v2[1] - v1[1])]
                        
                        # Calculate spline point
                        spline_point = self.catmull_rom_point(P0, P1, P2, P3, t)
                        interp_vertices.append(spline_point)
                
                interpolated.append({
                    'vertices': interp_vertices,
                    'closed': shape1.get('closed', True),
                    'visible': True,
                    'vertex_count': len(interp_vertices)
                })
        
        return interpolated
    
    def ensure_consistent_winding(self, vertices):
        """Ensure all shapes have counter-clockwise winding order"""
        if len(vertices) < 3:
            return vertices
            
        # Calculate signed area using the shoelace formula
        signed_area = 0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            signed_area += (vertices[j][0] - vertices[i][0]) * (vertices[j][1] + vertices[i][1])
        
        # If clockwise (negative area), reverse to make counter-clockwise
        if signed_area < 0:
            # print(f"ensure_consistent_winding: Reversing vertices (was clockwise)")
            return vertices[::-1]
        
        return vertices
    
    
    def resample_vertices(self, vertices, target_count, preserve_start_point=None, reference_signature=None):
        """Resample vertices to a specific count preserving the actual shape curve
        
        Args:
            vertices: List of vertices
            target_count: Target number of vertices
            preserve_start_point: Deprecated - no longer used
            reference_signature: Optional reference signature for consistent alignment
        """
        if len(vertices) == target_count and preserve_start_point is None and reference_signature is None:
            # Even if count matches, normalize starting point
            vertices = list(vertices)
            
        # Convert to list if needed
        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()
        
        # Choose starting point method based on whether we have a reference
        if reference_signature:
            # Use adaptive reference-based alignment for consistent parametrization
            start_idx = self.find_matching_vertex_adaptive(vertices, reference_signature)
            
            # Debug log for adaptive alignment
            if len(vertices) > 10:
                start_vertex = vertices[start_idx]
                algorithm = reference_signature.get('algorithm', 'unknown')
                shape_type = reference_signature.get('shape_type', 'unknown')
        else:
            # Use enhanced fallback logic that analyzes shape type for new shapes
            # This replaces the old 12 o'clock algorithm with shape-aware selection
            start_idx = self.get_fallback_starting_point(vertices, "no reference signature - first shape")
        
        # Reorder vertices to start from the consistent point
        vertices = vertices[start_idx:] + vertices[:start_idx]
        
        # If already at target count, return normalized vertices
        if len(vertices) == target_count:
            # But still verify the count is exactly what was requested
            if len(vertices) != target_count:
                print(f"Warning: Vertex count mismatch after normalization: {len(vertices)} vs {target_count}")
            return vertices
            
        # Use curve-aware resampling that preserves the actual shape
        return self.resample_vertices_curve_aware(vertices, target_count)
    
    def resample_vertices_curve_aware(self, vertices, target_count):
        """Resample vertices with even distribution"""
        if len(vertices) < 3:
            return vertices
            
        try:
            # Use simple distance-based resampling for predictable results
            points = np.array(vertices, dtype=np.float32)
            
            # Calculate cumulative distances
            distances = [0]
            for i in range(1, len(points)):
                dist = np.linalg.norm(points[i] - points[i-1])
                distances.append(distances[-1] + dist)
            
            # Add closing distance
            total_dist = distances[-1] + np.linalg.norm(points[0] - points[-1])
            
            # Resample at regular intervals
            resampled = []
            for i in range(target_count):
                target_dist = (i * total_dist) / target_count
                
                # Find segment
                for j in range(len(distances) - 1):
                    if distances[j] <= target_dist <= distances[j + 1]:
                        segment_length = distances[j + 1] - distances[j]
                        if segment_length > 0:
                            t = (target_dist - distances[j]) / segment_length
                        else:
                            t = 0
                        pt = points[j] * (1 - t) + points[j + 1] * t
                        resampled.append(pt.tolist())
                        break
                else:
                    # Handle wrap-around
                    if target_dist > distances[-1]:
                        remaining = target_dist - distances[-1]
                        last_dist = np.linalg.norm(points[0] - points[-1])
                        if last_dist > 0:
                            t = remaining / last_dist
                            pt = points[-1] * (1 - t) + points[0] * t
                            resampled.append(pt.tolist())
            
            # Ensure we have exactly target_count vertices
            if len(resampled) != target_count:
                print(f"Warning: Curve-aware resampling returned {len(resampled)} instead of {target_count} vertices")
                # Fallback to simple resampling for consistency
                return self.resample_vertices_simple(vertices, target_count)
            
            return resampled
            
        except Exception as e:
            print(f"Error in curve-aware resampling: {e}")
            # Fallback to simple resampling
            return self.resample_vertices_simple(vertices, target_count)
    
    
    def resample_vertices_simple(self, vertices, target_count):
        """Simple linear resampling (fallback method)"""
        # Convert to numpy array for easier manipulation
        points = np.array(vertices)
        
        # Calculate cumulative distances
        distances = [0]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            distances.append(distances[-1] + dist)
        
        # Add closing distance
        total_dist = distances[-1] + np.linalg.norm(points[0] - points[-1])
        
        # Resample at regular intervals
        resampled = []
        for i in range(target_count):
            target_dist = (i * total_dist) / target_count
            
            # Find segment
            for j in range(len(distances) - 1):
                if distances[j] <= target_dist <= distances[j + 1]:
                    t = (target_dist - distances[j]) / (distances[j + 1] - distances[j])
                    pt = points[j] * (1 - t) + points[j + 1] * t
                    resampled.append(pt.tolist())
                    break
            else:
                # Handle wrap-around
                if target_dist > distances[-1]:
                    remaining = target_dist - distances[-1]
                    last_dist = np.linalg.norm(points[0] - points[-1])
                    if last_dist > 0:
                        t = remaining / last_dist
                        pt = points[-1] * (1 - t) + points[0] * t
                        resampled.append(pt.tolist())
        
        # Ensure we have exactly target_count vertices
        if len(resampled) != target_count:
            print(f"Warning: Simple resampling returned {len(resampled)} instead of {target_count} vertices")
            # If we have fewer vertices, pad by duplicating the last vertex
            while len(resampled) < target_count:
                if resampled:
                    resampled.append(resampled[-1])
                else:
                    resampled.append([0, 0])  # Fallback point
            # If we have too many vertices, truncate
            resampled = resampled[:target_count]
        
        return resampled
    
    def compute_local_geometry_signature(self, vertices, vertex_index, radius=3):
        """Compute a local geometry signature for a vertex to enable consistent matching
        
        Args:
            vertices: List of vertices defining the shape
            vertex_index: Index of the vertex to compute signature for
            radius: Number of neighboring vertices to include on each side
            
        Returns:
            dict: Signature containing local geometric characteristics
        """
        if len(vertices) < 3:
            return {}
            
        n = len(vertices)
        signature = {}
        
        try:
            # Get the central vertex and its neighbors
            central = np.array(vertices[vertex_index])
            neighbors = []
            
            # Collect neighboring vertices (wrapping around)
            for offset in range(-radius, radius + 1):
                idx = (vertex_index + offset) % n
                neighbors.append(np.array(vertices[idx]))
            
            # Compute local edge lengths
            edge_lengths = []
            for i in range(len(neighbors) - 1):
                length = np.linalg.norm(neighbors[i + 1] - neighbors[i])
                edge_lengths.append(length)
            
            # Compute local angles at each vertex
            angles = []
            for i in range(1, len(neighbors) - 1):
                vec1 = neighbors[i - 1] - neighbors[i] 
                vec2 = neighbors[i + 1] - neighbors[i]
                
                # Normalize vectors
                len1 = np.linalg.norm(vec1)
                len2 = np.linalg.norm(vec2)
                
                if len1 > 0 and len2 > 0:
                    vec1 = vec1 / len1
                    vec2 = vec2 / len2
                    
                    # Compute angle
                    dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    angles.append(angle)
                else:
                    angles.append(0.0)
            
            # Compute curvature approximation at central vertex
            if len(neighbors) >= 3:
                # Use three-point curvature approximation
                p1 = neighbors[radius - 1] if radius > 0 else neighbors[0]
                p2 = neighbors[radius]  # Central vertex
                p3 = neighbors[radius + 1] if radius < len(neighbors) - 1 else neighbors[-1]
                
                # Compute curvature using cross product formula
                v1 = p2 - p1
                v2 = p3 - p2
                
                cross = v1[0] * v2[1] - v1[1] * v2[0]  # 2D cross product
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 > 0 and norm_v2 > 0:
                    curvature = abs(cross) / (norm_v1 * norm_v2)
                else:
                    curvature = 0.0
            else:
                curvature = 0.0
            
            # Create signature
            signature = {
                'edge_lengths': edge_lengths,
                'angles': angles,
                'curvature': curvature,
                'avg_edge_length': np.mean(edge_lengths) if edge_lengths else 0.0,
                'std_edge_length': np.std(edge_lengths) if edge_lengths else 0.0,
                'avg_angle': np.mean(angles) if angles else 0.0,
                'position': central.tolist()
            }
            
        except Exception as e:
            print(f"Error computing geometry signature: {e}")
            # Return minimal signature
            signature = {
                'position': vertices[vertex_index] if vertex_index < len(vertices) else [0, 0],
                'curvature': 0.0,
                'avg_edge_length': 0.0,
                'avg_angle': 0.0
            }
        
        return signature
    
    def compute_signature_distance(self, sig1, sig2):
        """Compute distance between two geometry signatures
        
        Args:
            sig1, sig2: Geometry signatures from compute_local_geometry_signature
            
        Returns:
            float: Distance score (lower = more similar)
        """
        if not sig1 or not sig2:
            return float('inf')
        
        try:
            # Position distance (normalized by average edge length)
            pos1 = np.array(sig1.get('position', [0, 0]))
            pos2 = np.array(sig2.get('position', [0, 0])) 
            pos_dist = np.linalg.norm(pos1 - pos2)
            
            # Normalize by edge length scale
            avg_edge1 = sig1.get('avg_edge_length', 1.0)
            avg_edge2 = sig2.get('avg_edge_length', 1.0)
            avg_edge = (avg_edge1 + avg_edge2) / 2.0
            if avg_edge > 0:
                pos_dist = pos_dist / avg_edge
            
            # Curvature difference
            curv1 = sig1.get('curvature', 0.0)
            curv2 = sig2.get('curvature', 0.0)
            curv_dist = abs(curv1 - curv2)
            
            # Edge length difference (relative)
            edge1 = sig1.get('avg_edge_length', 0.0)
            edge2 = sig2.get('avg_edge_length', 0.0)
            if edge1 > 0 and edge2 > 0:
                edge_dist = abs(edge1 - edge2) / max(edge1, edge2)
            else:
                edge_dist = 1.0 if edge1 != edge2 else 0.0
            
            # Angle difference
            angle1 = sig1.get('avg_angle', 0.0)
            angle2 = sig2.get('avg_angle', 0.0)
            angle_dist = abs(angle1 - angle2) / np.pi  # Normalize by π
            
            # Weighted combination (position is most important)
            total_distance = (
                pos_dist * 0.5 +      # Position weight
                curv_dist * 0.2 +     # Curvature weight  
                edge_dist * 0.2 +     # Edge length weight
                angle_dist * 0.1      # Angle weight
            )
            
            return total_distance
            
        except Exception as e:
            print(f"Error computing signature distance: {e}")
            return float('inf')
    
    def find_best_matching_vertex(self, vertices, reference_signature):
        """Find the vertex that best matches a reference geometry signature
        
        Args:
            vertices: List of vertices in the shape
            reference_signature: Reference signature to match against
            
        Returns:
            int: Index of the best matching vertex
        """
        if not vertices or not reference_signature:
            return 0
            
        best_vertex = 0
        best_distance = float('inf')
        
        # Compute signature for each vertex and find best match
        for i in range(len(vertices)):
            vertex_signature = self.compute_local_geometry_signature(vertices, i)
            distance = self.compute_signature_distance(vertex_signature, reference_signature)
            
            if distance < best_distance:
                best_distance = distance
                best_vertex = i
        
        return best_vertex
    
    def create_reference_signature(self, vertices, start_vertex_index):
        """Create a reference signature for consistent parametrization
        
        Args:
            vertices: List of vertices
            start_vertex_index: Index of the starting vertex
            
        Returns:
            dict: Reference signature for future matching
        """
        if not vertices or start_vertex_index >= len(vertices):
            return {}
            
        # Compute detailed signature for the reference vertex
        signature = self.compute_local_geometry_signature(vertices, start_vertex_index, radius=3)
        
        # Add metadata
        signature['vertex_count'] = len(vertices)
        signature['creation_timestamp'] = time.time()
        
        return signature
    
    def analyze_shape_geometry(self, vertices):
        """Comprehensive shape analysis to determine type and characteristics
        
        Args:
            vertices: List of [x, y] vertices defining the shape
            
        Returns:
            dict: Shape analysis results with type classification and metrics
        """
        if len(vertices) < 3:
            return {'type': 'invalid', 'confidence': 0.0}
        
        try:
            points = np.array(vertices, dtype=np.float32)
            
            # 1. Bounding box aspect ratio analysis
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            width = max_x - min_x
            height = max_y - min_y
            
            if width < 1 or height < 1:
                return {'type': 'degenerate', 'confidence': 0.0}
            
            bbox_aspect_ratio = max(width, height) / min(width, height)
            
            # 2. Principal Component Analysis for oriented bounding box
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            
            # Compute covariance matrix
            cov = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eig(cov)
            
            # Sort by eigenvalue (largest first)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Principal axis ratio (how elongated along main axis)
            pca_ratio = eigenvals[0] / max(eigenvals[1], 0.001)
            
            # 3. Compactness analysis (perimeter-to-area ratio)
            # Calculate area using shoelace formula
            area = abs(sum(vertices[i][0] * vertices[i+1][1] - 
                          vertices[i+1][0] * vertices[i][1] 
                          for i in range(-1, len(vertices)-1))) / 2
            
            # Calculate perimeter
            perimeter = sum(np.linalg.norm(points[i] - points[i-1]) 
                           for i in range(len(points)))
            
            # Compactness metric (1.0 = perfect circle, lower = more elongated)
            compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # 4. Curvature variance analysis
            curvatures = []
            n = len(vertices)
            for i in range(n):
                p1 = points[i-1]
                p2 = points[i]
                p3 = points[(i+1) % n]
                
                v1 = p2 - p1
                v2 = p3 - p2
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                
                if norm_product > 0:
                    curvature = abs(cross) / norm_product
                    curvatures.append(curvature)
            
            avg_curvature = np.mean(curvatures) if curvatures else 0
            curvature_variance = np.var(curvatures) if curvatures else 0
            
            # 5. Shape classification using multi-criteria scoring
            thin_strip_score = 0
            circular_score = 0
            
            # Thin strip indicators
            if bbox_aspect_ratio > 4.0: thin_strip_score += 2
            if pca_ratio > 10.0: thin_strip_score += 3
            if compactness < 0.2: thin_strip_score += 2
            if curvature_variance < 0.01: thin_strip_score += 1
            if avg_curvature < 0.1: thin_strip_score += 1
            
            # Circular/compact indicators
            if bbox_aspect_ratio < 1.5: circular_score += 2
            if compactness > 0.6: circular_score += 3
            if curvature_variance < 0.05 and avg_curvature > 0.3: circular_score += 2
            
            # Determine shape type
            max_score = max(thin_strip_score, circular_score)
            if thin_strip_score >= 4 and thin_strip_score >= circular_score:
                shape_type = 'thin_strip'
                confidence = min(thin_strip_score / 9.0, 1.0)
            elif circular_score >= 4 and circular_score >= thin_strip_score:
                shape_type = 'circular'
                confidence = min(circular_score / 9.0, 1.0)
            else:
                shape_type = 'irregular'
                confidence = 0.5
            
            return {
                'type': shape_type,
                'confidence': confidence,
                'bbox_aspect_ratio': bbox_aspect_ratio,
                'pca_ratio': pca_ratio,
                'compactness': compactness,
                'avg_curvature': avg_curvature,
                'curvature_variance': curvature_variance,
                'principal_axis': eigenvecs[:, 0],
                'secondary_axis': eigenvecs[:, 1],
                'centroid': centroid,
                'width': width,
                'height': height,
                'area': area,
                'perimeter': perimeter
            }
            
        except Exception as e:
            return {'type': 'irregular', 'confidence': 0.0}
    
    def find_thin_strip_reference_points(self, vertices, shape_analysis):
        """Find stable reference points for thin strip shapes using principal axis
        
        Args:
            vertices: List of vertices
            shape_analysis: Results from analyze_shape_geometry
            
        Returns:
            tuple: (start_point_index, end_point_index) or (start_index, None) if single point
        """
        if shape_analysis['type'] != 'thin_strip':
            return None, None
        
        try:
            points = np.array(vertices, dtype=np.float32)
            principal_axis = shape_analysis['principal_axis']
            centroid = shape_analysis['centroid']
            
            # Project all points onto the principal axis
            projections = []
            for i, point in enumerate(points):
                # Vector from centroid to point
                vec = point - centroid
                # Project onto principal axis
                projection = np.dot(vec, principal_axis)
                projections.append((projection, i))
            
            # Find the points with minimum and maximum projections (extremes along principal axis)
            projections.sort(key=lambda x: x[0])
            
            min_proj_idx = projections[0][1]  # Most negative projection
            max_proj_idx = projections[-1][1]  # Most positive projection
            
            # For thin strips, use the extremal point with maximum distance from centroid as reference
            min_point = points[min_proj_idx]
            max_point = points[max_proj_idx]
            
            min_dist = np.linalg.norm(min_point - centroid)
            max_dist = np.linalg.norm(max_point - centroid)
            
            # Choose the extremal point furthest from centroid as primary reference
            if max_dist >= min_dist:
                primary_idx = max_proj_idx
                secondary_idx = min_proj_idx
            else:
                primary_idx = min_proj_idx
                secondary_idx = max_proj_idx
            
            
            return primary_idx, secondary_idx
            
        except Exception as e:
            return None, None
    
    def find_circular_reference_point(self, vertices, shape_analysis):
        """Find stable reference point for circular/compact shapes
        
        Args:
            vertices: List of vertices
            shape_analysis: Results from analyze_shape_geometry
            
        Returns:
            int: Index of reference vertex
        """
        try:
            points = np.array(vertices, dtype=np.float32)
            centroid = shape_analysis['centroid']
            
            # For circular shapes, find the topmost point (12 o'clock position)
            # This is more stable than the previous centroid-based approach
            
            # Find point with minimum Y coordinate (topmost in image coordinates)
            min_y = np.min(points[:, 1])
            topmost_candidates = [i for i, p in enumerate(points) if abs(p[1] - min_y) < 1.0]
            
            if len(topmost_candidates) == 1:
                return topmost_candidates[0]
            
            # If multiple points at the top, choose the one closest to the center X
            center_x = centroid[0]
            best_idx = topmost_candidates[0]
            best_dist = abs(points[best_idx][0] - center_x)
            
            for idx in topmost_candidates[1:]:
                dist = abs(points[idx][0] - center_x)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            return best_idx
            
        except Exception as e:
            return 0
    
    def create_adaptive_reference_signature(self, vertices):
        """Create a reference signature adapted to the shape type
        
        Args:
            vertices: List of vertices for the reference shape
            
        Returns:
            dict: Adaptive reference signature with shape-aware information
        """
        # Analyze the shape to determine appropriate parametrization strategy
        shape_analysis = self.analyze_shape_geometry(vertices)
        
        reference_signature = {
            'shape_type': shape_analysis['type'],
            'confidence': shape_analysis['confidence'],
            'creation_timestamp': time.time(),
            'vertex_count': len(vertices)
        }
        
        if shape_analysis['type'] == 'thin_strip' and shape_analysis['confidence'] > 0.5:
            # Use principal axis endpoints for thin strips
            primary_idx, secondary_idx = self.find_thin_strip_reference_points(vertices, shape_analysis)
            if primary_idx is not None:
                reference_signature.update({
                    'algorithm': 'thin_strip_principal_axis',
                    'primary_reference_idx': primary_idx,
                    'secondary_reference_idx': secondary_idx,
                    'principal_axis': shape_analysis['principal_axis'].tolist(),
                    'centroid': shape_analysis['centroid'].tolist(),
                    'pca_ratio': shape_analysis['pca_ratio'],
                    'primary_position': vertices[primary_idx]
                })
            else:
                # Fallback to geometry signature for thin strips
                reference_signature['algorithm'] = 'geometry_signature_fallback'
                reference_signature.update(self.compute_local_geometry_signature(vertices, 0, radius=2))
                
        elif shape_analysis['type'] == 'circular' and shape_analysis['confidence'] > 0.6:
            # Use topmost point for circular shapes
            ref_idx = self.find_circular_reference_point(vertices, shape_analysis)
            reference_signature.update({
                'algorithm': 'circular_topmost',
                'reference_idx': ref_idx,
                'centroid': shape_analysis['centroid'].tolist(),
                'compactness': shape_analysis['compactness'],
                'reference_position': vertices[ref_idx]
            })
            
        else:
            # Use enhanced geometry signature for irregular shapes
            reference_signature['algorithm'] = 'enhanced_geometry_signature'
            
            # Try to find the most stable vertex using multiple criteria
            best_idx = 0
            best_stability_score = -1
            
            for i in range(len(vertices)):
                # Compute stability score based on local geometry
                local_sig = self.compute_local_geometry_signature(vertices, i, radius=3)
                
                # Higher curvature and more distinct local geometry = more stable reference
                stability_score = (local_sig.get('curvature', 0) * 10 + 
                                 local_sig.get('avg_edge_length', 0) / 10 +
                                 local_sig.get('angle_variance', 0) * 5)
                
                if stability_score > best_stability_score:
                    best_stability_score = stability_score
                    best_idx = i
            
            reference_signature.update({
                'reference_idx': best_idx,
                'stability_score': best_stability_score
            })
            reference_signature.update(self.compute_local_geometry_signature(vertices, best_idx, radius=3))
        
        return reference_signature
    
    def find_matching_vertex_adaptive(self, vertices, reference_signature):
        """Find matching vertex using adaptive algorithm based on reference type with validation and fallbacks
        
        Args:
            vertices: List of vertices in the current shape
            reference_signature: Reference signature from create_adaptive_reference_signature
            
        Returns:
            int: Index of best matching vertex
        """
        if not reference_signature or len(vertices) < 3:
            return self.get_fallback_starting_point(vertices, "no reference signature")
        
        try:
            # First, validate compatibility with reference
            is_compatible, confidence = self.validate_reference_compatibility(vertices, reference_signature)
            
            if not is_compatible or confidence < 0.3:
                return self.get_fallback_starting_point(vertices, f"low compatibility: {confidence:.2f}")
            
            algorithm = reference_signature.get('algorithm', 'geometry_signature')
            
            # Try the algorithm specified in the reference
            if algorithm == 'thin_strip_principal_axis':
                result_idx = self.find_thin_strip_matching_vertex(vertices, reference_signature)
            elif algorithm == 'circular_topmost':
                result_idx = self.find_circular_matching_vertex(vertices, reference_signature)
            else:
                # Enhanced geometry signature matching
                result_idx = self.find_geometry_signature_matching_vertex(vertices, reference_signature)
            
            # Validate the result
            if result_idx is None or result_idx < 0 or result_idx >= len(vertices):
                print(f"[Adaptive Matching] Invalid result index {result_idx}, using fallback")
                return self.get_fallback_starting_point(vertices, f"invalid result index: {result_idx}")
            
            return result_idx
                
        except Exception as e:
            return self.get_fallback_starting_point(vertices, f"exception: {str(e)}")
    
    def find_thin_strip_matching_vertex(self, vertices, reference_signature):
        """Find matching vertex for thin strip using principal axis projection"""
        try:
            # Analyze current shape
            current_analysis = self.analyze_shape_geometry(vertices)
            
            # Check if current shape is still a thin strip
            if current_analysis['type'] != 'thin_strip' or current_analysis['confidence'] < 0.4:
                print(f"[Thin Strip Match] Shape type changed, falling back to geometry signature")
                return self.find_geometry_signature_matching_vertex(vertices, reference_signature)
            
            # Use principal axis to find corresponding vertex
            current_points = np.array(vertices, dtype=np.float32)
            current_centroid = current_analysis['centroid']
            current_principal_axis = current_analysis['principal_axis']
            
            # Reference principal axis and position
            ref_principal_axis = np.array(reference_signature['principal_axis'])
            ref_centroid = np.array(reference_signature['centroid'])
            ref_position = np.array(reference_signature['primary_position'])
            
            # Project reference position onto current shape's coordinate system
            # Find which end of the current principal axis corresponds to the reference
            
            projections = []
            for i, point in enumerate(current_points):
                vec = point - current_centroid
                projection = np.dot(vec, current_principal_axis)
                projections.append((projection, i))
            
            # Sort by projection to find extremes
            projections.sort(key=lambda x: x[0])
            
            # Determine which extreme (min or max projection) corresponds to reference
            # by comparing the relative position along the principal axis
            ref_vec = ref_position - ref_centroid
            ref_projection = np.dot(ref_vec, ref_principal_axis)
            
            # Find the extreme that has the same sign as the reference projection
            if ref_projection >= 0:
                # Reference was on positive side, use max projection
                best_idx = projections[-1][1]
            else:
                # Reference was on negative side, use min projection
                best_idx = projections[0][1]
            
            print(f"[Thin Strip Match] Found matching vertex {best_idx} using principal axis projection")
            return best_idx
            
        except Exception as e:
            return self.find_geometry_signature_matching_vertex(vertices, reference_signature)
    
    def find_circular_matching_vertex(self, vertices, reference_signature):
        """Find matching vertex for circular shape using topmost point strategy"""
        try:
            current_analysis = self.analyze_shape_geometry(vertices)
            
            # For circular shapes, always try to find the topmost point
            points = np.array(vertices, dtype=np.float32)
            min_y = np.min(points[:, 1])
            topmost_candidates = [i for i, p in enumerate(points) if abs(p[1] - min_y) < 2.0]
            
            if len(topmost_candidates) == 1:
                return topmost_candidates[0]
            
            # If multiple candidates, choose the one closest to the center X
            centroid = current_analysis.get('centroid', np.mean(points, axis=0))
            center_x = centroid[0]
            
            best_idx = topmost_candidates[0]
            best_dist = abs(points[best_idx][0] - center_x)
            
            for idx in topmost_candidates[1:]:
                dist = abs(points[idx][0] - center_x)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            print(f"[Circular Match] Found matching vertex {best_idx} using topmost point strategy")
            return best_idx
            
        except Exception as e:
            return self.find_geometry_signature_matching_vertex(vertices, reference_signature)
    
    def find_geometry_signature_matching_vertex(self, vertices, reference_signature):
        """Find matching vertex using enhanced geometry signature matching"""
        # This is the existing geometry signature matching, enhanced
        return self.find_best_matching_vertex(vertices, reference_signature)
    
    def get_sequence_reference_signature(self):
        """Get or create the adaptive reference signature for consistent parametrization across the sequence
        
        Returns:
            Reference signature dictionary or None if no keyframes exist
        """
        if not hasattr(self, '_sequence_reference_signature'):
            # Find the first keyframe to use as reference
            if hasattr(self, 'shape_keyframes') and self.shape_keyframes:
                first_frame = min(self.shape_keyframes.keys())
                first_shapes = self.shape_keyframes[first_frame]
                if first_shapes and len(first_shapes) > 0:
                    # Use the first shape of the first keyframe as reference
                    first_shape = first_shapes[0]
                    if 'vertices' in first_shape and len(first_shape['vertices']) > 0:
                        # Create adaptive reference signature based on shape type
                        self._sequence_reference_signature = self.create_adaptive_reference_signature(
                            first_shape['vertices']
                        )
                    else:
                        self._sequence_reference_signature = None
                else:
                    self._sequence_reference_signature = None
            else:
                self._sequence_reference_signature = None
        
        return self._sequence_reference_signature
    
    def reset_sequence_reference_signature(self):
        """Reset the reference signature - call when sequence changes significantly"""
        if hasattr(self, '_sequence_reference_signature'):
            delattr(self, '_sequence_reference_signature')
    
    def validate_reference_compatibility(self, vertices, reference_signature):
        """Validate if the current shape is compatible with the reference signature
        
        Args:
            vertices: Current shape vertices
            reference_signature: Existing reference signature
            
        Returns:
            tuple: (is_compatible, confidence_score)
        """
        if not reference_signature or len(vertices) < 3:
            return False, 0.0
        
        try:
            # Analyze current shape
            current_analysis = self.analyze_shape_geometry(vertices)
            ref_shape_type = reference_signature.get('shape_type', 'unknown')
            
            # Check type compatibility
            if current_analysis['type'] == ref_shape_type:
                # Same type - high compatibility
                if current_analysis['type'] == 'thin_strip':
                    # For thin strips, check if PCA ratios are similar
                    current_pca_ratio = current_analysis['pca_ratio']
                    ref_pca_ratio = reference_signature.get('pca_ratio', 0)
                    
                    if ref_pca_ratio > 0:
                        ratio_similarity = min(current_pca_ratio, ref_pca_ratio) / max(current_pca_ratio, ref_pca_ratio)
                        compatibility = ratio_similarity > 0.5
                        confidence = ratio_similarity
                    else:
                        compatibility = current_pca_ratio > 8.0  # Still looks like thin strip
                        confidence = 0.7
                        
                elif current_analysis['type'] == 'circular':
                    # For circular shapes, check compactness similarity
                    current_compactness = current_analysis['compactness']
                    ref_compactness = reference_signature.get('compactness', 0)
                    
                    if ref_compactness > 0:
                        compactness_similarity = min(current_compactness, ref_compactness) / max(current_compactness, ref_compactness)
                        compatibility = compactness_similarity > 0.6
                        confidence = compactness_similarity
                    else:
                        compatibility = current_compactness > 0.5
                        confidence = 0.7
                else:
                    # Irregular shapes - moderate compatibility
                    compatibility = True
                    confidence = 0.6
                    
            else:
                # Different types - check if we can adapt
                if (ref_shape_type == 'thin_strip' and current_analysis['type'] == 'irregular' and 
                    current_analysis['bbox_aspect_ratio'] > 2.0):
                    # Thin strip became irregular but still elongated - can adapt
                    compatibility = True
                    confidence = 0.4
                elif (ref_shape_type == 'circular' and current_analysis['type'] == 'irregular' and
                      current_analysis['compactness'] > 0.4):
                    # Circular became irregular but still compact - can adapt
                    compatibility = True
                    confidence = 0.4
                else:
                    # Incompatible types
                    compatibility = False
                    confidence = 0.0
            
            return compatibility, confidence
            
        except Exception as e:
            return False, 0.0
    
    def get_fallback_starting_point(self, vertices, reason="unknown"):
        """Get a fallback starting point using the most stable available method
        
        Args:
            vertices: List of vertices
            reason: Reason for fallback (for debugging)
            
        Returns:
            int: Index of starting vertex
        """
        
        try:
            # Try shape analysis first for adaptive fallback
            shape_analysis = self.analyze_shape_geometry(vertices)
            
            if shape_analysis['type'] == 'thin_strip' and shape_analysis['confidence'] > 0.5:
                # Use extremal point for thin strips
                primary_idx, _ = self.find_thin_strip_reference_points(vertices, shape_analysis)
                if primary_idx is not None:
                    return primary_idx
            
            elif shape_analysis['type'] == 'circular' and shape_analysis['confidence'] > 0.6:
                # Use topmost point for circular shapes
                ref_idx = self.find_circular_reference_point(vertices, shape_analysis)
                return ref_idx
            
            # Default to enhanced 12 o'clock method
            points = np.array(vertices, dtype=np.float32)
            centroid = np.mean(points, axis=0)
            
            # Find vertex with minimum Y coordinate (topmost)
            min_y = np.min(points[:, 1])
            topmost_candidates = [i for i, p in enumerate(points) if abs(p[1] - min_y) < 2.0]
            
            if len(topmost_candidates) == 1:
                best_idx = topmost_candidates[0]
            else:
                # Multiple candidates - choose one closest to center X
                center_x = centroid[0]
                best_idx = topmost_candidates[0]
                best_dist = abs(points[best_idx][0] - center_x)
                
                for idx in topmost_candidates[1:]:
                    dist = abs(points[idx][0] - center_x)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
            
            return best_idx
            
        except Exception as e:
            return 0
    
    def update_mask_from_shapes(self):
        """Update the mask based on current shapes"""
        if self.mask is None:
            return
            
        # Clear the mask
        self.mask[:] = 0
        
        # Get shapes for current frame
        frame = self.parent_editor.current_frame_index if self.parent_editor else 0
        
        # If vertex slider is being dragged, use original shapes
        if (self.parent_editor and hasattr(self.parent_editor, '_vertex_slider_dragging') and 
            self.parent_editor._vertex_slider_dragging and 
            hasattr(self.parent_editor, '_original_shape_keyframes')):
            # Use original shapes
            shapes = []
            if frame in self.parent_editor._original_shape_keyframes:
                shapes = self.parent_editor._original_shape_keyframes[frame]
            else:
                # Get interpolated shapes from original keyframes
                keyframes = sorted(self.parent_editor._original_shape_keyframes.keys())
                if keyframes:
                    # Find surrounding keyframes
                    prev_frame = None
                    next_frame = None
                    for kf in keyframes:
                        if kf <= frame:
                            prev_frame = kf
                        if kf >= frame and next_frame is None:
                            next_frame = kf
                    
                    # Interpolate
                    if prev_frame is not None and next_frame is not None and prev_frame != next_frame:
                        t = (frame - prev_frame) / (next_frame - prev_frame)
                        shapes = self.interpolate_shapes(
                            self.parent_editor._original_shape_keyframes[prev_frame],
                            self.parent_editor._original_shape_keyframes[next_frame],
                            t
                        )
                    elif prev_frame is not None:
                        shapes = self.parent_editor._original_shape_keyframes[prev_frame].copy()
                    elif next_frame is not None:
                        shapes = self.parent_editor._original_shape_keyframes[next_frame].copy()
        # If we're in liquify mode and have deformed shapes, use those for display
        elif (self.drawing_mode == "liquify" and 
            hasattr(self, '_temp_deformed_shapes') and self._temp_deformed_shapes):
            shapes = self._temp_deformed_shapes
        else:
            shapes = self.get_shapes_for_frame(frame)
        
        # Draw each shape
        for shape in shapes:
            if shape['visible']:
                if self.should_use_smooth_shapes() and len(shape['vertices']) >= 3:
                    # Generate smooth curve points for mask creation
                    smooth_vertices = self.generate_smooth_vertices_for_mask(shape['vertices'], shape.get('closed', True))
                    if len(smooth_vertices) >= 3:
                        vertices = np.array(smooth_vertices, dtype=np.int32)
                        cv2.fillPoly(self.mask, [vertices], 255)
                else:
                    # Use original polygon method
                    vertices = np.array(shape['vertices'], dtype=np.int32)
                    if len(vertices) >= 3:
                        cv2.fillPoly(self.mask, [vertices], 255)
        
        # Add temporary interpolated shapes if any
        if (hasattr(self, '_temp_interpolated_shapes') and 
            self.parent_editor and 
            self.parent_editor.current_frame_index in self._temp_interpolated_shapes):
            # Overlay the temporary shape mask
            temp_mask = self._temp_interpolated_shapes[self.parent_editor.current_frame_index]
            self.mask = cv2.bitwise_or(self.mask, temp_mask)
        
        self.update()
        
        # Don't update timeline here - only actual painted keyframes should be tracked
    
    def update_mask_from_shapes_liquify(self, shapes):
        """Update the mask from specific shapes (used for liquify preview)"""
        if self.mask is None:
            return
            
        # Clear the mask
        self.mask[:] = 0
        
        # Draw each shape
        for shape in shapes:
            if shape['visible']:
                vertices = np.array(shape['vertices'], dtype=np.int32)
                if len(vertices) >= 3:
                    cv2.fillPoly(self.mask, [vertices], 255)
        
        self.update()
    
    def normalize_all_shape_keyframes(self):
        """Normalize all shape vertices to have consistent starting points"""
        for frame in self.shape_keyframes:
            for shape in self.shape_keyframes[frame]:
                if 'vertices' in shape and len(shape['vertices']) > 0:
                    # Normalize vertices in place
                    shape['vertices'] = self.resample_vertices(shape['vertices'], len(shape['vertices']))
        # Clear interpolation cache after normalization
        self.invalidate_shape_cache()
    
    def apply_shape_eraser_with_fill(self):
        """Apply shape eraser with filled interior when shape is closed"""
        if not hasattr(self, 'temp_shape_mask'):
            return
            
        try:
            # First, check if the stroke forms a closed shape
            if len(self.current_stroke_points) > 10:  # Need enough points
                first_point = np.array(self.current_stroke_points[0])
                last_point = np.array(self.current_stroke_points[-1])
                distance = np.linalg.norm(last_point - first_point)
                
                # If start and end are close, fill the interior
                if distance < self.brush_size * 2:
                    # Create a polygon from the stroke points
                    points = np.array(self.current_stroke_points, dtype=np.int32)
                    cv2.fillPoly(self.temp_shape_mask, [points], 255)
            
            # Apply morphological operations to ensure smooth boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            filled_mask = cv2.morphologyEx(self.temp_shape_mask, cv2.MORPH_CLOSE, kernel)
            
            # Now use this filled mask to erase from shapes
            frame = self.parent_editor.current_frame_index if self.parent_editor else 0
            
            # If no keyframe exists at current frame, create one from interpolated shapes
            if frame not in self.shape_keyframes:
                current_shapes = self.get_shapes_for_frame(frame)
                if current_shapes:
                    
                    # Create keyframe preserving the exact interpolated vertices
                    # DO NOT resample - keep the exact parameterization from interpolation
                    simple_shapes = []
                    for s in current_shapes:
                        # Keep the exact vertices from the interpolated shape
                        simple_shapes.append({
                            'vertices': [list(v) for v in s['vertices']],
                            'vertex_count': len(s['vertices']),  # Keep the current vertex count
                            'filled': s.get('filled', True),
                            'visible': s.get('visible', True),
                            'closed': s.get('closed', True),
                            'is_shape': s.get('is_shape', True)
                        })
                    
                    self.shape_keyframes[frame] = simple_shapes
                    
                    # Invalidate cache since shapes changed
                    self.invalidate_shape_cache()
                    
                    # Update timeline
                    if self.parent_editor:
                        self.parent_editor.update_mask_frame_tracking()
                else:
                    # No shapes to erase from
                    return
                
            # Process each shape
            shapes_to_keep = []
            for shape in self.shape_keyframes[frame]:
                if not shape['visible']:
                    shapes_to_keep.append(shape)
                    continue
                    
                # Create mask for this shape
                shape_mask = np.zeros_like(self.mask)
                vertices = np.array(shape['vertices'], dtype=np.int32)
                cv2.fillPoly(shape_mask, [vertices], 255)
                
                # Preserve the original start point
                preserved_start_point = vertices[0].tolist() if len(vertices) > 0 else None
                
                # Apply boolean subtraction using the filled eraser mask
                result_mask = cv2.bitwise_and(shape_mask, cv2.bitwise_not(filled_mask))
                
                # Check if shape still exists after erasing
                if np.sum(result_mask) < 100:  # Threshold for minimum shape size
                    # Shape is completely erased, don't keep it
                    continue
                
                # Find contours of the result
                contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Sort contours by area
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    # Process each significant contour as a separate shape
                    for idx, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        if area < 50:  # Skip tiny fragments
                            continue
                            
                        # Resample to maintain consistent vertex count using reference signature
                        reference_sig = self.get_sequence_reference_signature()
                        new_vertices = self.resample_contour_fixed_vertices(contour, 
                                                                          num_vertices=self.target_vertex_count,
                                                                          preserve_start_point=preserved_start_point,
                                                                          reference_signature=reference_sig)
                        
                        if len(new_vertices) < 3:
                            continue
                            
                        # Note: winding order is already handled inside resample_contour_fixed_vertices
                        
                        if len(new_vertices) >= 3:
                            if idx == 0:
                                # Update the original shape with the largest fragment
                                shape['vertices'] = new_vertices
                                shapes_to_keep.append(shape)
                            else:
                                # Create new shapes for additional fragments
                                new_shape = {
                                    'vertices': new_vertices,
                                    'vertex_count': self.target_vertex_count,
                                    'filled': shape.get('filled', True),
                                    'visible': True,
                                    'closed': True,
                                    'is_shape': True
                                }
                                shapes_to_keep.append(new_shape)
            
            # Update keyframe with remaining shapes
            self.shape_keyframes[frame] = shapes_to_keep
            
            # Clear interpolation cache
            self.invalidate_shape_cache()
            
            # Normalize all shape keyframes
            self.normalize_all_shape_keyframes()
            
            # Update the mask
            self.update_mask_from_shapes()
            
            # Update timeline
            if self.parent_editor and hasattr(self.parent_editor, 'timeline_widget'):
                if not self.shape_keyframes[frame]:  # No shapes left
                    self.parent_editor.timeline_widget.mask_frames.discard(frame)
                self.parent_editor.timeline_widget.update()
                
        except Exception as e:
            print(f"Error in shape eraser with fill: {e}")
            import traceback
            traceback.print_exc()
    
    def apply_eraser_to_shapes(self):
        """Apply eraser stroke as boolean subtraction from shapes"""
        if len(self.current_stroke_points) < 3:
            return
            
        try:
            frame = self.parent_editor.current_frame_index if self.parent_editor else 0
            
            # If no keyframe exists at current frame, create one from interpolated shapes
            if frame not in self.shape_keyframes:
                current_shapes = self.get_shapes_for_frame(frame)
                if current_shapes:
                    
                    # Get the current vertex count from the editor
                    target_vertex_count = 32  # Default
                    if self.parent_editor and hasattr(self.parent_editor, 'vertex_count_slider'):
                        target_vertex_count = self.parent_editor.vertex_count_slider.value()
                    
                    # Create keyframe with proper vertex count
                    simple_shapes = []
                    for s in current_shapes:
                        vertices = s['vertices']
                        if len(vertices) != target_vertex_count:
                            vertices = self.resample_vertices(vertices, target_vertex_count)
                        
                        simple_shapes.append({
                            'vertices': [list(v) for v in vertices],
                            'vertex_count': target_vertex_count,
                            'filled': s.get('filled', True),
                            'visible': s.get('visible', True),
                            'closed': s.get('closed', True),
                            'is_shape': s.get('is_shape', True)
                        })
                    
                    self.shape_keyframes[frame] = simple_shapes
                    
                    print(f"After: Keyframes = {sorted(self.shape_keyframes.keys())}")
                    
                    # Update timeline
                    if self.parent_editor:
                        self.parent_editor.update_mask_frame_tracking()
                else:
                    # No shapes to erase from
                    return
                
            # Create eraser shape mask
            eraser_mask = np.zeros_like(self.mask)
            
            # Draw the eraser stroke with thickness
            for i in range(1, len(self.current_stroke_points)):
                pt1 = tuple(map(int, self.current_stroke_points[i-1]))
                pt2 = tuple(map(int, self.current_stroke_points[i]))
                cv2.line(eraser_mask, pt1, pt2, 255, self.brush_size * 2)
            
            # Process each shape
            shapes_to_keep = []
            for shape in self.shape_keyframes[frame]:
                if not shape['visible']:
                    shapes_to_keep.append(shape)
                    continue
                    
                # Create mask for this shape
                shape_mask = np.zeros_like(self.mask)
                vertices = np.array(shape['vertices'], dtype=np.int32)
                cv2.fillPoly(shape_mask, [vertices], 255)
                
                # Preserve the original start point
                preserved_start_point = vertices[0].tolist() if len(vertices) > 0 else None
                
                # Apply boolean subtraction
                result_mask = cv2.bitwise_and(shape_mask, cv2.bitwise_not(eraser_mask))
                
                # Check if shape still exists after erasing
                if np.sum(result_mask) < 100:  # Threshold for minimum shape size
                    # Shape is completely erased, don't keep it
                    continue
                
                # Find contours of the result
                contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Sort contours by area
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    # Process each significant contour as a separate shape
                    for idx, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        if area < 50:  # Skip tiny fragments
                            continue
                            
                        # IMPORTANT: Resample to maintain consistent vertex count using reference signature
                        reference_sig = self.get_sequence_reference_signature()
                        new_vertices = self.resample_contour_fixed_vertices(contour, 
                                                                          num_vertices=self.target_vertex_count,
                                                                          preserve_start_point=preserved_start_point,
                                                                          reference_signature=reference_sig)
                        
                        if len(new_vertices) >= 3:
                            # Note: winding order is already handled inside resample_contour_fixed_vertices
                            if idx == 0:
                                # Update the original shape with the largest fragment
                                shape['vertices'] = new_vertices
                                shapes_to_keep.append(shape)
                            else:
                                # Create new shapes for additional fragments
                                new_shape = {
                                    'vertices': new_vertices,
                                    'closed': True,
                                    'visible': True,
                                    'vertex_count': self.target_vertex_count
                                }
                                shapes_to_keep.append(new_shape)
            
            # Update shapes for this frame
            self.shape_keyframes[frame] = shapes_to_keep
            
            # Invalidate cache since shapes changed
            self.invalidate_shape_cache()
            
            # Clear cache and update
            # Shapes will be recalculated
            
            # Don't normalize - it destroys our preserved start points
            # self.normalize_all_shape_keyframes()
            
            self.update_mask_from_shapes()
            
            # Update timeline to show the new or modified keyframe
            if self.parent_editor:
                self.parent_editor.update_mask_frame_tracking()
            
        except Exception as e:
            print(f"Error applying eraser: {e}")
        
    def paintEvent(self, event):
        import time
        paint_start = time.time()
        
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        
        if self.mask is None or self.video_frame is None:
            # Clear display rect if no image
            self.display_rect = QRect()
            return
        
        # If initializing, show a loading message instead of expensive rendering
        if self._initializing:
            painter.setPen(QPen(Qt.white))
            painter.setFont(QFont("Arial", 16))
            painter.drawText(self.rect(), Qt.AlignCenter, "Loading mask editor...")
            return
        
        # Log expensive paint events
        if hasattr(self, 'parent_editor') and self.parent_editor:
            frame = self.parent_editor.current_frame_index if self.parent_editor else 0
            keyframe_count = len(self.shape_keyframes) if hasattr(self, 'shape_keyframes') else 0
            if keyframe_count > 0:
                import logging
                logger = logging.getLogger(__name__)
            
        try:
            # Calculate display rect early for zoom calculations
            img_width = self.video_frame.shape[1]
            img_height = self.video_frame.shape[0]
            scaled_width = max(1, int(img_width * self.zoom_level))
            scaled_height = max(1, int(img_height * self.zoom_level))
            x = (self.width() - scaled_width) // 2 + self.pan_offset.x()
            y = (self.height() - scaled_height) // 2 + self.pan_offset.y()
            self.display_rect = QRect(x, y, scaled_width, scaled_height)
            
            # Create composite image (video with mask overlay)
            composite = self.video_frame.copy()
            
            # Create colored mask overlay
            mask_colored = np.zeros_like(composite)
            mask_colored[:, :, 2] = self.mask  # Red channel for mask
            
            # Blend with video - adjust opacity based on drawing state
            # Only dim when drawing new shape on interpolated frame (not keyframes, not when merging)
            current_frame = self.parent_editor.current_frame_index if self.parent_editor else 0
            is_keyframe = current_frame in self.shape_keyframes
            
            if (self.is_drawing_new_shape and not is_keyframe):
                # Dim interpolated shapes when drawing new shape on interpolated frame
                alpha = 0.08  # Very dim for better video visibility during tracing
            else:
                # Normal opacity for keyframes or when not drawing new shapes
                alpha = 0.3  # Reduced from 0.5 to make mask less intrusive
            composite = cv2.addWeighted(composite, 1 - alpha, mask_colored, alpha, 0)
            
            # Convert to QImage and display
            height, width = composite.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(composite.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale pixmap
            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Draw the image
            painter.drawPixmap(x, y, scaled_pixmap)
            
            # Draw shape painting preview (including shape eraser)
            if self.is_drawing and self.drawing_mode == "shape" and hasattr(self, 'temp_shape_mask') and self.temp_shape_mask is not None:
                # Check if this will merge with existing shapes
                frame = self.parent_editor.current_frame_index if self.parent_editor else 0
                will_merge = False
                
                if frame in self.shape_keyframes:
                    for existing_shape in self.shape_keyframes[frame]:
                        if existing_shape['visible']:
                            # Create mask for existing shape
                            shape_mask = np.zeros_like(self.mask)
                            vertices = np.array(existing_shape['vertices'], dtype=np.int32)
                            cv2.fillPoly(shape_mask, [vertices], 255)
                            
                            # Check overlap
                            overlap = cv2.bitwise_and(self.temp_shape_mask, shape_mask)
                            if np.sum(overlap) > 0:
                                will_merge = True
                                break
                
                # Create overlay - use different color if merging or erasing
                painter.setOpacity(0.5)
                temp_colored = np.zeros_like(composite)
                
                if self.shape_eraser_mode:
                    # Red for erasing
                    temp_colored[:, :, 2] = self.temp_shape_mask  # Red channel only
                elif will_merge:
                    # Cyan for merging
                    temp_colored[:, :, 0] = self.temp_shape_mask  # Blue channel
                    temp_colored[:, :, 1] = self.temp_shape_mask  # Green channel
                else:
                    # Yellow for new shape
                    temp_colored[:, :, 1] = self.temp_shape_mask  # Green channel
                    temp_colored[:, :, 2] = self.temp_shape_mask  # Red channel
                
                # Convert to QImage
                temp_height, temp_width = temp_colored.shape[:2]
                temp_bytes_per_line = 3 * temp_width
                temp_q_image = QImage(temp_colored.data, temp_width, temp_height, 
                                     temp_bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                temp_pixmap = QPixmap.fromImage(temp_q_image)
                
                # Scale and draw
                temp_scaled = temp_pixmap.scaled(scaled_width, scaled_height, 
                                                 Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.drawPixmap(x, y, temp_scaled)
                painter.setOpacity(1.0)
            
            # Draw eraser preview if erasing (only for regular eraser, not shape eraser)
            show_eraser_preview = False
            if self.is_drawing and len(self.current_stroke_points) > 0:
                if self.drawing_mode == "eraser":
                    show_eraser_preview = True
                # Shape eraser is now handled by the shape painting preview above
            
            if show_eraser_preview:
                # Scale brush size with zoom
                brush_display_size = max(1, int(self.brush_size * self.zoom_level))
                
                # Draw the stroke as a thick line with round caps
                if len(self.current_stroke_points) > 1:
                    # Create a path for the eraser stroke
                    path = QPainterPath()
                    
                    # Start from first point
                    first_pt = self.current_stroke_points[0]
                    x = int((first_pt[0] * self.display_rect.width() / self.mask.shape[1]) + self.display_rect.x())
                    y = int((first_pt[1] * self.display_rect.height() / self.mask.shape[0]) + self.display_rect.y())
                    path.moveTo(x, y)
                    
                    # Add all other points
                    for pt in self.current_stroke_points[1:]:
                        x = int((pt[0] * self.display_rect.width() / self.mask.shape[1]) + self.display_rect.x())
                        y = int((pt[1] * self.display_rect.height() / self.mask.shape[0]) + self.display_rect.y())
                        path.lineTo(x, y)
                    
                    # Draw the path with thick red stroke
                    painter.setPen(QPen(QColor(255, 100, 100, 150), brush_display_size * 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path)
                else:
                    # Single point - draw a circle
                    pt = self.current_stroke_points[0]
                    x = int((pt[0] * self.display_rect.width() / self.mask.shape[1]) + self.display_rect.x())
                    y = int((pt[1] * self.display_rect.height() / self.mask.shape[0]) + self.display_rect.y())
                    
                    painter.setPen(QPen(QColor(255, 100, 100, 150), 2))
                    painter.setBrush(QBrush(QColor(255, 100, 100, 100)))
                    painter.drawEllipse(QPoint(x, y), brush_display_size, brush_display_size)
            
            # Draw liquify lattice if enabled and in liquify mode
            if self.drawing_mode == "liquify" and self.show_lattice and self.mask is not None:
                # Draw deformation lattice
                h, w = self.mask.shape[:2]
                grid_size = self.liquify_grid_size
                
                # Set lattice style
                painter.setPen(QPen(QColor(100, 100, 255, 100), 1))  # Light blue, semi-transparent
                
                # Draw vertical lines
                for x in range(0, w + 1, grid_size):
                    # Apply deformation if available
                    if self.liquify_deformation_field is not None:
                        # Draw deformed line
                        path = QPainterPath()
                        first_point = True
                        for y in range(0, h + 1, 5):  # Sample every 5 pixels for smooth curve
                            y_clamped = min(y, h - 1)
                            x_clamped = min(x, w - 1)
                            dx = self.liquify_deformation_field[y_clamped, x_clamped, 0]
                            dy = self.liquify_deformation_field[y_clamped, x_clamped, 1]
                            
                            # Convert to widget coordinates
                            deformed_x = x + dx
                            deformed_y = y + dy
                            widget_x = int((deformed_x * self.display_rect.width() / w) + self.display_rect.x())
                            widget_y = int((deformed_y * self.display_rect.height() / h) + self.display_rect.y())
                            
                            if first_point:
                                path.moveTo(widget_x, widget_y)
                                first_point = False
                            else:
                                path.lineTo(widget_x, widget_y)
                        painter.drawPath(path)
                    else:
                        # Draw straight line
                        x1 = int((x * self.display_rect.width() / w) + self.display_rect.x())
                        y1 = self.display_rect.y()
                        y2 = self.display_rect.y() + self.display_rect.height()
                        painter.drawLine(x1, y1, x1, y2)
                
                # Draw horizontal lines
                for y in range(0, h + 1, grid_size):
                    # Apply deformation if available
                    if self.liquify_deformation_field is not None:
                        # Draw deformed line
                        path = QPainterPath()
                        first_point = True
                        for x in range(0, w + 1, 5):  # Sample every 5 pixels for smooth curve
                            y_clamped = min(y, h - 1)
                            x_clamped = min(x, w - 1)
                            dx = self.liquify_deformation_field[y_clamped, x_clamped, 0]
                            dy = self.liquify_deformation_field[y_clamped, x_clamped, 1]
                            
                            # Convert to widget coordinates
                            deformed_x = x + dx
                            deformed_y = y + dy
                            widget_x = int((deformed_x * self.display_rect.width() / w) + self.display_rect.x())
                            widget_y = int((deformed_y * self.display_rect.height() / h) + self.display_rect.y())
                            
                            if first_point:
                                path.moveTo(widget_x, widget_y)
                                first_point = False
                            else:
                                path.lineTo(widget_x, widget_y)
                        painter.drawPath(path)
                    else:
                        # Draw straight line
                        y1 = int((y * self.display_rect.height() / h) + self.display_rect.y())
                        x1 = self.display_rect.x()
                        x2 = self.display_rect.x() + self.display_rect.width()
                        painter.drawLine(x1, y1, x2, y1)
            
            # Draw shape vertices if in shape mode
            if self.drawing_mode == "shape":
                frame = self.parent_editor.current_frame_index if self.parent_editor else 0
                shapes = self.get_shapes_for_frame(frame)
                
                # If in vertex preview mode, resample shapes for display
                if hasattr(self, '_vertex_preview_mode') and self._vertex_preview_mode and hasattr(self, '_preview_vertex_count'):
                    preview_shapes = []
                    for shape in shapes:
                        if shape['visible'] and 'vertices' in shape and len(shape['vertices']) > 0:
                            preview_shape = shape.copy()
                            preview_shape['vertices'] = self.resample_vertices(shape['vertices'], self._preview_vertex_count)
                            preview_shapes.append(preview_shape)
                        else:
                            preview_shapes.append(shape)
                    shapes = preview_shapes
                
                # Debug visualization - show face orientation
                if self.show_shape_debug:
                    for shape_idx, shape in enumerate(shapes):
                        if shape['visible'] and 'vertices' in shape and shape['vertices'] is not None and len(shape['vertices']) >= 3:
                            # Convert vertices to widget coordinates
                            widget_vertices = []
                            for vertex in shape['vertices']:
                                x = int((vertex[0] * self.display_rect.width() / self.mask.shape[1]) + self.display_rect.x())
                                y = int((vertex[1] * self.display_rect.height() / self.mask.shape[0]) + self.display_rect.y())
                                widget_vertices.append(QPoint(x, y))
                            
                            # Calculate winding order to determine face
                            signed_area = 0
                            n = len(shape['vertices'])
                            for i in range(n):
                                j = (i + 1) % n
                                signed_area += (shape['vertices'][j][0] - shape['vertices'][i][0]) * (shape['vertices'][j][1] + shape['vertices'][i][1])
                            
                            # Draw filled polygon with face color
                            polygon = QPolygon(widget_vertices)
                            if signed_area > 0:  # Counter-clockwise = front face
                                # Red with transparency for front face
                                painter.setBrush(QBrush(QColor(255, 0, 0, 50)))
                            else:  # Clockwise = back face
                                # Blue with transparency for back face
                                painter.setBrush(QBrush(QColor(0, 0, 255, 50)))
                            
                            # Draw shape with smooth curves if enabled
                            if self.should_use_smooth_shapes():
                                self.draw_shape_smooth(painter, shape['vertices'], shape.get('closed', True))
                            else:
                                painter.setPen(Qt.NoPen)
                                painter.drawPolygon(polygon)
                                
                                # Draw shape outline
                                painter.setPen(QPen(QColor(200, 200, 200), 1))
                                painter.setBrush(Qt.NoBrush)
                                painter.drawPolygon(polygon)
                
                # Helper function to calculate adaptive vertex sizes
                def calculate_vertex_display_size(vertex_count, base_size):
                    """Calculate adaptive vertex display size based on vertex count"""
                    if vertex_count <= 25:
                        return base_size  # Full size for low vertex counts
                    elif vertex_count <= 100:
                        # Smooth scaling from full size to ~50% for 100 vertices
                        scale_factor = 1.0 - (vertex_count - 25) * 0.5 / 75
                        return max(2, int(base_size * scale_factor))
                    else:
                        # Minimum size for high vertex counts (200+ vertices)
                        return max(2, int(base_size * 0.3))
                
                for shape_idx, shape in enumerate(shapes):
                    if shape['visible'] and 'vertices' in shape and shape['vertices'] is not None:
                        # Calculate adaptive sizes for this shape's vertex count
                        vertex_count = len(shape['vertices'])
                        
                        # Draw vertices with special colors for start/end
                        for vertex_idx, vertex in enumerate(shape['vertices']):
                            # Convert to widget coordinates
                            x = int((vertex[0] * self.display_rect.width() / self.mask.shape[1]) + self.display_rect.x())
                            y = int((vertex[1] * self.display_rect.height() / self.mask.shape[0]) + self.display_rect.y())
                            
                            # Determine vertex opacity based on drawing state
                            # Only dim when drawing new shape on interpolated frame (not keyframes, not when merging)
                            current_frame = self.parent_editor.current_frame_index if self.parent_editor else 0
                            is_keyframe = current_frame in self.shape_keyframes
                            vertex_alpha = 64 if (self.is_drawing_new_shape and not is_keyframe) else 255
                            
                            # Color code vertices with adaptive sizing
                            if vertex_idx == 0:
                                # Start vertex - green (largest)
                                pen_color = QColor(Qt.green)
                                brush_color = QColor(Qt.green)
                                pen_color.setAlpha(vertex_alpha)
                                brush_color.setAlpha(vertex_alpha)
                                painter.setPen(QPen(pen_color, 3))
                                painter.setBrush(QBrush(brush_color))
                                size = calculate_vertex_display_size(vertex_count, 8)  # Larger for start
                            elif vertex_idx == len(shape['vertices']) - 1:
                                # End vertex - red (medium)
                                pen_color = QColor(Qt.red)
                                brush_color = QColor(Qt.red)
                                pen_color.setAlpha(vertex_alpha)
                                brush_color.setAlpha(vertex_alpha)
                                painter.setPen(QPen(pen_color, 3))
                                painter.setBrush(QBrush(brush_color))
                                size = calculate_vertex_display_size(vertex_count, 6)  # Medium for end
                            elif shape_idx == self.selected_shape_index and vertex_idx == self.selected_vertex_index:
                                # Selected vertex - yellow (medium-large)
                                pen_color = QColor(Qt.yellow)
                                brush_color = QColor(Qt.yellow)
                                pen_color.setAlpha(vertex_alpha)
                                brush_color.setAlpha(vertex_alpha)
                                painter.setPen(QPen(pen_color, 3))
                                painter.setBrush(QBrush(brush_color))
                                size = calculate_vertex_display_size(vertex_count, 7)
                            else:
                                # Regular vertex - cyan (smallest)
                                pen_color = QColor(Qt.cyan)
                                brush_color = QColor(Qt.cyan)
                                pen_color.setAlpha(vertex_alpha)
                                brush_color.setAlpha(vertex_alpha)
                                painter.setPen(QPen(pen_color, 2))
                                painter.setBrush(QBrush(brush_color))
                                size = calculate_vertex_display_size(vertex_count, 5)
                            
                            painter.drawEllipse(x - size, y - size, size * 2, size * 2)
                            
                            # Draw vertex number for first few vertices
                            if self.show_shape_debug and vertex_idx < 5:
                                text_color = QColor(Qt.white)
                                text_color.setAlpha(vertex_alpha)  # Apply same dimming to text
                                painter.setPen(QPen(text_color, 1))
                                painter.setFont(QFont("Arial", 8))
                                painter.drawText(x + size + 2, y - size, str(vertex_idx))
            
            # Draw temporary interpolated shape indicator
            if (hasattr(self, '_temp_interpolated_shapes') and 
                self.parent_editor and 
                self.parent_editor.current_frame_index in self._temp_interpolated_shapes):
                # Draw warning text in top-left corner
                painter.setPen(QPen(Qt.yellow, 2))
                painter.setFont(QFont("Arial", 12, QFont.Bold))
                warning_text = "Temporary Shape - Click 'Apply to Current' to save"
                text_rect = painter.fontMetrics().boundingRect(warning_text)
                bg_rect = QRect(10, 10, text_rect.width() + 20, text_rect.height() + 10)
                
                # Draw background
                painter.fillRect(bg_rect, QColor(0, 0, 0, 180))
                
                # Draw text
                painter.drawText(bg_rect, Qt.AlignCenter, warning_text)
            
            # Draw brush cursor
            if self.underMouse() and (self.current_tool == "brush" or self.current_tool == "liquify"):
                cursor_pos = self.mapFromGlobal(QCursor.pos())
                
                if self.drawing_mode == "liquify":
                    # Show liquify cursor with purple outline (or cyan for smooth mode)
                    if self.shift_pressed or self.relax_mode:
                        # Smooth/relax mode - cyan color
                        painter.setPen(QPen(QColor(0, 255, 255), 2))
                        # Draw with dashed line to indicate smooth mode
                        pen = painter.pen()
                        pen.setStyle(Qt.DashLine)
                        painter.setPen(pen)
                    else:
                        # Regular liquify - purple color
                        painter.setPen(QPen(QColor(148, 0, 211), 2))  # Purple color matching the mode indicator
                    painter.setBrush(Qt.NoBrush)
                    # Scale brush size with zoom
                    brush_display_size = max(1, int(self.brush_size * self.zoom_level))
                    painter.drawEllipse(cursor_pos, brush_display_size, brush_display_size)
                elif self.drawing_mode == "eraser" or (self.drawing_mode == "shape" and self.shape_eraser_mode):
                    # Show eraser cursor with red outline
                    painter.setPen(QPen(Qt.red, 2))
                    painter.setBrush(Qt.NoBrush)
                    # Scale brush size with zoom
                    brush_display_size = max(1, int(self.brush_size * self.zoom_level))
                    painter.drawEllipse(cursor_pos, brush_display_size, brush_display_size)
                elif self.drawing_mode != "shape" or not self.shape_eraser_mode:
                    # Regular brush cursor (including shape brush)
                    # Check if we're in shape mode and on an interpolated frame
                    if self.drawing_mode == "shape" and self.parent_editor:
                        frame = self.parent_editor.current_frame_index
                        is_interpolated = frame not in self.shape_keyframes
                        modifiers = QApplication.keyboardModifiers()
                        
                        if is_interpolated and (modifiers & Qt.ShiftModifier):
                            # Show merge mode cursor - cyan color
                            painter.setPen(QPen(Qt.cyan, 3))
                            painter.setBrush(Qt.NoBrush)
                            # Draw inner circle to make it more visible
                            brush_display_size = max(1, int(self.brush_size * self.zoom_level))
                            painter.drawEllipse(cursor_pos, brush_display_size, brush_display_size)
                            # Draw text indicator
                            painter.setPen(QPen(Qt.cyan, 2))
                            painter.setFont(QFont("Arial", 10, QFont.Bold))
                            painter.drawText(cursor_pos.x() + brush_display_size + 5, 
                                           cursor_pos.y() - 5, "MERGE")
                        elif is_interpolated:
                            # Show new shape cursor - yellow color
                            painter.setPen(QPen(Qt.yellow, 3))
                            painter.setBrush(Qt.NoBrush)
                            brush_display_size = max(1, int(self.brush_size * self.zoom_level))
                            painter.drawEllipse(cursor_pos, brush_display_size, brush_display_size)
                            # Draw text indicator
                            painter.setPen(QPen(Qt.yellow, 2))
                            painter.setFont(QFont("Arial", 10, QFont.Bold))
                            painter.drawText(cursor_pos.x() + brush_display_size + 5, 
                                           cursor_pos.y() - 5, "NEW")
                        else:
                            # Regular white cursor for keyframes
                            painter.setPen(QPen(Qt.white, 2))
                            painter.setBrush(Qt.NoBrush)
                            brush_display_size = max(1, int(self.brush_size * self.zoom_level))
                            painter.drawEllipse(cursor_pos, brush_display_size, brush_display_size)
                    else:
                        # Non-shape modes - regular white cursor
                        painter.setPen(QPen(Qt.white, 2))
                        painter.setBrush(Qt.NoBrush)
                        brush_display_size = max(1, int(self.brush_size * self.zoom_level))
                        painter.drawEllipse(cursor_pos, brush_display_size, brush_display_size)
            
            # Draw keyboard shortcuts overlay in top-right
            self.draw_shortcuts_overlay(painter)
            
            # Log paint time if it was expensive
            paint_time = time.time() - paint_start
            if paint_time > 0.1:  # Log if paint took more than 100ms
                import logging
                logger = logging.getLogger(__name__)
            
        except Exception as e:
            print(f"Error in paintEvent: {e}")
            import traceback
            traceback.print_exc()
            # Don't fill with gray - just try to show what we can
            painter.setPen(Qt.white)
            painter.drawText(10, 30, f"Paint error: {str(e)}")
    
    def should_use_smooth_shapes(self):
        """Check if smooth shape rendering is enabled"""
        return (hasattr(self.parent_editor, 'spline_shapes_check') and 
                self.parent_editor.spline_shapes_check.isChecked())
    
    def draw_shape_smooth(self, painter, vertices, closed=True):
        """Draw shape with Catmull-Rom spline curves between vertices"""
        if len(vertices) < 3:
            # Fall back to linear drawing for shapes with too few vertices
            self.draw_shape_linear(painter, vertices, closed)
            return
        
        # Create smooth spline path through vertices
        path = QPainterPath()
        
        # Convert vertices to widget coordinates
        widget_vertices = []
        for vertex in vertices:
            x = int((vertex[0] * self.display_rect.width() / self.mask.shape[1]) + self.display_rect.x())
            y = int((vertex[1] * self.display_rect.height() / self.mask.shape[0]) + self.display_rect.y())
            widget_vertices.append([x, y])
        
        if closed and len(widget_vertices) >= 3:
            # Closed shape: create smooth curves between all vertices
            first_point = True
            
            for i in range(len(widget_vertices)):
                # Get the 4 control points for this segment
                p0 = widget_vertices[(i-1) % len(widget_vertices)]
                p1 = widget_vertices[i]
                p2 = widget_vertices[(i+1) % len(widget_vertices)]
                p3 = widget_vertices[(i+2) % len(widget_vertices)]
                
                # Draw spline segment from p1 to p2
                self.add_spline_segment_to_path(path, p0, p1, p2, p3, first_point)
                first_point = False
            
            # Close the path
            path.closeSubpath()
        else:
            # Open shape: draw spline through vertices with extrapolated endpoints
            if len(widget_vertices) >= 2:
                # Start from first vertex
                path.moveTo(widget_vertices[0][0], widget_vertices[0][1])
                
                for i in range(len(widget_vertices) - 1):
                    # Get control points for this segment
                    if i == 0:
                        # First segment: extrapolate P0
                        p0 = [widget_vertices[0][0] + (widget_vertices[0][0] - widget_vertices[1][0]),
                              widget_vertices[0][1] + (widget_vertices[0][1] - widget_vertices[1][1])]
                    else:
                        p0 = widget_vertices[i-1]
                    
                    p1 = widget_vertices[i]
                    p2 = widget_vertices[i+1]
                    
                    if i == len(widget_vertices) - 2:
                        # Last segment: extrapolate P3
                        p3 = [widget_vertices[i+1][0] + (widget_vertices[i+1][0] - widget_vertices[i][0]),
                              widget_vertices[i+1][1] + (widget_vertices[i+1][1] - widget_vertices[i][1])]
                    else:
                        p3 = widget_vertices[i+2] if i+2 < len(widget_vertices) else widget_vertices[i+1]
                    
                    # Add spline segment to path
                    self.add_spline_segment_to_path(path, p0, p1, p2, p3, i == 0)
        
        # Draw the smooth path
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)
        
        # Draw outline
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
    
    def draw_shape_linear(self, painter, vertices, closed=True):
        """Draw shape with straight lines (fallback method)"""
        if len(vertices) < 2:
            return
            
        # Convert to widget coordinates and create polygon
        polygon_points = []
        for vertex in vertices:
            x = int((vertex[0] * self.display_rect.width() / self.mask.shape[1]) + self.display_rect.x())
            y = int((vertex[1] * self.display_rect.height() / self.mask.shape[0]) + self.display_rect.y())
            polygon_points.append(QPoint(x, y))
        
        if closed and len(polygon_points) >= 3:
            from PyQt5.QtGui import QPolygon
            polygon = QPolygon(polygon_points)
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(polygon)
            
            # Draw outline
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(polygon)
        else:
            # Open shape: draw lines
            painter.setPen(QPen(QColor(200, 200, 200), 2))
            for i in range(len(polygon_points) - 1):
                painter.drawLine(polygon_points[i], polygon_points[i+1])
    
    def add_spline_segment_to_path(self, path, p0, p1, p2, p3, is_first=False, segments=20):
        """Add smooth Catmull-Rom curve segment to painter path"""
        if is_first:
            path.moveTo(p1[0], p1[1])
        
        # Generate smooth curve points
        for i in range(1, segments + 1):
            t = i / segments
            point = self.catmull_rom_point(p0, p1, p2, p3, t)
            path.lineTo(point[0], point[1])
    
    def generate_smooth_vertices_for_mask(self, vertices, closed=True, segments_per_curve=20):
        """Generate smooth spline vertices for mask creation"""
        if len(vertices) < 3:
            return vertices
        
        smooth_points = []
        
        if closed:
            # Closed shape: generate smooth curves between all vertices
            for i in range(len(vertices)):
                # Get the 4 control points for this segment
                p0 = vertices[(i-1) % len(vertices)]
                p1 = vertices[i]
                p2 = vertices[(i+1) % len(vertices)]
                p3 = vertices[(i+2) % len(vertices)]
                
                # Generate curve points from p1 to p2
                for j in range(segments_per_curve):
                    t = j / segments_per_curve
                    if j == 0 and i > 0:
                        continue  # Skip first point of subsequent segments to avoid duplication
                    point = self.catmull_rom_point(p0, p1, p2, p3, t)
                    smooth_points.append(point)
        else:
            # Open shape: add first vertex
            smooth_points.append(vertices[0])
            
            # Generate smooth curves between vertices
            for i in range(len(vertices) - 1):
                # Get control points for this segment
                if i == 0:
                    # First segment: extrapolate P0
                    p0 = [vertices[0][0] + (vertices[0][0] - vertices[1][0]),
                          vertices[0][1] + (vertices[0][1] - vertices[1][1])]
                else:
                    p0 = vertices[i-1]
                
                p1 = vertices[i]
                p2 = vertices[i+1]
                
                if i == len(vertices) - 2:
                    # Last segment: extrapolate P3
                    p3 = [vertices[i+1][0] + (vertices[i+1][0] - vertices[i][0]),
                          vertices[i+1][1] + (vertices[i+1][1] - vertices[i][1])]
                else:
                    p3 = vertices[i+2] if i+2 < len(vertices) else vertices[i+1]
                
                # Generate curve points (skip first point to avoid duplication)
                for j in range(1, segments_per_curve + 1):
                    t = j / segments_per_curve
                    point = self.catmull_rom_point(p0, p1, p2, p3, t)
                    smooth_points.append(point)
        
        return smooth_points
    
    def draw_shortcuts_overlay(self, painter):
        """Draw keyboard shortcuts overlay in top-right corner"""
        # Save painter state to avoid any interference from previous drawing
        painter.save()
        
        # Reset pen and brush to defaults
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.NoBrush)
        
        # Define shortcuts based on current mode
        shortcuts = [
            ("B", "Brush Tool"),
            ("Shift+B", "Toggle Pixel/Shape"),
            ("E", "Eraser (Hold)"),
            ("W", "Liquify (Hold)"),
            ("Z", "Zoom (Hold)"),
            ("K", "Timeline Scrub (Hold)"),
            ("A", "Apply to Current"),
            ("Ctrl+S", "Save Project"),
            ("Ctrl+O", "Load Project"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Shift+Z", "Redo"),
            ("Alt+Drag", "Timeline Scrub"),
        ]
        
        # Add mode-specific Alt+Arrow description
        if self.drawing_mode in ["shape", "liquify"]:
            shortcuts.append(("Alt+←/→", "Prev/Next Keyframe"))
        else:
            shortcuts.append(("Alt+←/→", "Prev/Next Mask"))
        
        shortcuts.extend([
            ("Space+Drag", "Pan"),
            ("0", "Reset Zoom"),
        ])
        
        # Add mode-specific shortcuts
        if self.drawing_mode == "shape":
            shortcuts.append(("Ctrl", "Shape Eraser (Hold)"))
            shortcuts.append(("Shift", "Relax Vertices"))
            shortcuts.append(("Shift+Click", "Merge Shape (Interp)"))
            shortcuts.append(("D", "Toggle Debug View"))
        
        # Set up overlay styling
        overlay_width = 240  # Increased from 220 for even more space
        line_height = 18
        padding = 10
        overlay_height = len(shortcuts) * line_height + padding * 2
        
        # Position in top-right
        x = self.width() - overlay_width - 20
        y = 20
        
        # Draw semi-transparent background
        painter.fillRect(x, y, overlay_width, overlay_height, QColor(0, 0, 0, 180))
        
        # Draw title
        painter.setPen(QPen(Qt.white, 1))
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.drawText(x + padding, y + padding + 12, "Keyboard Shortcuts")
        
        # Draw shortcuts
        font = QFont("Arial", 9)
        painter.setFont(font)
        y_offset = y + padding + 30  # Increased from 25 to give more space after title
        
        for key, description in shortcuts:
            # Draw key
            painter.setPen(QPen(QColor(100, 200, 255), 1))
            painter.drawText(x + padding, y_offset, key)
            
            # Draw description
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            key_width = 100  # Increased from 85 for even more spacing
            painter.drawText(x + padding + key_width, y_offset, description)
            
            y_offset += line_height
        
        # Restore painter state
        painter.restore()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for tools"""
        try:
            if event.key() == Qt.Key_B:
                if event.modifiers() & Qt.ShiftModifier:
                    # Shift+B - Toggle between brush modes (pixel <-> shape)
                    if self.parent_editor:
                        self.parent_editor.toggle_brush_mode()
                    else:
                        # Toggle locally if no parent editor
                        if self._last_brush_mode == "brush":
                            self._last_brush_mode = "shape"
                            self.set_drawing_mode("shape")
                        else:
                            self._last_brush_mode = "brush"
                            self.set_drawing_mode("brush")
                    self.shape_eraser_mode = False  # Reset shape eraser mode
                else:
                    # B - Activate brush tool with current brush mode
                    if not event.isAutoRepeat() and Qt.Key_B in self.temp_tool_keys:
                        # Store current state for temporary tool switching
                        key_state = self.temp_tool_keys[Qt.Key_B]
                        if not key_state['pressed']:
                            key_state['pressed'] = True
                            key_state['stroke_made'] = False
                            key_state['previous_tool'] = self.current_tool
                            key_state['previous_mode'] = self.drawing_mode
                    # Note: Don't clear temp states here as it interferes with temporary tool system
                    
                    self.set_current_tool("brush")
                    # Reset shape eraser mode when switching to brush
                    self.shape_eraser_mode = False
                    if self.parent_editor:
                        self.parent_editor.select_brush_mode()
                    else:
                        self.set_drawing_mode(self._last_brush_mode)
            elif event.key() == Qt.Key_E:
                # Store current state for temporary tool switching
                if not event.isAutoRepeat() and Qt.Key_E in self.temp_tool_keys:
                    key_state = self.temp_tool_keys[Qt.Key_E]
                    if not key_state['pressed']:
                        key_state['pressed'] = True
                        key_state['stroke_made'] = False
                        key_state['previous_tool'] = self.current_tool
                        key_state['previous_mode'] = self.drawing_mode
                # Note: Don't clear temp states here as it interferes with temporary tool system
                
                # Always activate eraser (don't toggle)
                self.set_current_tool("brush")  # Eraser is part of brush tool
                if self.drawing_mode == "shape":
                    # In shape mode, activate shape eraser
                    self.shape_eraser_mode = True
                    self.update()
                else:
                    # In regular mode, activate eraser
                    self.set_drawing_mode("eraser")
                
                # Update parent editor UI
                if self.parent_editor and hasattr(self.parent_editor, 'eraser_btn'):
                    self.parent_editor.eraser_btn.setChecked(True)
            elif event.key() == Qt.Key_Z and not (event.modifiers() & Qt.ControlModifier):
                # Store current state for temporary tool switching
                if not event.isAutoRepeat() and Qt.Key_Z in self.temp_tool_keys:
                    key_state = self.temp_tool_keys[Qt.Key_Z]
                    if not key_state['pressed']:
                        key_state['pressed'] = True
                        key_state['stroke_made'] = False
                        key_state['previous_tool'] = self.current_tool
                        key_state['previous_mode'] = self.drawing_mode
                # Note: Don't clear temp states here as it interferes with temporary tool system
                
                # Activate zoom tool
                self.set_current_tool("zoom")
                if self.parent_editor and hasattr(self.parent_editor, 'zoom_tool_btn'):
                    self.parent_editor.zoom_tool_btn.setChecked(True)
            elif event.key() == Qt.Key_0:
                # Reset zoom to 100%
                self.zoom_level = 1.0
                self.pan_offset = QPoint(0, 0)
                self.update()
            elif event.key() == Qt.Key_Space and not event.isAutoRepeat():
                # Enable panning while space is held
                self.space_pressed = True
                self.setCursor(Qt.OpenHandCursor)
            elif event.key() == Qt.Key_Shift and not event.isAutoRepeat():
                # Enable relax mode while shift is held
                self.relax_mode = True
                self.shift_pressed = True
                # Update display to show different cursor
                if self.drawing_mode == "liquify":
                    self.update()
            elif event.key() == Qt.Key_W:
                # Store current state for temporary tool switching
                if not event.isAutoRepeat() and Qt.Key_W in self.temp_tool_keys:
                    key_state = self.temp_tool_keys[Qt.Key_W]
                    if not key_state['pressed']:
                        key_state['pressed'] = True
                        key_state['stroke_made'] = False
                        key_state['previous_tool'] = self.current_tool
                        key_state['previous_mode'] = self.drawing_mode
                # Note: Don't clear temp states here as it interferes with temporary tool system
                
                # Activate liquify tool - pass to parent editor
                if self.parent_editor:
                    # The parent will handle switching to shape mode if needed
                    self.parent_editor.keyPressEvent(event)
            elif event.key() == Qt.Key_K:
                # Store current state for temporary tool switching
                if not event.isAutoRepeat() and Qt.Key_K in self.temp_tool_keys:
                    key_state = self.temp_tool_keys[Qt.Key_K]
                    if not key_state['pressed']:
                        key_state['pressed'] = True
                        key_state['stroke_made'] = False
                        key_state['previous_tool'] = self.current_tool
                        key_state['previous_mode'] = self.drawing_mode
                # Activate timeline scrub tool
                self.set_current_tool("timeline_scrub")
                if self.parent_editor:
                    # Update UI if needed
                    pass
            elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
                if event.modifiers() & Qt.ShiftModifier:
                    # Ctrl+Shift+Z - Redo
                    if self.redo():
                        print("Redo performed")
                else:
                    # Ctrl+Z - Undo
                    if self.undo():
                        print("Undo performed")
            elif event.key() == Qt.Key_D:
                # Toggle debug visualization for shapes
                self.show_shape_debug = not self.show_shape_debug
                self.update()
                status = "enabled" if self.show_shape_debug else "disabled"
                print(f"Shape debug visualization {status}")
            elif event.key() == Qt.Key_Control and not event.isAutoRepeat():
                # Ctrl key to temporarily activate eraser in shape mode
                if self.drawing_mode == "shape" and self.current_tool == "brush":
                    key_state = self.temp_tool_keys[Qt.Key_Control]
                    if not key_state['pressed']:
                        key_state['pressed'] = True
                        key_state['stroke_made'] = False
                        key_state['previous_tool'] = self.current_tool
                        key_state['previous_mode'] = self.drawing_mode
                        key_state['previous_shape_eraser'] = self.shape_eraser_mode
                        
                        # Activate shape eraser
                        self.shape_eraser_mode = True
                        self.update()
                        
                        # Update parent editor UI if available
                        if self.parent_editor and hasattr(self.parent_editor, 'eraser_btn'):
                            self.parent_editor.eraser_btn.setChecked(True)
            else:
                super().keyPressEvent(event)
        except Exception as e:
            print(f"Error in keyPressEvent: {e}")
            import traceback
            traceback.print_exc()
    
    def keyReleaseEvent(self, event):
        """Handle key release events"""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            # Restore cursor based on current tool
            self.space_pressed = False
            if self.current_tool == "zoom":
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        elif event.key() == Qt.Key_Shift and not event.isAutoRepeat():
            # Disable relax mode
            self.relax_mode = False
            self.shift_pressed = False
            # Update display to show normal cursor
            if self.drawing_mode == "liquify":
                self.update()
        elif event.key() in self.temp_tool_keys and not event.isAutoRepeat():
            # Handle release of temporary tool keys
            key_state = self.temp_tool_keys[event.key()]
            if key_state['pressed']:
                key_state['pressed'] = False
                
                # If a stroke was made, restore the previous tool (temporary use)
                if key_state['stroke_made'] and key_state['previous_tool'] is not None:
                    # If we're leaving liquify mode, ensure changes are baked - EXACTLY like on_frame_changed does it
                    if event.key() == Qt.Key_W:
                        # Check the PARENT EDITOR's drawing mode, not just the widget's
                        if self.parent_editor and self.parent_editor.drawing_mode == "liquify":
                            
                            # First, ensure liquify is set up if it wasn't already
                            if self.liquify_deformation_field is None or self.liquify_original_shapes is None:
                                self.setup_liquify_for_current_frame()
                            
                            # Now check if we have deformations to bake
                            if (self.liquify_deformation_field is not None and 
                                self.liquify_original_shapes is not None):
                                
                                # Check if there are any actual deformations to preserve
                                max_deform = np.max(np.abs(self.liquify_deformation_field))
                                
                                if max_deform > 0.01:  # If there's significant deformation
                                    # Bake the current liquify deformations into the keyframe
                                    self.bake_liquify_deformation()
                                    
                                    # Force update the display
                                    self.update_mask_from_shapes()
                                    
                                    # Update parent editor display
                                    if self.parent_editor:
                                        self.parent_editor.update_display()
                                        self.parent_editor.update_mask_frame_tracking()
                    
                    # Special handling for Ctrl key (shape eraser)
                    if event.key() == Qt.Key_Control:
                        # Restore shape eraser mode state
                        if 'previous_shape_eraser' in key_state:
                            self.shape_eraser_mode = key_state['previous_shape_eraser']
                        else:
                            self.shape_eraser_mode = False
                        self.update()
                        
                        # Update UI
                        if self.parent_editor and hasattr(self.parent_editor, 'eraser_btn'):
                            if not self.shape_eraser_mode:
                                self.parent_editor.eraser_btn.setChecked(False)
                                # Restore brush button if we were in shape mode
                                if self.drawing_mode == "shape":
                                    self.parent_editor.brush_btn.setChecked(True)
                        
                        # Clear the state
                        key_state['stroke_made'] = False
                        key_state['previous_tool'] = None
                        key_state['previous_mode'] = None
                        if 'previous_shape_eraser' in key_state:
                            key_state['previous_shape_eraser'] = False
                        return  # Exit early for Ctrl key
                    
                    # Restore the previous tool
                    self.set_current_tool(key_state['previous_tool'])
                    if key_state['previous_mode'] is not None:
                        self.set_drawing_mode(key_state['previous_mode'])
                        # If restoring to shape mode, make sure shape_eraser_mode is off
                        if key_state['previous_mode'] == "shape":
                            self.shape_eraser_mode = False
                        
                        # IMPORTANT: Also sync the parent editor's UI controls
                        if self.parent_editor:
                            self.parent_editor.set_drawing_mode(key_state['previous_mode'])
                    
                    # Update UI buttons to match restored tool
                    if self.parent_editor:
                        if key_state['previous_tool'] == "brush":
                            if key_state['previous_mode'] in ["brush", "shape"]:
                                self.parent_editor.brush_btn.setChecked(True)
                                # Restore the correct brush mode icon
                                self.parent_editor.update_brush_button_icon()
                            elif key_state['previous_mode'] == "eraser":
                                self.parent_editor.eraser_btn.setChecked(True)
                        elif key_state['previous_tool'] == "zoom":
                            self.parent_editor.zoom_tool_btn.setChecked(True)
                        elif key_state['previous_tool'] == "liquify":
                            self.parent_editor.liquify_btn.setChecked(True)
                        elif event.key() == Qt.Key_Z:
                            self.parent_editor.zoom_tool_btn.setChecked(False)
                else:
                    # No stroke was made - this was a quick tap, treat as permanent tool switch
                    # Special handling for Ctrl key - always restore shape mode
                    if event.key() == Qt.Key_Control:
                        # Restore shape eraser mode state
                        if 'previous_shape_eraser' in key_state:
                            self.shape_eraser_mode = key_state['previous_shape_eraser']
                        else:
                            self.shape_eraser_mode = False
                        self.update()
                        
                        # Update UI
                        if self.parent_editor and hasattr(self.parent_editor, 'eraser_btn'):
                            if not self.shape_eraser_mode:
                                self.parent_editor.eraser_btn.setChecked(False)
                                # Restore brush button if we were in shape mode
                                if self.drawing_mode == "shape":
                                    self.parent_editor.brush_btn.setChecked(True)
                        
                        # Clear the state
                        key_state['stroke_made'] = False
                        key_state['previous_tool'] = None
                        key_state['previous_mode'] = None
                        if 'previous_shape_eraser' in key_state:
                            key_state['previous_shape_eraser'] = False
                        return  # Exit early for Ctrl key
                    # For E key, if no stroke was made, it should behave like clicking eraser button
                    elif event.key() == Qt.Key_E:
                        # This was a quick tap, not temporary use - clear ALL temp states
                        self.clear_temp_tool_states()
                        # Make sure we're properly in eraser mode permanently
                        if self.parent_editor:
                            self.parent_editor.select_eraser_mode()
                        return  # Exit early since we cleared all states
                
                # Clear the state
                key_state['stroke_made'] = False
                key_state['previous_tool'] = None
                key_state['previous_mode'] = None
        else:
            super().keyReleaseEvent(event)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.mask is None or self.video_frame is None:
            return
            
        # Get mouse position relative to widget
        mouse_pos = event.pos()
        
        # Get zoom delta from wheel
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        # Store old zoom
        old_zoom = self.zoom_level
        new_zoom = max(0.1, min(10.0, self.zoom_level * zoom_factor))
        
        if old_zoom != new_zoom:
            # Get the current display rect to find exact image position
            if hasattr(self, 'display_rect') and self.display_rect.isValid():
                # Use the actual display rect from paintEvent
                old_rect = self.display_rect
                
                # Find which point in the image is under the mouse
                if old_rect.contains(mouse_pos):
                    # Calculate relative position within the image (0-1)
                    rel_x = (mouse_pos.x() - old_rect.x()) / float(old_rect.width())
                    rel_y = (mouse_pos.y() - old_rect.y()) / float(old_rect.height())
                    
                    # Apply new zoom
                    self.zoom_level = new_zoom
                    
                    # Calculate new image dimensions
                    img_width = self.video_frame.shape[1]
                    img_height = self.video_frame.shape[0]
                    new_width = int(img_width * self.zoom_level)
                    new_height = int(img_height * self.zoom_level)
                    
                    # Calculate where the image would be centered
                    new_x = (self.width() - new_width) // 2
                    new_y = (self.height() - new_height) // 2
                    
                    # Find where the mouse point would be in the new image
                    new_mouse_x = new_x + rel_x * new_width
                    new_mouse_y = new_y + rel_y * new_height
                    
                    # Adjust pan to keep the mouse point in the same position
                    self.pan_offset.setX(self.pan_offset.x() + int(mouse_pos.x() - new_mouse_x))
                    self.pan_offset.setY(self.pan_offset.y() + int(mouse_pos.y() - new_mouse_y))
                else:
                    # Mouse is outside image, just zoom toward center
                    self.zoom_level = new_zoom
            else:
                # Fallback if display_rect not available
                self.zoom_level = new_zoom
        
        self.update()
        event.accept()
        
    def apply_liquify_deformation(self, pos):
        """Apply liquify deformation at the given position"""
        if self.liquify_deformation_field is None or self.mask is None:
            return
            
        # Get brush position in image coordinates
        img_x, img_y = self.widget_to_image_coords(pos)
        if img_x is None or img_y is None:
            return
            
        # Check if we're in smooth/relax mode (shift pressed)
        if self.shift_pressed or self.relax_mode:
            # Apply smoothing/relaxation to the deformation field
            self.smooth_liquify_deformation(pos)
            return
            
        # Get last position
        last_x, last_y = self.widget_to_image_coords(self.last_point)
        if last_x is None or last_y is None:
            return
            
        # Calculate displacement
        dx = img_x - last_x
        dy = img_y - last_y
        
        # Debug displacement - commented out to reduce lag
        # if abs(dx) > 0.1 or abs(dy) > 0.1:
        #     print(f"DEBUG: Mouse displacement: dx={dx:.2f}, dy={dy:.2f}")
        
        # Apply deformation in a brush radius
        h, w = self.mask.shape[:2]
        brush_radius = self.brush_size
        
        # Create a gaussian falloff for smooth deformation
        y_indices, x_indices = np.ogrid[-brush_radius:brush_radius+1, -brush_radius:brush_radius+1]
        mask = x_indices**2 + y_indices**2 <= brush_radius**2
        
        # Calculate the area to apply deformation
        y_start = max(0, int(img_y) - brush_radius)
        y_end = min(h, int(img_y) + brush_radius + 1)
        x_start = max(0, int(img_x) - brush_radius)
        x_end = min(w, int(img_x) + brush_radius + 1)
        
        # Calculate the mask boundaries
        mask_y_start = max(0, -int(img_y) + brush_radius)
        mask_y_end = mask_y_start + (y_end - y_start)
        mask_x_start = max(0, -int(img_x) + brush_radius)
        mask_x_end = mask_x_start + (x_end - x_start)
        
        # Apply displacement with gaussian falloff
        valid_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        
        # Get the actual dimensions of the region we're updating
        region_height = y_end - y_start
        region_width = x_end - x_start
        
        # Create gaussian weights with the correct shape
        y_coords, x_coords = np.ogrid[0:region_height, 0:region_width]
        # Calculate distance from brush center
        center_y = img_y - y_start
        center_x = img_x - x_start
        dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
        weights = np.exp(-dist_sq / (2 * (brush_radius/2)**2))
        
        # Ensure valid_mask has the same shape as weights
        if valid_mask.shape != weights.shape:
            # Resize valid_mask to match weights shape
            valid_mask = valid_mask[:weights.shape[0], :weights.shape[1]]
        
        weights = weights * valid_mask
        
        # Apply weighted displacement - ensure shapes match
        deform_slice_y = slice(y_start, y_start + weights.shape[0])
        deform_slice_x = slice(x_start, x_start + weights.shape[1])
        
        self.liquify_deformation_field[deform_slice_y, deform_slice_x, 0] += dx * weights * 0.5
        self.liquify_deformation_field[deform_slice_y, deform_slice_x, 1] += dy * weights * 0.5
        
        # Debug deformation field update - commented out to reduce lag
        # if abs(dx) > 0.1 or abs(dy) > 0.1:
        #     max_weight = np.max(weights)
        #     print(f"DEBUG: Applied deformation - max weight: {max_weight:.3f}, affected area: ({x_start},{y_start}) to ({x_end},{y_end})")
        
        # Apply deformation to shapes immediately for preview
        if self.liquify_original_shapes:
            self.apply_deformation_to_shapes()
            # Force an update to show the deformed shapes
            self.update()
    
    def apply_deformation_to_shapes(self):
        """Apply the deformation field to the original shapes for display only"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"[apply_deformation_to_shapes] Called")
        logger.info(f"[apply_deformation_to_shapes] Has liquify_original_shapes: {self.liquify_original_shapes is not None}")
        logger.info(f"[apply_deformation_to_shapes] Has liquify_deformation_field: {self.liquify_deformation_field is not None}")
        
        if not self.liquify_original_shapes or self.liquify_deformation_field is None:
            logger.info(f"[apply_deformation_to_shapes] Returning early - missing data")
            return
            
        frame = self.parent_editor.current_frame_index if self.parent_editor else 0
        deformed_shapes = []
        
        # Log deformation stats
        max_deform = np.max(np.abs(self.liquify_deformation_field))
        logger.info(f"[apply_deformation_to_shapes] Max deformation magnitude: {max_deform}")
        logger.info(f"[apply_deformation_to_shapes] Number of original shapes: {len(self.liquify_original_shapes)}")
        
        # Check if we have any deformation - commented out to reduce lag
        # max_deform = np.max(np.abs(self.liquify_deformation_field))
        # if max_deform > 0.01:  # Only log if there's significant deformation
        #     print(f"DEBUG: Max deformation magnitude: {max_deform}")
        
        for shape_idx, shape in enumerate(self.liquify_original_shapes):
            deformed_shape = shape.copy()
            deformed_vertices = []
            
            total_displacement = 0.0
            for i, vertex in enumerate(shape['vertices']):
                x, y = vertex
                # Ensure coordinates are within bounds
                x = max(0, min(self.liquify_deformation_field.shape[1] - 1, int(x)))
                y = max(0, min(self.liquify_deformation_field.shape[0] - 1, int(y)))
                
                # Get displacement at this point
                dx = self.liquify_deformation_field[y, x, 0]
                dy = self.liquify_deformation_field[y, x, 1]
                
                # Apply displacement
                new_x = x + dx
                new_y = y + dy
                
                # Track total displacement
                total_displacement += abs(dx) + abs(dy)
                
                # Debug first vertex if there's deformation - commented out to reduce lag
                # if i == 0 and (abs(dx) > 0.01 or abs(dy) > 0.01):
                #     print(f"DEBUG: Vertex 0: ({x},{y}) -> ({new_x},{new_y}), displacement: ({dx},{dy})")
                
                deformed_vertices.append([new_x, new_y])
            
            if shape_idx == 0 and total_displacement > 0:
                logger.info(f"[apply_deformation_to_shapes] Shape 0 total displacement: {total_displacement}")
            
            # DO NOT resample vertices - preserve exact parameterization to prevent morphing
            deformed_shape['vertices'] = deformed_vertices
            deformed_shape['visible'] = shape.get('visible', True)
            deformed_shape['is_shape'] = shape.get('is_shape', True)
            deformed_shapes.append(deformed_shape)
        
        # Store deformed shapes in a temporary attribute for display purposes only
        # DO NOT update the actual keyframes!
        self._temp_deformed_shapes = deformed_shapes
        logger.info(f"[apply_deformation_to_shapes] Stored {len(deformed_shapes)} deformed shapes in _temp_deformed_shapes")
        
        # Log some details about the first deformed shape
        if deformed_shapes:
            first_shape = deformed_shapes[0]
            logger.info(f"[apply_deformation_to_shapes] First deformed shape has {len(first_shape.get('vertices', []))} vertices")
            if first_shape.get('vertices'):
                logger.info(f"[apply_deformation_to_shapes] First vertex of first shape: {first_shape['vertices'][0]}")
        
        # Update the mask from these deformed shapes
        self.update_mask_from_shapes_liquify(deformed_shapes)
    
    def smooth_liquify_deformation(self, pos):
        """Smooth/relax the liquify deformation using Gaussian blur on the deformation field"""
        if self.liquify_deformation_field is None or self.mask is None:
            return
        
        # Get brush position in image coordinates
        img_x, img_y = self.widget_to_image_coords(pos)
        if img_x is None or img_y is None:
            return
            
        # Apply smoothing in a brush radius
        h, w = self.mask.shape[:2]
        brush_radius = self.brush_size
        
        # Calculate the area to affect (expand for blur kernel)
        blur_expansion = int(brush_radius * 0.5)
        y_start = max(0, int(img_y) - brush_radius - blur_expansion)
        y_end = min(h, int(img_y) + brush_radius + blur_expansion + 1)
        x_start = max(0, int(img_x) - brush_radius - blur_expansion)
        x_end = min(w, int(img_x) + brush_radius + blur_expansion + 1)
        
        # Get the deformation field in this region
        deform_x = self.liquify_deformation_field[y_start:y_end, x_start:x_end, 0].copy()
        deform_y = self.liquify_deformation_field[y_start:y_end, x_start:x_end, 1].copy()
        
        # Apply Gaussian blur to smooth the deformation field
        # Kernel size based on brush size for appropriate smoothing
        kernel_size = int(brush_radius * 0.6)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        kernel_size = max(3, kernel_size)  # Minimum kernel size of 3
        
        # Apply Gaussian blur
        smoothed_x = cv2.GaussianBlur(deform_x, (kernel_size, kernel_size), 0)
        smoothed_y = cv2.GaussianBlur(deform_y, (kernel_size, kernel_size), 0)
        
        # Calculate brush weights for blending
        region_h, region_w = deform_x.shape
        brush_center_y = img_y - y_start
        brush_center_x = img_x - x_start
        
        # Create distance weight map
        y_coords, x_coords = np.mgrid[0:region_h, 0:region_w]
        dist_sq = (x_coords - brush_center_x)**2 + (y_coords - brush_center_y)**2
        brush_weights = np.exp(-dist_sq / (2 * (brush_radius * 0.7)**2))
        
        # Mask to only affect points within brush radius
        brush_mask = dist_sq <= brush_radius**2
        
        # Blend original and smoothed based on brush weights
        blend_weights = brush_weights * brush_mask
        
        # Apply weighted blend
        blended_x = deform_x * (1 - blend_weights) + smoothed_x * blend_weights
        blended_y = deform_y * (1 - blend_weights) + smoothed_y * blend_weights
        
        # Apply the smoothed deformation back
        self.liquify_deformation_field[y_start:y_end, x_start:x_end, 0] = blended_x
        self.liquify_deformation_field[y_start:y_end, x_start:x_end, 1] = blended_y
        
        # Apply deformation to shapes immediately for preview
        if self.liquify_original_shapes:
            self.apply_deformation_to_shapes()
            # Force an update to show the deformed shapes
            self.update()
    
    def get_liquified_shapes(self):
        """Get the current liquified shapes without resetting"""
        if not self.liquify_original_shapes:
            return []
            
        self.apply_deformation_to_shapes()
        
        # Return the temporarily deformed shapes without modifying keyframes
        return self._temp_deformed_shapes.copy() if hasattr(self, '_temp_deformed_shapes') else []
    
    def save_undo_state(self, description=""):
        """Save current state to undo stack"""
        state = {
            'mask': self.mask.copy() if self.mask is not None else None,
            'shape_keyframes': {},
            'description': description,
            'frame': self.parent_editor.current_frame_index if self.parent_editor else 0
        }
        
        # Deep copy shape keyframes
        for frame, shapes in self.shape_keyframes.items():
            state['shape_keyframes'][frame] = []
            for shape in shapes:
                state['shape_keyframes'][frame].append({
                    'vertices': [list(v) for v in shape['vertices']],
                    'vertex_count': shape.get('vertex_count', len(shape['vertices'])),
                    'filled': shape.get('filled', True),
                    'visible': shape.get('visible', True),
                    'closed': shape.get('closed', True),
                    'is_shape': shape.get('is_shape', True)
                })
        
        # Add to undo stack
        self.undo_stack.append(state)
        
        # Limit undo stack size
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
        
        # Clear redo stack when new action is performed
        self.redo_stack.clear()
        
        # Mark unsaved changes in parent editor
        if self.parent_editor and hasattr(self.parent_editor, 'mark_unsaved_changes'):
            self.parent_editor.mark_unsaved_changes()
    
    def undo(self):
        """Undo last action"""
        if not self.undo_stack:
            return False
            
        # Save current state to redo stack
        current_state = {
            'mask': self.mask.copy() if self.mask is not None else None,
            'shape_keyframes': {},
            'frame': self.parent_editor.current_frame_index if self.parent_editor else 0
        }
        
        # Deep copy current shape keyframes
        for frame, shapes in self.shape_keyframes.items():
            current_state['shape_keyframes'][frame] = []
            for shape in shapes:
                current_state['shape_keyframes'][frame].append({
                    'vertices': [list(v) for v in shape['vertices']],
                    'vertex_count': shape.get('vertex_count', len(shape['vertices'])),
                    'filled': shape.get('filled', True),
                    'visible': shape.get('visible', True),
                    'closed': shape.get('closed', True),
                    'is_shape': shape.get('is_shape', True)
                })
        
        self.redo_stack.append(current_state)
        
        # Restore previous state
        state = self.undo_stack.pop()
        self.restore_state(state)
        return True
    
    def redo(self):
        """Redo previously undone action"""
        if not self.redo_stack:
            return False
            
        # Save current state to undo stack
        self.save_undo_state("Before redo")
        
        # Restore redo state
        state = self.redo_stack.pop()
        self.restore_state(state)
        return True
    
    def restore_state(self, state):
        """Restore a saved state"""
        # Restore mask
        if state['mask'] is not None:
            self.mask = state['mask'].copy()
        
        # Restore shape keyframes
        self.shape_keyframes.clear()
        for frame, shapes in state['shape_keyframes'].items():
            self.shape_keyframes[frame] = []
            for shape in shapes:
                self.shape_keyframes[frame].append({
                    'vertices': [list(v) for v in shape['vertices']],
                    'vertex_count': shape.get('vertex_count', len(shape['vertices'])),
                    'filled': shape.get('filled', True),
                    'visible': shape.get('visible', True),
                    'closed': shape.get('closed', True),
                    'is_shape': shape.get('is_shape', True)
                })
        
        # Clear interpolation cache
        self.invalidate_shape_cache()
        
        # Update display
        if self.drawing_mode == "shape":
            self.update_mask_from_shapes()
        else:
            self.update()
        
        # Update timeline
        if self.parent_editor:
            self.parent_editor.update_mask_frame_tracking()
    
    def setup_liquify_for_current_frame(self):
        """Set up liquify for the current frame - called on first mouse move"""
        if not hasattr(self, 'parent_editor'):
            return
            
        # Initialize deformation field if needed
        if self.liquify_deformation_field is None and self.mask is not None:
            h, w = self.mask.shape[:2]
            self.liquify_deformation_field = np.zeros((h, w, 2), dtype=np.float32)
        
        frame = self.parent_editor.current_frame_index if self.parent_editor else 0
        
        # Check if we need to create a keyframe for this frame
        if frame not in self.shape_keyframes:
            current_shapes = self.get_shapes_for_frame(frame)
            if current_shapes and self._last_liquify_keyframe_frame != frame:
                
                # Get the current vertex count from the editor
                target_vertex_count = 32  # Default
                if self.parent_editor and hasattr(self.parent_editor, 'vertex_count_slider'):
                    target_vertex_count = self.parent_editor.vertex_count_slider.value()
                
                # Create keyframe preserving the exact interpolated vertices
                # DO NOT resample - keep the exact parameterization from interpolation
                simple_shapes = []
                for s in current_shapes:
                    # Keep the exact vertices from the interpolated shape
                    simple_shapes.append({
                        'vertices': [list(v) for v in s['vertices']],
                        'vertex_count': len(s['vertices']),  # Keep the current vertex count
                        'filled': s.get('filled', True),
                        'visible': s.get('visible', True),
                        'closed': s.get('closed', True),
                        'is_shape': s.get('is_shape', True)
                    })
                
                self.shape_keyframes[frame] = simple_shapes
                self._last_liquify_keyframe_frame = frame
                
                print(f"After: Keyframes = {sorted(self.shape_keyframes.keys())}")
                
                # Update timeline
                if self.parent_editor:
                    self.parent_editor.update_mask_frame_tracking()
        
        # Now store original shapes from the keyframe
        if frame in self.shape_keyframes:
            self.liquify_original_shapes = []
            for shape in self.shape_keyframes[frame]:
                self.liquify_original_shapes.append({
                    'vertices': [list(v) for v in shape['vertices']],
                    'vertex_count': shape.get('vertex_count', len(shape['vertices'])),
                    'filled': shape.get('filled', True),
                    'visible': shape.get('visible', True),
                    'closed': shape.get('closed', True),
                    'is_shape': shape.get('is_shape', True)
                })
    
    def bake_liquify_deformation(self, force=False):
        """Bake the liquify deformation and reset the deformation field"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Called")
        logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Has liquify_original_shapes: {self.liquify_original_shapes is not None}")
        logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] liquify_original_shapes: {len(self.liquify_original_shapes) if self.liquify_original_shapes else 0} shapes")
        logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Has liquify_deformation_field: {self.liquify_deformation_field is not None}")
        
        # ------------------------------------------------------------------
        # 1.  Guarantee we have something to bake.
        #     If the user switches tools before the first brush-move, we have
        #     a deformation field but no stored 'original' shapes yet.  In
        #     that case clone the shapes that are currently shown so the
        #     deformation can be baked instead of silently discarded.
        # ------------------------------------------------------------------
        if (not self.liquify_original_shapes or len(self.liquify_original_shapes) == 0) \
                and self.liquify_deformation_field is not None:
            frame_idx = self.parent_editor.current_frame_index if self.parent_editor else 0
            if frame_idx in self.shape_keyframes:
                import copy
                self.liquify_original_shapes = copy.deepcopy(self.shape_keyframes[frame_idx])
                logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Populated liquify_original_shapes from keyframe {frame_idx}")
        
        if self.liquify_deformation_field is None or not self.liquify_original_shapes:
            logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Returning early - no deformation to bake")
            return
        
        logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Proceeding with bake")
        # Save undo state before baking
        self.save_undo_state("Bake liquify")
            
        # Check the deformation field
        max_deform = np.max(np.abs(self.liquify_deformation_field))
        logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Max deformation before applying: {max_deform}")
        
        # Apply the deformation to get the final shapes
        self.apply_deformation_to_shapes()
        frame = self.parent_editor.current_frame_index if self.parent_editor else 0
        
        logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Has _temp_deformed_shapes: {hasattr(self, '_temp_deformed_shapes')}")
        if hasattr(self, '_temp_deformed_shapes'):
            logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] _temp_deformed_shapes is not None: {self._temp_deformed_shapes is not None}")
            if self._temp_deformed_shapes:
                logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Number of deformed shapes: {len(self._temp_deformed_shapes)}")
        
        # Get the deformed shapes and update the keyframe
        if hasattr(self, '_temp_deformed_shapes') and self._temp_deformed_shapes:
            logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Updating keyframe {frame} with {len(self._temp_deformed_shapes)} deformed shapes")
            # Update the keyframe with the deformed shapes
            if frame in self.shape_keyframes:
                old_shape_count = len(self.shape_keyframes[frame])
                self.shape_keyframes[frame] = []
                for shape in self._temp_deformed_shapes:
                    # DO NOT resample vertices - preserve exact parameterization to prevent morphing
                    self.shape_keyframes[frame].append({
                        'vertices': [list(v) for v in shape['vertices']],
                        'vertex_count': shape.get('vertex_count', len(shape['vertices'])),
                        'filled': shape.get('filled', True),
                        'visible': shape.get('visible', True),
                        'closed': shape.get('closed', True),
                        'is_shape': shape.get('is_shape', True)
                    })
                logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] Updated keyframe {frame}: {old_shape_count} -> {len(self.shape_keyframes[frame])} shapes")
        else:
            logger.info(f"[MaskDrawingWidget.bake_liquify_deformation] WARNING: No deformed shapes to bake!")
        
        # Update original shapes to the baked result (only if we have deformed shapes)
        if hasattr(self, '_temp_deformed_shapes') and self._temp_deformed_shapes:
            self.liquify_original_shapes = []
            for shape in self._temp_deformed_shapes:
                self.liquify_original_shapes.append({
                    'vertices': [list(v) for v in shape['vertices']],
                    'vertex_count': shape.get('vertex_count', len(shape['vertices'])),
                    'filled': shape.get('filled', True),
                    'visible': shape.get('visible', True),
                    'closed': shape.get('closed', True),
                    'is_shape': shape.get('is_shape', True)
                })
        
        # Reset the deformation field
        if self.mask is not None:
            h, w = self.mask.shape[:2]
            self.liquify_deformation_field = np.zeros((h, w, 2), dtype=np.float32)
        
        # Clear temporary deformed shapes
        self._temp_deformed_shapes = None
        
        # Invalidate the shape interpolation cache since shapes have changed
        self.invalidate_shape_cache()
        
        # Force update of the mask with the new shapes
        self.update_mask_from_shapes()
        
        # Update the display
        self.update()


