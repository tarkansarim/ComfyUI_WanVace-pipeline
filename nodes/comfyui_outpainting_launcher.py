#!/usr/bin/env python3
"""
ComfyUI Outpainting Editor Launcher
Launcher that initializes the outpainting editor with file operations support
"""

import sys
import os

print("=" * 80)
print("[OUTPAINTING EDITOR LAUNCHER] Starting outpainting editor launcher script")
print(f"[OUTPAINTING EDITOR LAUNCHER] Script path: {os.path.abspath(__file__)}")
print(f"[OUTPAINTING EDITOR LAUNCHER] Python version: {sys.version}")
print(f"[OUTPAINTING EDITOR LAUNCHER] Command line args: {sys.argv}")
print("=" * 80)
sys.stdout.flush()  # Force output to be flushed

# Fix Qt platform plugin issue
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import json
import argparse
import cv2
import numpy as np
import base64
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

print("[OUTPAINTING EDITOR LAUNCHER] Importing PyQt5...")
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, 
                            QHBoxLayout, QMenuBar, QMenu, QAction, QFileDialog, QMessageBox, 
                            QToolBar, QProgressDialog, QDialog)
from PyQt5.QtCore import Qt, QTimer
print("[OUTPAINTING EDITOR LAUNCHER] PyQt5 imported successfully")

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(f"[OUTPAINTING EDITOR LAUNCHER] Added to sys.path: {os.path.dirname(os.path.abspath(__file__))}")

# Import the outpainting editor
print("[OUTPAINTING EDITOR LAUNCHER] Importing OutpaintingEditor from outpainting_editor...")
try:
    # Remove from sys.modules to force fresh import
    if 'outpainting_editor' in sys.modules:
        print("[OUTPAINTING EDITOR LAUNCHER] Removing outpainting_editor from sys.modules for fresh import")
        del sys.modules['outpainting_editor']
    
    # Import the module fresh
    import outpainting_editor
    from outpainting_editor import OutpaintingEditor
    print("[OUTPAINTING EDITOR LAUNCHER] OutpaintingEditor imported successfully")
    print(f"[OUTPAINTING EDITOR LAUNCHER] outpainting_editor module path: {outpainting_editor.__file__}")
    
except Exception as e:
    print(f"[OUTPAINTING EDITOR LAUNCHER] ERROR importing OutpaintingEditor: {e}")
    import traceback
    traceback.print_exc()
    raise

from pathlib import Path


def load_single_image(img_path):
    """Load a single image file - for parallel processing"""
    try:
        img = cv2.imread(img_path)
        if img is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
    except Exception as e:
        print(f"[Loading] Error loading image {img_path}: {e}")
    return None


def load_video_optimized(video_path, progress_dialog=None):
    """Optimized video loading with progress feedback"""
    frames = []
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    
    # Get total frame count for progress
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Loading] Video has {total_frames} frames")
    
    # Pre-allocate list for better performance
    frames = [None] * total_frames
    frame_idx = 0
    
    # Process in batches for better performance
    batch_size = 30
    batch_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        batch_frames.append((frame_idx, frame))
        frame_idx += 1
        
        # Process batch
        if len(batch_frames) >= batch_size or frame_idx >= total_frames:
            for idx, frame in batch_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[idx] = frame_rgb
            
            # Update progress
            if progress_dialog:
                progress_dialog.setValue(frame_idx)
                if progress_dialog.wasCanceled():
                    cap.release()
                    return []
            
            batch_frames = []
    
    cap.release()
    
    # Remove any None values (in case frame count was wrong)
    frames = [f for f in frames if f is not None]
    
    load_time = time.time() - start_time
    print(f"[Loading] Video loaded in {load_time:.2f}s ({len(frames)} frames, {len(frames)/load_time:.1f} fps)")
    
    return frames


def load_image_sequence_optimized(dir_path, progress_dialog=None):
    """Optimized image sequence loading with parallel processing"""
    frames = []
    start_time = time.time()
    
    # Get all valid image files
    image_files = sorted(glob.glob(os.path.join(dir_path, "*.*")))
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    valid_files = [
        img_path for img_path in image_files 
        if Path(img_path).suffix.lower() in valid_extensions
    ]
    
    if not valid_files:
        return frames
    
    print(f"[Loading] Found {len(valid_files)} image files")
    
    # Use parallel processing for faster loading
    max_workers = min(8, len(valid_files))  # Don't use too many threads
    frames = [None] * len(valid_files)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all loading tasks
        future_to_index = {
            executor.submit(load_single_image, img_path): idx 
            for idx, img_path in enumerate(valid_files)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                if result is not None:
                    frames[idx] = result
                completed += 1
                
                # Update progress
                if progress_dialog:
                    progress_dialog.setValue(completed)
                    if progress_dialog.wasCanceled():
                        # Cancel remaining tasks
                        for f in future_to_index:
                            f.cancel()
                        return []
                        
            except Exception as e:
                print(f"[Loading] Error loading image {idx}: {e}")
                completed += 1
    
    # Remove any None values
    frames = [f for f in frames if f is not None]
    
    load_time = time.time() - start_time
    print(f"[Loading] Image sequence loaded in {load_time:.2f}s ({len(frames)} images, {len(frames)/load_time:.1f} imgs/s)")
    
    return frames


def load_video_file_with_source(parent=None):
    """Load video file or image sequence - returns (frames, source_path, source_type)"""
    msg = QMessageBox()
    msg.setWindowTitle("Select Input Type")
    msg.setText("Choose input type:")
    video_btn = msg.addButton("Video File", QMessageBox.AcceptRole)
    images_btn = msg.addButton("Image Sequence", QMessageBox.AcceptRole)
    cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
    msg.exec_()
    
    frames = []
    source_path = None
    source_type = None
    
    if msg.clickedButton() == video_btn:
        # Browse for video file
        video_path, _ = QFileDialog.getOpenFileName(
            parent,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*.*)"
        )
        
        if video_path:
            source_path = video_path
            source_type = "video"
            
            # Create progress dialog
            progress = QProgressDialog("Loading video frames...", "Cancel", 0, 100, parent)
            progress.setWindowTitle("Loading Video")
            progress.setModal(True)
            progress.show()
            
            # Load video frames with optimization
            frames = load_video_optimized(video_path, progress)
            progress.close()
            
            if not frames:
                if not progress.wasCanceled():
                    QMessageBox.warning(parent, "Error", "Failed to load video file")
            else:
                print(f"[ComfyUI Outpainting Editor] Loaded {len(frames)} frames from video")
    
    elif msg.clickedButton() == images_btn:
        # Browse for image directory
        dir_path = QFileDialog.getExistingDirectory(
            parent,
            "Select Image Sequence Directory"
        )
        
        if dir_path:
            source_path = dir_path
            source_type = "image_sequence"
            
            # Quick scan to get image count for progress
            image_files = sorted(glob.glob(os.path.join(dir_path, "*.*")))
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            valid_files = [
                img_path for img_path in image_files 
                if Path(img_path).suffix.lower() in valid_extensions
            ]
            
            if valid_files:
                # Create progress dialog
                progress = QProgressDialog("Loading image sequence...", "Cancel", 0, len(valid_files), parent)
                progress.setWindowTitle("Loading Images")
                progress.setModal(True)
                progress.show()
                
                # Load image sequence with parallel processing
                frames = load_image_sequence_optimized(dir_path, progress)
                progress.close()
                
                if not frames:
                    if not progress.wasCanceled():
                        QMessageBox.warning(parent, "Error", "Failed to load image sequence")
                else:
                    print(f"[ComfyUI Outpainting Editor] Loaded {len(frames)} images from sequence")
            else:
                QMessageBox.warning(parent, "Error", "No valid images found in directory")
    
    return frames, source_path, source_type


class EnhancedOutpaintingEditor(OutpaintingEditor):
    """Enhanced outpainting editor with file operations for ComfyUI"""
    
    def __init__(self, parent=None, video_frames=None, video_path=None, 
                 canvas_settings=None, scale_canvas=False, output_dir=None, session_data=None):
        # Store output directory
        self.output_dir = output_dir
        self.current_frames = video_frames or []
        self.source_path = video_path
        self.source_type = None  # Will be set when loading files
        
        # Initialize parent class
        super().__init__(parent, video_frames, video_path, canvas_settings, scale_canvas)
        
        # Add file operations toolbar
        self.add_file_operations_toolbar()
        
        # Load session data if provided
        if session_data:
            self.load_session_data(session_data)
        
        # Set up auto-save timer for session persistence
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.auto_save_session)
        self.session_timer.start(30000)  # Auto-save every 30 seconds
        print("[EnhancedOutpaintingEditor] Auto-save timer started (30 second interval)")
        
    def add_file_operations_toolbar(self):
        """Add toolbar with file operations"""
        toolbar = QToolBar("File Operations")
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        # Create a widget to hold the toolbar at the top of the dialog
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Load Video/Images button
        load_btn = QPushButton("Load Video/Images")
        load_btn.clicked.connect(self.load_video_files)
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                border: 1px solid #666;
                padding: 8px 16px;
                border-radius: 4px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
                border-color: #777;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """)
        toolbar_layout.addWidget(load_btn)
        
        toolbar_layout.addSpacing(10)
        
        # Save Project button
        save_btn = QPushButton("Save Project")
        save_btn.clicked.connect(self.save_project)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                padding: 6px 12px;
                border-radius: 4px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        toolbar_layout.addWidget(save_btn)
        
        # Load Project button
        load_project_btn = QPushButton("Load Project")
        load_project_btn.clicked.connect(self.load_project)
        load_project_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                padding: 6px 12px;
                border-radius: 4px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        toolbar_layout.addWidget(load_project_btn)
        
        toolbar_layout.addStretch()
        
        # Insert the toolbar at the top of the dialog
        self.layout().insertWidget(0, toolbar_widget)
        
    def load_video_files(self):
        """Load video file or image sequence"""
        frames, source_path, source_type = load_video_file_with_source(self)
        
        if frames:
            self.current_frames = frames
            self.source_path = source_path
            self.source_type = source_type
            self.video_frames = frames
            
            # Update the preview widget
            self.preview_widget.set_frames(frames)
            self.setup_playback_controls()
            
            # Auto-save after loading
            self.auto_save_session()
            
            print(f"[EnhancedOutpaintingEditor] Loaded {len(frames)} frames from {source_type}")
            
    def save_project(self):
        """Save current project to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Outpainting Project",
            "",
            "Outpainting Project (*.opproj);;All Files (*.*)"
        )
        
        if file_path:
            try:
                project_data = self.get_project_data()
                with open(file_path, 'w') as f:
                    json.dump(project_data, f, indent=2)
                
                QMessageBox.information(self, "Success", "Project saved successfully!")
                print(f"[EnhancedOutpaintingEditor] Project saved to: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
                print(f"[EnhancedOutpaintingEditor] Error saving project: {e}")
                
    def load_project(self):
        """Load project from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Outpainting Project",
            "",
            "Outpainting Project (*.opproj);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    project_data = json.load(f)
                
                self.load_session_data(json.dumps(project_data))
                
                QMessageBox.information(self, "Success", "Project loaded successfully!")
                print(f"[EnhancedOutpaintingEditor] Project loaded from: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")
                print(f"[EnhancedOutpaintingEditor] Error loading project: {e}")
                
    def get_project_data(self):
        """Get current project data"""
        canvas_settings = self.get_canvas_settings()
        
        project_data = {
            "canvas_settings": canvas_settings,
            "video_info": {
                "path": self.source_path,
                "type": self.source_type,
                "total_frames": len(self.current_frames) if self.current_frames else 0,
                "width": self.current_frames[0].shape[1] if self.current_frames else 0,
                "height": self.current_frames[0].shape[0] if self.current_frames else 0,
                "fps": self.preview_widget.video_fps if hasattr(self.preview_widget, 'video_fps') else 30
            }
        }
        
        return project_data
        
    def auto_save_session(self):
        """Auto-save session data to temporary location"""
        if self.output_dir:
            try:
                # Create session data
                project_data = self.get_project_data()
                
                # Wrap in the expected format for ComfyUI
                session_data = {
                    "project_data": project_data
                }
                
                # Save to session file
                session_file = os.path.join(self.output_dir, "canvas_data.json")
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                print(f"[EnhancedOutpaintingEditor] Auto-saved session to: {session_file}")
                
            except Exception as e:
                print(f"[EnhancedOutpaintingEditor] Error auto-saving session: {e}")
                
    def load_session_data(self, session_data_str):
        """Load session data from JSON string"""
        try:
            session_data = json.loads(session_data_str)
            
            # Check if we have project_data wrapper
            if "project_data" in session_data:
                project_data = session_data["project_data"]
            else:
                project_data = session_data
            
            # Load video if path is available
            video_info = project_data.get("video_info", {})
            if video_info.get("path") and os.path.exists(video_info["path"]):
                print(f"[EnhancedOutpaintingEditor] Loading video from session: {video_info['path']}")
                
                # Create progress dialog for session loading
                progress = QProgressDialog("Loading video from session...", "Cancel", 0, 100, self)
                progress.setWindowTitle("Loading Session")
                progress.setModal(True)
                progress.show()
                
                frames = []
                if video_info.get("type") == "video":
                    frames = load_video_optimized(video_info["path"], progress)
                elif video_info.get("type") == "image_sequence":
                    frames = load_image_sequence_optimized(video_info["path"], progress)
                
                progress.close()
                
                if frames:
                    self.current_frames = frames
                    self.video_frames = frames
                    self.source_path = video_info["path"]
                    self.source_type = video_info.get("type")
                    self.preview_widget.set_frames(frames)
                    self.setup_playback_controls()
            
            # Apply canvas settings
            canvas_settings = project_data.get("canvas_settings", {})
            if canvas_settings:
                self.apply_canvas_settings(canvas_settings)
                
        except Exception as e:
            print(f"[EnhancedOutpaintingEditor] Error loading session data: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for the launcher"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config JSON file")
    args = parser.parse_args()
    
    print(f"[ComfyUI Outpainting Editor] Starting with config: {args.config_path}")
    
    # Read config
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = config["output_dir"]
    session_data = config.get("session_data")
    
    print(f"[ComfyUI Outpainting Editor] Output directory: {output_dir}")
    print(f"[ComfyUI Outpainting Editor] Has session data: {bool(session_data)}")
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Apply comprehensive dark theme style
    app.setStyle("Fusion")
    app.setStyleSheet("""
        /* Main window and dialogs */
        QMainWindow, QDialog {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        /* Title bar and window frame */
        QMainWindow::title {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        /* All widgets */
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            border: none;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #3a3a3a;
            border: 1px solid #555;
            padding: 6px 12px;
            border-radius: 4px;
            color: #ffffff;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
            border-color: #666;
        }
        QPushButton:pressed {
            background-color: #2a2a2a;
        }
        QPushButton:checked {
            background-color: #404040;
            border-color: #777;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            background: #3a3a3a;
            height: 8px;
            border-radius: 4px;
            margin: 0px;
        }
        QSlider::handle:horizontal {
            background: #ffffff;
            border: 1px solid #555;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #cccccc;
        }
        
        /* Checkboxes and radio buttons */
        QCheckBox, QRadioButton {
            background-color: transparent;
            color: #ffffff;
            spacing: 5px;
        }
        QCheckBox::indicator, QRadioButton::indicator {
            width: 16px;
            height: 16px;
        }
        QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {
            background-color: #2b2b2b;
            border: 1px solid #3a3a3a;
            border-radius: 4px;
        }
        
        /* Tool panels and sidebars - darker background */
        QFrame[frameShape="1"], QFrame[frameShape="2"] {
            background-color: #1e1e1e;
            border: 1px solid #3a3a3a;
        }
        
        /* Tool panel specific styling */
        QFrame#tool_panel {
            background-color: #1e1e1e;
            border-right: 1px solid #3a3a3a;
        }
        
        /* Group boxes in tool panel */
        QGroupBox {
            background-color: #252525;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 16px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px 0 4px;
            color: #cccccc;
        }
        
        /* Labels */
        QLabel {
            background-color: transparent;
            color: #ffffff;
        }
        
        /* Spin boxes and line edits */
        QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #3a3a3a;
            border: 1px solid #555;
            padding: 4px;
            border-radius: 3px;
            color: #ffffff;
        }
        QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
            border-color: #666;
        }
        
        /* Combo boxes */
        QComboBox {
            background-color: #3a3a3a;
            border: 1px solid #555;
            padding: 4px;
            border-radius: 3px;
            color: #ffffff;
        }
        QComboBox:hover {
            border-color: #666;
        }
        QComboBox::drop-down {
            border: none;
            background-color: #4a4a4a;
        }
        
        /* Menu bar and menus */
        QMenuBar {
            background-color: #1e1e1e;
            color: #ffffff;
            border-bottom: 1px solid #3a3a3a;
        }
        QMenuBar::item {
            background-color: transparent;
            padding: 4px 8px;
        }
        QMenuBar::item:selected {
            background-color: #3a3a3a;
        }
        QMenu {
            background-color: #2b2b2b;
            border: 1px solid #3a3a3a;
            color: #ffffff;
        }
        QMenu::item {
            padding: 4px 16px;
        }
        QMenu::item:selected {
            background-color: #3a3a3a;
        }
        
        /* Status bar */
        QStatusBar {
            background-color: #1e1e1e;
            color: #ffffff;
            border-top: 1px solid #3a3a3a;
        }
        
        /* Tool tips */
        QToolTip {
            background-color: #3a3a3a;
            color: #ffffff;
            border: 1px solid #555;
            padding: 4px;
            border-radius: 3px;
        }
    """)
    
    # Create and show editor
    editor = EnhancedOutpaintingEditor(
        output_dir=output_dir,
        session_data=session_data
    )
    
    result = editor.exec_()
    
    if result == QDialog.Accepted:
        print("[ComfyUI Outpainting Editor] User clicked OK")
        # Save the final canvas data
        canvas_settings = editor.get_canvas_settings()
        
        # Create the full data structure
        canvas_data = {
            "project_data": {
                "canvas_settings": canvas_settings,
                "video_info": {
                    "path": editor.source_path,
                    "type": editor.source_type,
                    "total_frames": len(editor.current_frames) if editor.current_frames else 0,
                    "width": editor.current_frames[0].shape[1] if editor.current_frames else 0,
                    "height": editor.current_frames[0].shape[0] if editor.current_frames else 0,
                    "fps": editor.preview_widget.video_fps if hasattr(editor.preview_widget, 'video_fps') else 30
                }
            }
        }
        
        # Save to output directory
        output_path = os.path.join(output_dir, "canvas_data.json")
        with open(output_path, 'w') as f:
            json.dump(canvas_data, f, indent=2)
        
        print(f"[ComfyUI Outpainting Editor] Saved canvas data to: {output_path}")
    else:
        print("[ComfyUI Outpainting Editor] User cancelled")
        
    # Add session save to persist state for reload
    editor.auto_save_session()
    
    print("[ComfyUI Outpainting Editor] Exiting...")
    return 0


if __name__ == "__main__":
    sys.exit(main())