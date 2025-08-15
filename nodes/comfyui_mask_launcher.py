#!/usr/bin/env python3
"""
ComfyUI Mask Editor Launcher
Simplified launcher that properly initializes the mask editor with timeline support
"""

import sys
import os

print("=" * 80)
print("[MASK EDITOR LAUNCHER] Starting mask editor launcher script")
print(f"[MASK EDITOR LAUNCHER] Script path: {os.path.abspath(__file__)}")
print(f"[MASK EDITOR LAUNCHER] Python version: {sys.version}")
print(f"[MASK EDITOR LAUNCHER] Command line args: {sys.argv}")
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

print("[MASK EDITOR LAUNCHER] Importing PyQt5...")
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
print("[MASK EDITOR LAUNCHER] PyQt5 imported successfully")

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(f"[MASK EDITOR LAUNCHER] Added to sys.path: {os.path.dirname(os.path.abspath(__file__))}")

# Import the main application
print("[MASK EDITOR LAUNCHER] Importing InpaintingMaskEditor from mask_editor...")
try:
    # Remove from sys.modules to force fresh import
    import sys
    if 'mask_editor' in sys.modules:
        print("[MASK EDITOR LAUNCHER] Removing mask_editor from sys.modules for fresh import")
        del sys.modules['mask_editor']
    
    # Import the module fresh
    import mask_editor
    from mask_editor import InpaintingMaskEditor
    print("[MASK EDITOR LAUNCHER] InpaintingMaskEditor imported successfully")
    print(f"[MASK EDITOR LAUNCHER] mask_editor module path: {mask_editor.__file__}")
    
    # Check if our logging code is present
    import inspect
    if hasattr(InpaintingMaskEditor, 'set_drawing_mode'):
        source_lines = inspect.getsource(InpaintingMaskEditor.set_drawing_mode).split('\n')
        has_logging = any('logger.info' in line for line in source_lines[:15])
        print(f"[MASK EDITOR LAUNCHER] set_drawing_mode has logging code: {has_logging}")
except Exception as e:
    print(f"[MASK EDITOR LAUNCHER] ERROR importing InpaintingMaskEditor: {e}")
    import traceback
    traceback.print_exc()
    raise

from PyQt5.QtWidgets import QMenuBar, QMenu, QAction, QFileDialog, QMessageBox, QToolBar, QHBoxLayout, QPushButton, QWidget, QProgressDialog
from PyQt5.QtCore import Qt, QTimer
from pathlib import Path

def load_single_image_mask_editor(img_path):
    """Load a single image file - for parallel processing in mask editor"""
    try:
        img = cv2.imread(img_path)
        if img is not None:
            # Keep BGR format for mask editor compatibility
            return img
    except Exception as e:
        print(f"[Mask Editor Loading] Error loading image {img_path}: {e}")
    return None


def load_image_sequence_optimized_mask_editor(dir_path, progress_dialog=None):
    """Optimized image sequence loading for mask editor"""
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
    
    print(f"[Mask Editor Loading] Found {len(valid_files)} image files")
    
    # Use parallel processing for faster loading
    max_workers = min(8, len(valid_files))  # Don't use too many threads
    frames = [None] * len(valid_files)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all loading tasks
        future_to_index = {
            executor.submit(load_single_image_mask_editor, img_path): idx 
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
                        
                # Progress logging
                if completed % 100 == 0 or completed == len(valid_files):
                    print(f"[Mask Editor Loading] Loaded {completed}/{len(valid_files)} images...")
                        
            except Exception as e:
                print(f"[Mask Editor Loading] Error loading image {idx}: {e}")
                completed += 1
    
    # Remove any None values
    frames = [f for f in frames if f is not None]
    
    load_time = time.time() - start_time
    print(f"[Mask Editor Loading] Image sequence loaded in {load_time:.2f}s ({len(frames)} images, {len(frames)/load_time:.1f} imgs/s)")
    
    return frames

def prompt_for_missing_file(source_path, source_type, parent=None):
    """Show dialog when original file is missing and let user browse for new location"""
    import os
    from PyQt5.QtWidgets import QMessageBox, QFileDialog
    
    # Show informative message
    msg = QMessageBox(parent)
    msg.setWindowTitle("File Not Found")
    msg.setIcon(QMessageBox.Warning)
    
    file_type_str = "video file" if source_type == "video" else "image sequence folder"
    msg.setText(f"The original {file_type_str} could not be found:")
    msg.setInformativeText(f"{source_path}\n\nWould you like to browse for the new location?")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg.setDefaultButton(QMessageBox.Yes)
    
    if msg.exec_() == QMessageBox.Yes:
        if source_type == "video":
            # Browse for video file
            new_path, _ = QFileDialog.getOpenFileName(
                parent,
                "Locate Video File",
                os.path.dirname(source_path) if os.path.dirname(source_path) else "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*.*)"
            )
            if new_path and os.path.exists(new_path):
                print(f"[Mask Editor] User selected new video path: {new_path}")
                return new_path
        else:  # image_sequence
            # Browse for folder
            new_path = QFileDialog.getExistingDirectory(
                parent,
                "Locate Image Sequence Folder",
                os.path.dirname(source_path) if os.path.dirname(source_path) else ""
            )
            if new_path and os.path.exists(new_path):
                print(f"[Mask Editor] User selected new image sequence path: {new_path}")
                return new_path
    
    return None

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
            # Load video frames
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Keep frames in BGR format (OpenCV default) - mask editor expects this
                    frames.append(frame)
                cap.release()
                print(f"[Load Video] Loaded {len(frames)} frames from video")
            else:
                QMessageBox.critical(parent, "Error", "Failed to open video file")
                
    elif msg.clickedButton() == images_btn:
        # Browse for image folder
        folder_path = QFileDialog.getExistingDirectory(
            parent,
            "Select Image Sequence Folder",
            ""
        )
        
        if folder_path:
            source_path = folder_path
            source_type = "image_sequence"
            
            # Quick scan to get image count for progress
            image_files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
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
                frames = load_image_sequence_optimized_mask_editor(folder_path, progress)
                progress.close()
                
                if not frames:
                    if not progress.wasCanceled():
                        QMessageBox.warning(parent, "Error", "Failed to load image sequence")
                else:
                    print(f"[Load Video] Loaded {len(frames)} frames from image sequence")
            else:
                QMessageBox.warning(parent, "Warning", "No valid images found in selected folder")
    
    if frames:
        return frames, source_path, source_type
    else:
        return None

class EnhancedMaskEditor(InpaintingMaskEditor):
    """Enhanced mask editor with file operations built-in"""
    
    def __init__(self, frames):
        super().__init__(frames)
        self.current_project_path = None
        self.current_frames = frames
        self.output_dir = None  # Will be set when saving
        self.config_path = None  # Store config path for auto-save
        self.source_video_path = None  # Path to source video/images
        self.source_type = None  # "video" or "image_sequence"
        self.add_file_toolbar()
        
        # Set up logging
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("[EnhancedMaskEditor] Initializing...")
        self.logger.info(f"[EnhancedMaskEditor] Frames passed to init: {len(frames) if frames else 0}")
        self.logger.info(f"[EnhancedMaskEditor] source_video_path: {self.source_video_path}")
        self.logger.info(f"[EnhancedMaskEditor] source_type: {self.source_type}")
        
        # Set up auto-save timer for session persistence
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.auto_save_session)
        self.session_timer.start(30000)  # Auto-save every 30 seconds (reduced from 5)
        self.logger.info("[EnhancedMaskEditor] Auto-save timer started (30 second interval)")
        
        # Hide duplicate save/load buttons from parent class
        self.hide_duplicate_buttons()
        
        # Check if we need to reload video after initialization
        QTimer.singleShot(100, self.check_and_reload_video)
    
    def add_file_toolbar(self):
        """Add toolbar with file operations"""
        # Create a toolbar widget
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        
        # Load Video button
        load_btn = QPushButton("Load Video/Images")
        
        def load_with_debug():
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Load button clicked, preparing for video load...")
            
            # Hide the mask widget BEFORE loading to prevent freeze
            if hasattr(self, 'mask_widget'):
                logger.info("Hiding mask widget before load...")
                self.mask_widget.setVisible(False)
                self.mask_widget.setUpdatesEnabled(False)
            
            # Process events to ensure UI updates
            QApplication.processEvents()
            
            # Store reference to self for use in nested function
            editor_self = self
            
            # Override load_video_files to add progress bar AFTER file selection
            original_load_video_files = self.load_video_files
            
            def load_video_files_with_progress():
                import time  # Add missing import
                logger.info("[WRAPPER] Starting load_video_files_with_progress wrapper...")
                
                # Call the original load_video_files which has keyframe preservation logic
                # The original method returns True/False for the wrapper
                logger.info("[WRAPPER] Calling original load_video_files (which has keyframe preservation)")
                success = original_load_video_files()
                
                if not success:
                    # User cancelled, just return
                    logger.info("[WRAPPER] User cancelled file selection")
                    if hasattr(editor_self, 'mask_widget'):
                        editor_self.mask_widget.setVisible(True)
                        editor_self.mask_widget.setUpdatesEnabled(True)
                    return
                
                logger.info("[WRAPPER] Original load_video_files completed successfully")
                
                # Get the frames that were loaded
                frames = editor_self.current_frames
                if not frames:
                    logger.info("No frames loaded")
                    return
                
                # NOW show the progress bar after file is selected
                total_progress = QProgressDialog("Loading video and initializing UI...", None, 0, 100, editor_self)
                total_progress.setWindowTitle("Loading...")
                total_progress.setWindowModality(Qt.WindowModal)
                total_progress.setMinimumDuration(0)
                total_progress.setCancelButton(None)
                total_progress.setAutoReset(False)
                total_progress.setAutoClose(False)
                total_progress.setWindowFlags(total_progress.windowFlags() | Qt.WindowStaysOnTopHint)
                
                # Store reference to progress dialog so reinitialize_with_frames doesn't close it
                editor_self._progress_dialog = total_progress
                
                # Style it
                total_progress.setStyleSheet("""
                    QProgressDialog {
                        background-color: #2b2b2b;
                        color: #ffffff;
                        border: 1px solid #555;
                        border-radius: 4px;
                        padding: 20px;
                        min-width: 400px;
                    }
                    QProgressBar {
                        background-color: #3a3a3a;
                        border: 1px solid #555;
                        border-radius: 3px;
                        text-align: center;
                        color: #ffffff;
                    }
                    QProgressBar::chunk {
                        background-color: #4CAF50;
                        border-radius: 3px;
                    }
                    QLabel {
                        color: #ffffff;
                        font-size: 14px;
                        padding: 5px;
                    }
                """)
                
                total_progress.show()
                QApplication.processEvents()
                
                # Estimate TOTAL time based on frame count
                frame_count = len(frames)
                total_estimated = 10.0 + (frame_count * 0.05)  # More accurate estimate
                operation_start = time.time()
                
                logger.info(f"Video selected: {frame_count} frames, estimated time: {total_estimated:.1f}s")
                
                # Create a single timer for the entire operation
                progress_timer = QTimer()
                
                def update_total_progress():
                    elapsed = time.time() - operation_start
                    progress_percent = min(95, int((elapsed / total_estimated) * 100))
                    total_progress.setValue(progress_percent)
                    
                    if elapsed < 2:
                        total_progress.setLabelText(f"Processing {frame_count} frames... {elapsed:.1f}s")
                    elif elapsed < 5:
                        total_progress.setLabelText(f"Initializing mask editor... {elapsed:.1f}s")
                    else:
                        total_progress.setLabelText(f"Updating UI (this may take a while)... {elapsed:.1f}s")
                    
                    total_progress.show()
                    total_progress.raise_()
                    QApplication.processEvents()
                    
                    if progress_percent >= 95:
                        progress_timer.stop()
                
                # Start the timer
                progress_timer.timeout.connect(update_total_progress)
                progress_timer.start(100)
                
                # The frames have already been processed and reinitialized by the original load_video_files
                logger.info(f"[WRAPPER] Frames already loaded and reinitialized: {len(frames)} frames")
                
                logger.info("load_video_files completed, now in the FREEZE period...")
                
                # Keep the progress bar open during the freeze
                logger.info("Keeping progress bar open during freeze period...")
                
                # Don't close the progress bar yet - keep it open during the freeze
                def complete_initialization():
                    logger.info("Starting final initialization...")
                    
                    # Show the widget which triggers the freeze
                    if hasattr(editor_self, 'mask_widget'):
                        logger.info("Re-enabling mask widget...")
                        editor_self.mask_widget.setUpdatesEnabled(True)
                        editor_self.mask_widget.setVisible(True)
                        editor_self.mask_widget.update()
                        logger.info("Mask widget restored")
                    
                    # Keep updating progress during the freeze
                    def final_update():
                        elapsed = time.time() - operation_start
                        logger.info(f"Final update - elapsed: {elapsed:.1f}s")
                        progress_timer.stop()
                        total_progress.setValue(100)
                        total_progress.setLabelText("Complete!")
                        
                        # Clear the progress dialog reference
                        if hasattr(editor_self, '_progress_dialog'):
                            delattr(editor_self, '_progress_dialog')
                        
                        QTimer.singleShot(300, total_progress.close)
                    
                    # Short delay to ensure UI is responsive before closing progress
                    QTimer.singleShot(500, final_update)  # Reduced from 10 seconds
                
                # Start the final initialization after a brief delay
                QTimer.singleShot(100, complete_initialization)
            
            # Replace the method temporarily
            self.load_video_files = load_video_files_with_progress
            
            # Call the modified method
            logger.info("Now calling load_video_files...")
            self.load_video_files()
            
            # Restore original method
            self.load_video_files = original_load_video_files
            
        load_btn.clicked.connect(load_with_debug)
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
        
        # Add separator
        separator = QWidget()
        separator.setFixedWidth(10)
        toolbar_layout.addWidget(separator)
        
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
        
        # Save Project As button
        save_as_btn = QPushButton("Save As...")
        save_as_btn.clicked.connect(self.save_project_as)
        save_as_btn.setStyleSheet("""
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
        toolbar_layout.addWidget(save_as_btn)
        
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
        
        toolbar_layout.addStretch()  # Push everything to the left
        
        # Style the toolbar widget itself
        toolbar_widget.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border-bottom: 1px solid #3a3a3a;
                padding: 4px;
            }
        """)
        
        # Insert the toolbar at the top of the dialog
        main_layout = self.layout()
        if main_layout:
            main_layout.insertWidget(0, toolbar_widget)
    
    def load_video_files(self):
        """Load video files or image sequences"""
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        load_start = time.time()
        
        # Debug current state
        logger.info(f"[KEYFRAME DEBUG] === STARTING LOAD_VIDEO_FILES ===")
        logger.info(f"[KEYFRAME DEBUG] Has mask_widget: {hasattr(self, 'mask_widget')}")
        if hasattr(self, 'mask_widget'):
            logger.info(f"[KEYFRAME DEBUG] Has shape_keyframes: {hasattr(self.mask_widget, 'shape_keyframes')}")
            if hasattr(self.mask_widget, 'shape_keyframes'):
                logger.info(f"[KEYFRAME DEBUG] Current keyframe count: {len(self.mask_widget.shape_keyframes)}")
                logger.info(f"[KEYFRAME DEBUG] Keyframe indices: {list(self.mask_widget.shape_keyframes.keys())}")
        
        # Store existing keyframes before loading new video
        existing_keyframes = None
        preserve_keyframes = False
        if hasattr(self, 'mask_widget') and hasattr(self.mask_widget, 'shape_keyframes'):
            if self.mask_widget.shape_keyframes:
                import copy
                existing_keyframes = copy.deepcopy(self.mask_widget.shape_keyframes)
                logger.info(f"[KEYFRAME DEBUG] Stored {len(existing_keyframes)} existing keyframes before loading new video")
                # Verify the copy
                logger.info(f"[KEYFRAME DEBUG] Copied keyframe indices: {list(existing_keyframes.keys())}")
        
        logger.info(f"Starting load_video_files...")
        # Use the global load_video_file_with_source function (returns tuple)
        result = load_video_file_with_source(None)
        
        if result is not None:
            frames, source_path, source_type = result
            self.source_video_path = source_path
            self.source_type = source_type
            logger.info(f"Video loading took {time.time() - load_start:.2f}s")
            self.current_frames = frames
            logger.info(f"Loaded {len(frames)} frames from {source_type}: {source_path}")
            logger.info(f"Frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
            
            # Check if we should preserve keyframes
            if existing_keyframes:
                # Check frame count compatibility
                old_frame_count = len(self.video_frames) if hasattr(self, 'video_frames') else 0
                new_frame_count = len(frames)
                
                # Check which keyframes can be preserved
                max_keyframe = max(existing_keyframes.keys())
                keyframes_that_fit = {k: v for k, v in existing_keyframes.items() if k < new_frame_count}
                keyframes_to_discard = len(existing_keyframes) - len(keyframes_that_fit)
                
                # Ask user if they want to preserve keyframes
                msg = QMessageBox()
                msg.setWindowTitle("Preserve Mask Keyframes?")
                
                if keyframes_to_discard > 0:
                    # Some keyframes will be discarded
                    msg.setText(f"You have existing mask keyframes.\n\n"
                               f"Previous video: {old_frame_count} frames\n"
                               f"New video: {new_frame_count} frames\n"
                               f"Total keyframes: {len(existing_keyframes)}\n"
                               f"Keyframes that fit: {len(keyframes_that_fit)}\n"
                               f"Keyframes to discard: {keyframes_to_discard} (frames {new_frame_count}-{max_keyframe})\n\n"
                               f"Do you want to preserve the keyframes that fit?")
                else:
                    # All keyframes fit
                    msg.setText(f"You have existing mask keyframes.\n\n"
                               f"Previous video: {old_frame_count} frames\n"
                               f"New video: {new_frame_count} frames\n"
                               f"Keyframes: {len(existing_keyframes)} (all fit)\n\n"
                               f"Do you want to preserve the existing keyframes?")
                
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.Yes)
                
                response = msg.exec_()
                logger.info(f"[KEYFRAME DEBUG] Dialog response: {response} (Yes={QMessageBox.Yes}, No={QMessageBox.No})")
                
                if response == QMessageBox.Yes:
                    preserve_keyframes = True
                    # Only keep keyframes that fit in the new video
                    if keyframes_to_discard > 0:
                        existing_keyframes = keyframes_that_fit
                        logger.info(f"[KEYFRAME DEBUG] User chose to preserve {len(keyframes_that_fit)} keyframes that fit, discarding {keyframes_to_discard}")
                    else:
                        logger.info("[KEYFRAME DEBUG] User chose to preserve all existing keyframes")
                else:
                    preserve_keyframes = False
                    logger.info("[KEYFRAME DEBUG] User chose to discard existing keyframes")
            
            # Reinitialize the mask editor with new frames
            reinit_start = time.time()
            logger.info(f"Starting reinitialize_with_frames...")
            
            # Write to debug file
            debug_file = os.path.join(os.path.dirname(self.output_dir), "loading_debug.txt") if self.output_dir else "loading_debug.txt"
            with open(debug_file, 'w') as f:
                f.write(f"Loading started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Video loaded: {len(frames)} frames in {time.time() - load_start:.2f}s\n")
                f.write(f"Starting reinitialization...\n")
            
            try:
                logger.info(f"[KEYFRAME DEBUG] About to call reinitialize_with_frames:")
                logger.info(f"[KEYFRAME DEBUG]   - preserve_keyframes: {preserve_keyframes}")
                logger.info(f"[KEYFRAME DEBUG]   - saved_keyframes: {len(existing_keyframes) if existing_keyframes and preserve_keyframes else 'None'}")
                self.reinitialize_with_frames(frames, preserve_keyframes=preserve_keyframes, 
                                            saved_keyframes=existing_keyframes if preserve_keyframes else None)
                logger.info(f"Reinitialization took {time.time() - reinit_start:.2f}s")
                logger.info(f"TOTAL time from start: {time.time() - load_start:.2f}s")
                logger.info(f"========== Load complete ==========")
                
                with open(debug_file, 'a') as f:
                    f.write(f"Reinitialization completed in {time.time() - reinit_start:.2f}s\n")
                    f.write(f"TOTAL time: {time.time() - load_start:.2f}s\n")
                    f.write("========== Load complete ==========\n")
                    f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Post-load: Method returning...\n")
                
                # Log that we're exiting the method
                logger.info("load_video_files method completed, returning to caller")
                
                # Set up a timer to check what happens after we return
                def check_post_load():
                    elapsed = time.time() - load_start
                    logger.info(f"Post-load check: {elapsed:.1f}s since load started")
                    with open(debug_file, 'a') as f:
                        f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Post-load check: {elapsed:.1f}s since load started\n")
                        f.write(f"Main window responsive: {not self.isModal() if hasattr(self, 'isModal') else 'Unknown'}\n")
                
                # Check after 1, 5, and 10 seconds
                QTimer.singleShot(1000, check_post_load)
                QTimer.singleShot(5000, check_post_load)
                QTimer.singleShot(10000, check_post_load)
                    
            except Exception as e:
                logger.error(f"Error during reinitialization: {e}")
                import traceback
                traceback.print_exc()
                
                with open(debug_file, 'a') as f:
                    f.write(f"ERROR: {e}\n")
                    f.write(traceback.format_exc())
            
            # Return success for the wrapper
            return True
        else:
            # No frames loaded
            return False
    
    def reinitialize_with_frames(self, frames, preserve_keyframes=False, saved_keyframes=None):
        """Reinitialize the mask editor with new frames
        
        Args:
            frames: New video frames to load
            preserve_keyframes: Whether to preserve existing shape keyframes
            saved_keyframes: The saved keyframes to restore (if preserve_keyframes is True)
        """
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        # Start time tracking
        start_time = time.time()
        
        logger.info(f"reinitialize_with_frames called with preserve_keyframes={preserve_keyframes}")
        
        # Ensure frames are uint8 (0-255) not float
        convert_start = time.time()
        logger.info(f"Converting frames to uint8...")
        
        processed_frames = []
        for i, frame in enumerate(frames):
            if frame.dtype != np.uint8:
                # Convert to uint8 if needed
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            processed_frames.append(frame)
            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(frames)} frames...")
        
        frames = processed_frames
        logger.info(f"Frame conversion took {time.time() - convert_start:.2f}s")
        
        # Store the frames
        self.current_frames = frames
        
        # Update the actual frames in the parent class
        self.video_frames = frames
        self.frames = frames
        self.original_frames = frames  # Reference, not copy - much faster
        
        # Reset current frame index
        self.current_frame_index = 0
        
        # Create new mask frames
        mask_start = time.time()
        logger.info(f"Creating mask frames for {len(frames)} frames...")
        
        # Write status to file for debugging
        debug_file = os.path.join(os.path.dirname(self.output_dir), "loading_debug.txt") if self.output_dir else "loading_debug.txt"
        with open(debug_file, 'a') as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Creating mask frames...\n")
            f.write(f"Frame count: {len(frames)}\n")
            f.write(f"Frame shape: {frames[0].shape if frames else 'None'}\n")
        
        # Create masks
        self.mask_frames = []
        for i in range(len(frames)):
            frame = frames[i]
            self.mask_frames.append(np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8))
            if i % 50 == 0:
                logger.info(f"Created {i}/{len(frames)} masks...")
                # Keep UI responsive during mask creation
                QApplication.processEvents()
        
        # Store keyframes temporarily if we're preserving them
        temp_keyframes = None
        if preserve_keyframes and saved_keyframes and hasattr(self, 'mask_widget'):
            temp_keyframes = saved_keyframes
            logger.info(f"[KEYFRAME DEBUG] Temporarily storing {len(temp_keyframes)} keyframes during mask creation")
        
        mask_time = time.time() - mask_start
        logger.info(f"Mask creation took {mask_time:.2f}s")
        
        with open(debug_file, 'a') as f:
            f.write(f"Mask creation completed in {mask_time:.2f}s\n")
            
        # Log each subsequent step
        def log_step(step_name, elapsed_time=None):
            with open(debug_file, 'a') as f:
                if elapsed_time:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {step_name} took {elapsed_time:.2f}s\n")
                else:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {step_name}\n")
        
        # Update the mask widget if it exists
        widget_start = time.time()
        if hasattr(self, 'mask_widget'):
            print(f"[Reinit] Updating mask widget...")
            
            # Set initializing flag to defer expensive operations
            self.mask_widget._initializing = True
            
            # Update the mask widget's frames
            print(f"[Reinit] Setting frames on mask widget...")
            
            # Check keyframes before setting frames
            if preserve_keyframes and saved_keyframes:
                keyframe_count_before = len(self.mask_widget.shape_keyframes) if hasattr(self.mask_widget, 'shape_keyframes') else 0
                logger.info(f"[KEYFRAME DEBUG] Before setting frames: {keyframe_count_before} keyframes")
            
            self.mask_widget.frames = frames
            
            # Check after setting frames
            if preserve_keyframes and saved_keyframes:
                keyframe_count_after = len(self.mask_widget.shape_keyframes) if hasattr(self.mask_widget, 'shape_keyframes') else 0
                logger.info(f"[KEYFRAME DEBUG] After setting frames: {keyframe_count_after} keyframes")
                if keyframe_count_after == 0 and keyframe_count_before > 0:
                    logger.error(f"[KEYFRAME DEBUG] FRAMES ASSIGNMENT CLEARED KEYFRAMES!")
                    
            self.mask_widget.original_frames = self.original_frames
            self.mask_widget.video_frames = frames  # Some versions might use this
            
            # Reset mask data
            print(f"[Reinit] Resetting mask widget data...")
            # Always clear keyframes initially - we'll restore them later after everything is initialized
            self.mask_widget.shape_keyframes = {}
            if preserve_keyframes and saved_keyframes:
                logger.info(f"[KEYFRAME DEBUG] Temporarily cleared keyframes - will restore {len(saved_keyframes)} keyframes after initialization")
            else:
                logger.info(f"[KEYFRAME DEBUG] Cleared all shape keyframes (preserve_keyframes={preserve_keyframes})")
            self.mask_widget.current_frame = 0
            
            # Check keyframes before assigning masks
            if preserve_keyframes and saved_keyframes:
                keyframe_count_before = len(self.mask_widget.shape_keyframes) if hasattr(self.mask_widget, 'shape_keyframes') else 0
                logger.info(f"[KEYFRAME DEBUG] Before assigning masks: {keyframe_count_before} keyframes")
            
            self.mask_widget.masks = self.mask_frames
            
            # Check keyframes after assigning masks
            if preserve_keyframes and saved_keyframes:
                keyframe_count_after = len(self.mask_widget.shape_keyframes) if hasattr(self.mask_widget, 'shape_keyframes') else 0
                logger.info(f"[KEYFRAME DEBUG] After assigning masks: {keyframe_count_after} keyframes")
                if keyframe_count_after == 0 and keyframe_count_before > 0:
                    logger.error(f"[KEYFRAME DEBUG] MASKS ASSIGNMENT CLEARED KEYFRAMES!")
            
            # Force update the display with the new frame
            logger.info(f"Forcing display update...")
            log_step("Forcing mask widget display update")
            
            # Disable updates temporarily to prevent multiple repaints
            self.mask_widget.setUpdatesEnabled(False)
            
            if hasattr(self.mask_widget, 'set_frame'):
                set_frame_start = time.time()
                self.mask_widget.set_frame(0)
                logger.info(f"set_frame took {time.time() - set_frame_start:.2f}s")
                log_step(f"set_frame completed", time.time() - set_frame_start)
            elif hasattr(self.mask_widget, 'update_display'):
                update_display_start = time.time()
                self.mask_widget.update_display()
                logger.info(f"update_display took {time.time() - update_display_start:.2f}s")
                log_step(f"update_display completed", time.time() - update_display_start)
            else:
                # Force a repaint
                update_start = time.time()
                self.mask_widget.update()
                logger.info(f"widget update took {time.time() - update_start:.2f}s")
                log_step(f"widget update completed", time.time() - update_start)
                
            # Re-enable updates
            self.mask_widget.setUpdatesEnabled(True)
                
            print(f"[Reinit] Total mask widget update took {time.time() - widget_start:.2f}s")
            
        # Force close any hanging dialogs (BUT NOT OUR PROGRESS DIALOG!)
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QProgressDialog):
                if hasattr(self, '_progress_dialog'):
                    if widget != self._progress_dialog:
                        logger.info(f"Closing hanging dialog: {widget.windowTitle()}")
                        widget.close()
                else:
                    logger.info(f"Closing hanging dialog: {widget.windowTitle()}")
                    widget.close()
            elif isinstance(widget, QMessageBox) and widget.windowTitle() in ["Loading Video", "Select Input Type"]:
                logger.info(f"Closing hanging dialog: {widget.windowTitle()}")
                widget.close()
            
        # Update timeline if it exists (it's called timeline_widget in the mask editor)
        if hasattr(self, 'timeline_widget') and self.timeline_widget:
            self.timeline_widget.total_frames = len(frames)
            self.timeline_widget.current_frame = 0
            
            # Only reset mask indicators if we're not preserving keyframes
            if not (preserve_keyframes and saved_keyframes):
                self.timeline_widget.mask_frames = set()  # Reset mask indicators
                logger.info(f"[KEYFRAME DEBUG] Timeline mask_frames cleared")
            else:
                # Preserve timeline indicators for keyframes
                preserved_frames = set()
                if saved_keyframes:
                    for frame_idx in saved_keyframes.keys():
                        if frame_idx < len(frames):  # Only keep frames within new range
                            preserved_frames.add(frame_idx)
                self.timeline_widget.mask_frames = preserved_frames
                logger.info(f"[KEYFRAME DEBUG] Preserved {len(preserved_frames)} timeline indicators")
            
            self.timeline_widget.parent_editor = self  # Ensure parent reference is maintained
            self.timeline_widget.update()
            print(f"[Enhanced Mask Editor] Updated timeline with {len(frames)} frames")
            
            # Force timeline to be interactive
            self.timeline_widget.setEnabled(True)
            self.timeline_widget.setMouseTracking(True)
            
        # Update any frame counters or labels
        if hasattr(self, 'frame_label') and self.frame_label:
            self.frame_label.setText(f"Frame 1/{len(frames)}")
            
        # Force the dialog to update
        qt_start = time.time()
        print(f"[Reinit] Processing Qt events...")
        self.update()
        QApplication.processEvents()
        print(f"[Reinit] Qt event processing took {time.time() - qt_start:.2f}s")
        
        # Temporarily switch to brush mode to avoid shape calculations
        original_mode = self.drawing_mode if hasattr(self, 'drawing_mode') else None
        if original_mode == 'shape':
            logger.info("Temporarily switching to brush mode to avoid shape calculations during load")
            if hasattr(self, 'mask_widget'):
                self.mask_widget.drawing_mode = 'brush'
        
        # Update the display to show the first frame
        display_start = time.time()
        logger.info(f"Updating display...")
        logger.info(f"Current drawing mode: {self.drawing_mode if hasattr(self, 'drawing_mode') else 'None'}")
        log_step("Starting update_display")
        
        if hasattr(self, 'update_display'):
            try:
                self.update_display()
            except Exception as e:
                logger.error(f"ERROR in update_display: {e}")
                import traceback
                traceback.print_exc()
                log_step(f"ERROR in update_display: {e}")
                
        display_time = time.time() - display_start
        logger.info(f"Display update took {display_time:.2f}s")
        log_step(f"Display update completed", display_time)
        
        # Restore original mode
        if original_mode == 'shape' and hasattr(self, 'mask_widget'):
            self.mask_widget.drawing_mode = original_mode
            logger.info("Restored shape mode")
            
        # Log keyframe status before frame change
        if preserve_keyframes and saved_keyframes and hasattr(self, 'mask_widget'):
            keyframe_count = len(self.mask_widget.shape_keyframes) if hasattr(self.mask_widget, 'shape_keyframes') else 0
            logger.info(f"[KEYFRAME DEBUG] Before frame change: mask_widget has {keyframe_count} keyframes")
                
        # Trigger frame change to ensure everything is properly initialized
        frame_change_start = time.time()
        print(f"[Reinit] Triggering frame change...")
        
        if hasattr(self, 'on_frame_changed'):
            self.on_frame_changed(0)
        print(f"[Reinit] Frame change took {time.time() - frame_change_start:.2f}s")
        
        # Verify keyframes after frame change
        if preserve_keyframes and saved_keyframes and hasattr(self, 'mask_widget'):
            keyframe_count = len(self.mask_widget.shape_keyframes) if hasattr(self.mask_widget, 'shape_keyframes') else 0
            logger.info(f"[KEYFRAME DEBUG] After frame change: mask_widget has {keyframe_count} keyframes")
            if keyframe_count == 0 and len(saved_keyframes) > 0:
                logger.error(f"[KEYFRAME DEBUG] ERROR: Keyframes were cleared! Originally had {len(saved_keyframes)}")
        
        # Calculate actual time taken
        actual_time = time.time() - start_time
        logger.info(f"ACTUAL reinitialize_with_frames time: {actual_time:.1f} seconds")
        
        # Function to restore keyframes after everything is initialized
        def restore_keyframes_final():
            if preserve_keyframes and saved_keyframes and hasattr(self, 'mask_widget'):
                logger.info(f"[KEYFRAME DEBUG] Final keyframe restoration")
                self.mask_widget.shape_keyframes = saved_keyframes
                
                if hasattr(self.mask_widget, 'invalidate_shape_cache'):
                    self.mask_widget.invalidate_shape_cache()
                if hasattr(self.mask_widget, 'update_mask_from_shapes'):
                    self.mask_widget.update_mask_from_shapes()
                    
                restored_count = len(self.mask_widget.shape_keyframes)
                logger.info(f"[KEYFRAME DEBUG] Finally restored {restored_count} keyframes")
                
                # Ensure we're in shape mode
                if hasattr(self, 'drawing_mode') and self.drawing_mode != 'shape':
                    self.drawing_mode = 'shape'
                    logger.info(f"[KEYFRAME DEBUG] Switched to shape mode")
                    
                # Update timeline
                if hasattr(self, 'update_mask_frame_tracking'):
                    self.update_mask_frame_tracking()
                    logger.info(f"[KEYFRAME DEBUG] Updated mask frame tracking")
        
        # Complete the mask widget initialization if it was deferred
        if hasattr(self, 'mask_widget') and hasattr(self.mask_widget, '_initializing') and self.mask_widget._initializing:
            # Complete initialization after a brief delay to ensure UI is responsive
            def complete_mask_init():
                # Check keyframes before completing initialization
                if preserve_keyframes and saved_keyframes:
                    keyframe_count = len(self.mask_widget.shape_keyframes) if hasattr(self.mask_widget, 'shape_keyframes') else 0
                    logger.info(f"[KEYFRAME DEBUG] Before complete_initialization: {keyframe_count} keyframes")
                
                self.mask_widget.complete_initialization()
                logger.info("Completed deferred mask widget initialization")
                
                # Check keyframes after completing initialization
                if preserve_keyframes and saved_keyframes:
                    keyframe_count = len(self.mask_widget.shape_keyframes) if hasattr(self.mask_widget, 'shape_keyframes') else 0
                    logger.info(f"[KEYFRAME DEBUG] After complete_initialization: {keyframe_count} keyframes")
                    
                    # If we preserved keyframes, update the timeline
                    if keyframe_count > 0 and hasattr(self, 'update_mask_frame_tracking'):
                        logger.info(f"[KEYFRAME DEBUG] Calling update_mask_frame_tracking to refresh timeline")
                        original_mode = self.drawing_mode if hasattr(self, 'drawing_mode') else None
                        self.drawing_mode = 'shape'  # Ensure shape mode for tracking
                        self.update_mask_frame_tracking()
                        if original_mode:
                            self.drawing_mode = original_mode
                            
            # Store the saved_keyframes in closure for delayed restoration
            def complete_and_restore():
                complete_mask_init()
                restore_keyframes_final()
                        
            if preserve_keyframes and saved_keyframes:
                QTimer.singleShot(100, complete_and_restore)
            else:
                QTimer.singleShot(100, complete_mask_init)
        else:
            # If mask widget is not in initializing state, still restore keyframes after a delay
            if preserve_keyframes and saved_keyframes:
                logger.info(f"[KEYFRAME DEBUG] Mask widget not initializing, scheduling keyframe restoration")
                QTimer.singleShot(200, restore_keyframes_final)
            
        print(f"[Enhanced Mask Editor] Reinitialization complete: {len(frames)} frames, shape: {frames[0].shape if frames else 'None'}")
    
    def save_project(self):
        """Save current project"""
        if self.current_project_path:
            self._save_project_to_file(self.current_project_path)
        else:
            self.save_project_as()
        
        # Also save as the "saved" session state
        self.save_session_as_saved_state()
    
    def save_project_as(self):
        """Save project with file dialog"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "WAN Mask Project (*.wmp);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.current_project_path = file_path
            self._save_project_to_file(file_path)
    
    def _save_project_to_file(self, file_path):
        """Save project data to file"""
        try:
            project_data = {
                "version": "2.0",  # Updated version for new format
                "source_video": {
                    "path": self.source_video_path if self.source_video_path else "",
                    "type": self.source_type if self.source_type else "unknown"
                },
                "frames_info": {
                    "count": len(self.current_frames),
                    "width": self.current_frames[0].shape[1] if self.current_frames else 512,
                    "height": self.current_frames[0].shape[0] if self.current_frames else 512
                },
                "shape_keyframes": {},
                "settings": {
                    "drawing_mode": getattr(self, 'drawing_mode', 'brush'),
                    "brush_size": getattr(self, 'brush_size', 30),
                    "vertex_count": getattr(self.vertex_count_slider, 'value', lambda: 6)() if hasattr(self, 'vertex_count_slider') else 6
                }
            }
            
            # Get shape keyframes from mask widget
            if hasattr(self, 'mask_widget') and hasattr(self.mask_widget, 'shape_keyframes'):
                for frame, shapes in self.mask_widget.shape_keyframes.items():
                    project_data["shape_keyframes"][str(frame)] = shapes
            
            # Save non-empty mask frames as base64 (only the masks, not video frames)
            project_data["mask_frames"] = {}
            if hasattr(self, 'mask_frames'):
                for i, mask in enumerate(self.mask_frames):
                    if mask is not None and isinstance(mask, np.ndarray) and np.any(mask > 0):
                        # Encode mask as PNG
                        success, buffer = cv2.imencode('.png', mask)
                        if success:
                            encoded = base64.b64encode(buffer).decode('utf-8')
                            project_data["mask_frames"][str(i)] = encoded
            
            # Save project file
            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            print(f"[Save Project] Saved to {file_path} (video path: {self.source_video_path})")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save project: {e}")
    
    def load_project(self):
        """Load project from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "WAN Mask Project (*.wmp);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self._load_project_from_file(file_path)
    
    def _load_project_from_file(self, file_path):
        """Load project data from file"""

        # Import necessary modules
        from PyQt5.QtCore import QThread, pyqtSignal, QTimer

        # Define worker class
        class LoadWorker(QThread):
            finished = pyqtSignal(dict)

            def __init__(self, file_path):
                super().__init__()
                self.file_path = file_path

            def run(self):
                try:
                    with open(self.file_path, 'r') as f:
                        project_data = json.load(f)

                    # Load frames if new format
                    frames = []
                    source_path = None
                    source_type = None
                    if "source_video" in project_data and project_data["source_video"].get("path"):
                        source_path = project_data["source_video"]["path"]
                        source_type = project_data["source_video"]["type"]

                        if source_type == "video" and os.path.exists(source_path):
                            cap = cv2.VideoCapture(source_path)
                            if cap.isOpened():
                                while True:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    frames.append(frame)
                                cap.release()

                        elif source_type == "image_sequence" and os.path.exists(source_path):
                            # Use optimized parallel loading
                            frames = load_image_sequence_optimized_mask_editor(source_path)

                    # Load shape keyframes
                    shape_keyframes = {}
                    if "shape_keyframes" in project_data:
                        for frame_str, shapes in project_data["shape_keyframes"].items():
                            frame_num = int(frame_str)
                            shape_keyframes[frame_num] = shapes

                    # Load mask frames
                    loaded_mask_frames = {}
                    if "mask_frames" in project_data:
                        for frame_str, mask_data in project_data["mask_frames"].items():
                            frame_num = int(frame_str)
                            mask_bytes = base64.b64decode(mask_data)
                            mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
                            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
                            if mask is not None:
                                loaded_mask_frames[frame_num] = mask

                    # Load settings
                    settings = project_data.get("settings", {})

                    result = {
                        "success": True,
                        "project_data": project_data,
                        "frames": frames,
                        "source_path": source_path,
                        "source_type": source_type,
                        "shape_keyframes": shape_keyframes,
                        "mask_frames": loaded_mask_frames,
                        "settings": settings
                    }
                except Exception as e:
                    result = {"success": False, "error": str(e)}

                self.finished.emit(result)

        # Create and show progress dialog
        progress = QProgressDialog("Loading project...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        # Start fake progress timer
        progress_value = 0
        def update_fake_progress():
            nonlocal progress_value
            if progress_value < 90:
                progress_value += 1
                progress.setValue(progress_value)

        fake_timer = QTimer()
        fake_timer.timeout.connect(update_fake_progress)
        fake_timer.start(50)  # Increment every 50ms

        # Start worker
        worker = LoadWorker(file_path)
        def on_finished(result):
            fake_timer.stop()
            if result["success"]:
                # Apply loaded data
                project_data = result["project_data"]
                frames = result["frames"]
                if frames:
                    self.current_frames = frames
                    self.source_video_path = result["source_path"]
                    self.source_type = result["source_type"]
                    self.reinitialize_with_frames(frames)

                if hasattr(self, 'mask_widget'):
                    self.mask_widget.shape_keyframes = result["shape_keyframes"]
                    self.mask_widget.invalidate_shape_cache()
                    def complete_shape_loading():
                        self.mask_widget.update_mask_from_shapes()
                        self.mask_widget.complete_initialization()
                        if hasattr(self, 'update_mask_frame_tracking'):
                            original_mode = self.drawing_mode
                            self.drawing_mode = 'shape'
                            self.update_mask_frame_tracking()
                            self.drawing_mode = original_mode
                    QTimer.singleShot(100, complete_shape_loading)

                if hasattr(self, 'mask_frames'):
                    for frame_num, mask in result["mask_frames"].items():
                        if frame_num < len(self.mask_frames):
                            self.mask_frames[frame_num] = mask

                settings = result["settings"]
                if "drawing_mode" in settings:
                    self.drawing_mode = settings["drawing_mode"]
                if "brush_size" in settings:
                    self.brush_size = settings["brush_size"]
                    if hasattr(self, 'brush_size_slider'):
                        self.brush_size_slider.setValue(self.brush_size)
                if "vertex_count" in settings:
                    vertex_count = settings["vertex_count"]
                    if hasattr(self, 'vertex_count_slider'):
                        self.vertex_count_slider.setValue(vertex_count)
                        print(f"[Load Project] Restored vertex count to {vertex_count}")
                    if hasattr(self, 'mask_widget'):
                        self.mask_widget.target_vertex_count = vertex_count

                self.current_project_path = file_path

                progress.setValue(100)
            else:
                progress.cancel()
                QMessageBox.critical(self, "Load Error", f"Failed to load project: {result['error']}")

            self.session_timer.start(30000)
            progress.close()
            # Do NOT call worker.deleteLater() here!

        worker.finished.connect(on_finished)
        # Ensure thread is deleted after finishing
        worker.finished.connect(worker.deleteLater)
        worker.start()
        # Keep a reference to the worker to prevent garbage collection
        self._load_worker = worker
    
    def get_frames_for_export(self):
        """Get current frames for export"""
        return self.current_frames
    
    def auto_save_session(self):
        """Auto-save session data to temporary location"""
        # Don't auto-save if we're in the process of closing/discarding
        if hasattr(self, 'is_closing') and self.is_closing:
            self.logger.info("[Auto-save] Skipping auto-save - window is closing")
            return
            
        if self.output_dir and self.current_frames:
            try:
                import time
                start_time = time.time()
                self.logger.info("[Auto-save] Starting auto-save session...")
                self.logger.info(f"[Auto-save] output_dir: {self.output_dir}")
                self.logger.info(f"[Auto-save] source_video_path: {getattr(self, 'source_video_path', None)}")
                self.logger.info(f"[Auto-save] source_type: {getattr(self, 'source_type', None)}")
                
                # Don't save video frames at all - they're loaded from source
                # Only save mask data and shape keyframes
                frame_files = []
                
                # Create session data
                session_data = {
                    "frames_info": {
                        "count": len(self.current_frames),
                        "width": self.current_frames[0].shape[1] if self.current_frames else 512,
                        "height": self.current_frames[0].shape[0] if self.current_frames else 512
                    },
                    "shape_keyframes": {},
                    "current_frame": self.current_frame_index,
                    "drawing_mode": self.drawing_mode,
                    "current_project_path": self.current_project_path,
                    # Use video_info to match what the loader expects
                    "video_info": {
                        "path": getattr(self, 'source_video_path', None),
                        "type": getattr(self, 'source_type', None),
                        "total_frames": len(self.current_frames) if self.current_frames else 0,
                        "width": self.current_frames[0].shape[1] if self.current_frames else 512,
                        "height": self.current_frames[0].shape[0] if self.current_frames else 512
                    },
                    # Keep source_video for backward compatibility
                    "source_video": {
                        "path": getattr(self, 'source_video_path', None),
                        "type": getattr(self, 'source_type', None)
                    }
                }
                
                # Get shape keyframes from mask widget
                shape_copy_start = time.time()
                if hasattr(self, 'mask_widget') and hasattr(self.mask_widget, 'shape_keyframes'):
                    # Check if accessing shape_keyframes triggers something expensive
                    self.logger.info(f"[Auto-save] Copying {len(self.mask_widget.shape_keyframes)} shape keyframes...")
                    for frame, shapes in self.mask_widget.shape_keyframes.items():
                        session_data["shape_keyframes"][str(frame)] = shapes
                self.logger.info(f"[Auto-save] Shape copying took {time.time() - shape_copy_start:.2f}s")
                
                # Save to temporary auto-save file (NOT the current state)
                json_save_start = time.time()
                autosave_file = os.path.join(self.output_dir, "working_autosave.json")
                with open(autosave_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                self.logger.info(f"[Auto-save] JSON saving to working_autosave.json took {time.time() - json_save_start:.2f}s")
                
                total_time = time.time() - start_time
                self.logger.info(f"[Auto-save] Total auto-save time: {total_time:.2f}s")
                
                # If auto-save is taking too long, it might be causing the freeze
                if total_time > 1.0:
                    self.logger.warning(f"[Auto-save] WARNING: Auto-save took {total_time:.2f}s - this might cause UI freezing!")
                    
            except Exception as e:
                self.logger.error(f"[Enhanced Mask Editor] Auto-save session error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
    
    def check_and_reload_video(self):
        """Check if video needs to be reloaded after window is shown"""
        self.logger.info("[EnhancedMaskEditor] Checking if video needs reload...")
        self.logger.info(f"[EnhancedMaskEditor] Current frames: {len(self.current_frames) if self.current_frames else 0}")
        self.logger.info(f"[EnhancedMaskEditor] Source video path: {self.source_video_path}")
        self.logger.info(f"[EnhancedMaskEditor] Source type: {self.source_type}")
        
        # Check if source file exists, prompt for new location if not
        if self.source_video_path and not os.path.exists(self.source_video_path):
            self.logger.info(f"[EnhancedMaskEditor] Source file not found: {self.source_video_path}")
            new_path = prompt_for_missing_file(self.source_video_path, self.source_type, self)
            if new_path:
                self.logger.info(f"[EnhancedMaskEditor] User selected new path: {new_path}")
                self.source_video_path = new_path
                # Update session data with new path
                self.auto_save_session()
            else:
                self.logger.info("[EnhancedMaskEditor] User cancelled file browse")
                return
        
        # If we have a source path but no frames, reload the video/images
        if self.source_video_path and (not self.current_frames or len(self.current_frames) == 0):
            self.logger.info(f"[EnhancedMaskEditor] No frames but have source path, reloading from: {self.source_video_path}")
            
            if self.source_type == "video" and os.path.exists(self.source_video_path):
                # Reload video file
                cap = cv2.VideoCapture(self.source_video_path)
                if cap.isOpened():
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    cap.release()
                    if frames:
                        self.logger.info(f"[EnhancedMaskEditor] Reloaded {len(frames)} frames from video")
                        self.current_frames = frames
                        # Preserve keyframes when reloading the same video from a new location
                        existing_keyframes = None
                        if hasattr(self, 'mask_widget') and hasattr(self.mask_widget, 'shape_keyframes'):
                            if self.mask_widget.shape_keyframes:
                                import copy
                                existing_keyframes = copy.deepcopy(self.mask_widget.shape_keyframes)
                        self.reinitialize_with_frames(frames, preserve_keyframes=True, saved_keyframes=existing_keyframes)
                    else:
                        self.logger.error("[EnhancedMaskEditor] Failed to read frames from video")
                else:
                    self.logger.error(f"[EnhancedMaskEditor] Failed to open video: {self.source_video_path}")
                    
            elif self.source_type == "image_sequence" and os.path.exists(self.source_video_path):
                # Reload image sequence
                self.logger.info(f"[EnhancedMaskEditor] Reloading image sequence from folder: {self.source_video_path}")
                # source_video_path for image sequences is the folder path
                folder_path = self.source_video_path
                
                # Use optimized parallel loading
                frames = load_image_sequence_optimized_mask_editor(folder_path)
                
                if frames:
                    self.logger.info(f"[EnhancedMaskEditor] Reloaded {len(frames)} frames from image sequence")
                    self.current_frames = frames
                    # Preserve keyframes when reloading the same image sequence from a new location
                    existing_keyframes = None
                    if hasattr(self, 'mask_widget') and hasattr(self.mask_widget, 'shape_keyframes'):
                        if self.mask_widget.shape_keyframes:
                            import copy
                            existing_keyframes = copy.deepcopy(self.mask_widget.shape_keyframes)
                    self.reinitialize_with_frames(frames, preserve_keyframes=True, saved_keyframes=existing_keyframes)
                else:
                    self.logger.error(f"[EnhancedMaskEditor] No images found in folder: {self.source_video_path}")
                    
            elif self.source_type:
                self.logger.warning(f"[EnhancedMaskEditor] Source file/pattern doesn't exist: {self.source_video_path}")
            else:
                self.logger.warning(f"[EnhancedMaskEditor] Unknown source type: {self.source_type}")
        else:
            self.logger.info("[EnhancedMaskEditor] Video frames already loaded or no source path")
    
    def save_session_as_saved_state(self):
        """Save the current working session as the saved state (for explicit saves)"""
        if self.output_dir:
            try:
                # When user explicitly saves, update both current state and saved project
                autosave_file = os.path.join(self.output_dir, "working_autosave.json")
                current_state_file = os.path.join(self.output_dir, "current_state.json")
                saved_project_file = os.path.join(self.output_dir, "saved_project.json")
                
                # First ensure we have the latest data
                self.auto_save_session()  # Creates working_autosave.json
                
                if os.path.exists(autosave_file):
                    import shutil
                    # Update current state
                    shutil.copy2(autosave_file, current_state_file)
                    # Update saved project
                    shutil.copy2(autosave_file, saved_project_file)
                    self.logger.info(f"[EnhancedMaskEditor] Saved to both current_state.json and saved_project.json")
            except Exception as e:
                self.logger.error(f"[EnhancedMaskEditor] Error saving session state: {e}")
    
    def apply_working_to_current_state(self):
        """Apply working auto-save to current state (when accepting changes)"""
        if self.output_dir:
            try:
                autosave_file = os.path.join(self.output_dir, "working_autosave.json")
                current_state_file = os.path.join(self.output_dir, "current_state.json")
                
                if os.path.exists(autosave_file):
                    import shutil
                    shutil.copy2(autosave_file, current_state_file)
                    os.remove(autosave_file)  # Clean up working file after applying
                    self.logger.info(f"[EnhancedMaskEditor] Applied working changes to current_state.json")
                else:
                    self.logger.warning(f"[EnhancedMaskEditor] No working_autosave.json to apply")
            except Exception as e:
                self.logger.error(f"[EnhancedMaskEditor] Error applying working state: {e}")
    
    def hide_duplicate_buttons(self):
        """Hide duplicate save/load project buttons from the parent dialog"""
        try:
            # Find all QPushButtons in the dialog
            for widget in self.findChildren(QPushButton):
                text = widget.text()
                # Hide the duplicate save/load project buttons
                if "Save Project" in text or "Load Project" in text:
                    # Only hide if it's not in our toolbar
                    if widget.parent() and not isinstance(widget.parent().layout(), QHBoxLayout):
                        widget.hide()
                        print(f"[Enhanced Mask Editor] Hid duplicate button: {text}")
        except Exception as e:
            print(f"[Enhanced Mask Editor] Error hiding duplicate buttons: {e}")
    
    def closeEvent(self, event):
        """Override closeEvent to handle session timer and cleanup"""
        # Set flag to prevent auto-save during close
        self.is_closing = True
        
        # Stop session timer immediately
        if hasattr(self, 'session_timer'):
            self.session_timer.stop()
            self.logger.info("[EnhancedMaskEditor] Stopped session timer")
        
        # Let parent handle the close event
        # This will show the save dialog if needed and set is_discarding flag
        super().closeEvent(event)
        
        # If the event was rejected (user cancelled close)
        if not event.isAccepted():
            # Clear the closing flag
            self.is_closing = False
            # Restart session timer
            if hasattr(self, 'session_timer'):
                self.session_timer.start(30000)
                self.logger.info("[EnhancedMaskEditor] Close cancelled - restarted session timer")
        # Note: Actual cleanup happens in the main() function based on dialog result
    
    def clean_up_session_data(self):
        """Override parent's cleanup to use our two-state system"""
        # Don't delete saved session data - only clean up working data
        self.clean_up_enhanced_session_data()
    
    def cancel_without_prompts(self):
        """Override to clean up session data when Cancel is clicked"""
        self.logger.info("[EnhancedMaskEditor] Cancel button clicked - cleaning up session")
        # Set flag to indicate we're cancelling
        self.is_cancelling = True
        # Clean up session data
        self.clean_up_enhanced_session_data()
        # Call parent's cancel method
        super().cancel_without_prompts()
    
    def clean_up_enhanced_session_data(self):
        """Clean up working auto-save data only (preserves current and saved states)"""
        try:
            # Only clean up the temporary working auto-save, not current or saved states
            if self.output_dir:
                autosave_file = os.path.join(self.output_dir, "working_autosave.json")
                if os.path.exists(autosave_file):
                    os.remove(autosave_file)
                    self.logger.info("[EnhancedMaskEditor] Removed working_autosave.json")
                
                # Also clean up legacy files if they exist
                legacy_files = ["working_session_data.json", "session_data.json"]
                for legacy_file in legacy_files:
                    file_path = os.path.join(self.output_dir, legacy_file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        self.logger.info(f"[EnhancedMaskEditor] Removed legacy file: {legacy_file}")
            
            # Clean up persistent session storage
            import tempfile
            sessions_dir = os.path.join(tempfile.gettempdir(), "wan_mask_sessions")
            if os.path.exists(sessions_dir):
                # Find node ID from the server data if available
                # This is a bit tricky since we don't store node_id in the editor
                # For now, clean up based on output_dir matching
                for session_file in os.listdir(sessions_dir):
                    if session_file.startswith("node_") and session_file.endswith("_session.json"):
                        session_path = os.path.join(sessions_dir, session_file)
                        try:
                            with open(session_path, 'r') as f:
                                session_data = json.load(f)
                            if session_data.get("output_dir") == self.output_dir:
                                os.remove(session_path)
                                self.logger.info(f"[EnhancedMaskEditor] Removed persistent session: {session_file}")
                        except Exception as e:
                            self.logger.error(f"[EnhancedMaskEditor] Error checking session file {session_file}: {e}")
            
            self.logger.info("[EnhancedMaskEditor] Enhanced session data cleaned up")
        except Exception as e:
            self.logger.error(f"[EnhancedMaskEditor] Error during enhanced cleanup: {e}")

def load_video_file(app):
    """Load video file, image sequence, or single image and return (frames, source_path, source_type)"""
    from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
    import glob
    import time
    
    start_time = time.time()
    print(f"[Load Video] Starting video load process...")
    
    # Ask user what type of input they want to load
    msg = QMessageBox()
    msg.setWindowTitle("Select Input Type")
    msg.setText("What would you like to load?")
    
    video_btn = msg.addButton("Video File", QMessageBox.ActionRole)
    sequence_btn = msg.addButton("Image Sequence", QMessageBox.ActionRole)
    single_btn = msg.addButton("Single Image", QMessageBox.ActionRole)
    cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
    
    print(f"[Load Video] Showing input type dialog...")
    msg.exec_()
    print(f"[Load Video] Input type dialog closed")
    clicked = msg.clickedButton()
    
    if clicked == cancel_btn:
        return None, None, None
    
    frames = []
    
    if clicked == video_btn:
        # Load video file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        
        if file_path:
            print(f"[Load Video] Opening video file: {file_path}")
            load_start = time.time()
            
            # First get video info
            cap = cv2.VideoCapture(file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"[Load Video] Video info: {total_frames} frames, {fps} fps, {width}x{height}")
            
            # Create progress dialog
            progress = QProgressDialog("Loading video frames...", "Cancel", 0, total_frames)
            progress.setWindowTitle("Loading Video")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(1000)  # Only show if it takes more than 1 second
            progress.setValue(0)  # Force show immediately
            
            frame_count = 0
            last_update = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret or progress.wasCanceled():
                    break
                    
                # Keep as BGR - the mask editor expects BGR format
                frames.append(frame)
                frame_count += 1
                
                # Update progress every 10 frames or 0.1 seconds
                if frame_count % 10 == 0 or (time.time() - last_update) > 0.1:
                    progress.setValue(frame_count)
                    progress.setLabelText(f"Loading frame {frame_count}/{total_frames}...")
                    QApplication.processEvents()
                    last_update = time.time()
                    
                    if frame_count % 50 == 0:
                        elapsed = time.time() - load_start
                        fps_load = frame_count / elapsed
                        eta = (total_frames - frame_count) / fps_load
                        print(f"[Load Video] Progress: {frame_count}/{total_frames} frames ({fps_load:.1f} fps, ETA: {eta:.1f}s)")
            
            cap.release()
            progress.close()
            
            load_time = time.time() - load_start
            print(f"[Load Video] Loaded {len(frames)} frames in {load_time:.2f} seconds ({len(frames)/load_time:.1f} fps)")
            
            # Return frames with source info
            if frames:
                return frames, video_path, "video"
    
    elif clicked == sequence_btn:
        # Load image sequence
        choice = QMessageBox.question(
            None,
            "Image Sequence",
            "Load image sequence from:\n\nYes = Select multiple files\nNo = Select folder",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if choice == QMessageBox.Yes:
            # Select multiple files
            file_paths, _ = QFileDialog.getOpenFileNames(
                None,
                "Select Image Files (in sequence order)",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
            )
            
            if file_paths:
                # Sort files naturally
                file_paths.sort()
                # Use parallel loading for selected files
                max_workers = min(8, len(file_paths))
                frames_ordered = [None] * len(file_paths)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {
                        executor.submit(load_single_image_mask_editor, path): idx 
                        for idx, path in enumerate(file_paths)
                    }
                    
                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            result = future.result()
                            if result is not None:
                                frames_ordered[idx] = result
                        except Exception as e:
                            print(f"[Mask Editor Loading] Error loading {file_paths[idx]}: {e}")
                
                # Remove None values and add to frames
                frames = [f for f in frames_ordered if f is not None]
                print(f"[ComfyUI Mask Editor] Loaded {len(frames)} frames from selected images")
                
                # Return frames with source info - use parent directory
                if frames:
                    return frames, os.path.dirname(file_paths[0]), "image_sequence"
        else:
            # Select folder
            folder_path = QFileDialog.getExistingDirectory(
                None,
                "Select Folder Containing Image Sequence"
            )
            
            if folder_path:
                # Use optimized parallel loading
                frames = load_image_sequence_optimized_mask_editor(folder_path)
                print(f"[ComfyUI Mask Editor] Loaded {len(frames)} frames from folder")
                
                # Return frames with source info
                if frames:
                    return frames, folder_path, "image_sequence"
    
    elif clicked == single_btn:
        # Load single image
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Single Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                # Keep as BGR - the mask editor expects BGR format
                frames.append(img)
                print(f"[ComfyUI Mask Editor] Loaded single image")
                return frames, file_path, "single_image"
    
    total_time = time.time() - start_time
    print(f"[Load Video] Total load process took {total_time:.2f} seconds")
    return (frames, None, None) if frames else (None, None, None)

def main():
    print("[MASK EDITOR LAUNCHER] Entered main() function")
    
    parser = argparse.ArgumentParser(description="ComfyUI Mask Editor")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    args = parser.parse_args()
    
    print(f"[MASK EDITOR LAUNCHER] Config file: {args.config}")
    
    # Import QTimer here for later use
    from PyQt5.QtCore import QTimer
    
    # Set up logging to file
    import logging
    log_file = os.path.join(os.path.dirname(args.config), "mask_editor_debug.log")
    print(f"[MASK EDITOR LAUNCHER] Log file will be: {log_file}")
    
    # Set up logging with optional file handler
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    except (OSError, IOError) as e:
        print(f"[MASK EDITOR LAUNCHER] Warning: Could not create log file: {e}")
        print("[MASK EDITOR LAUNCHER] Continuing with console output only")
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(message)s',
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info("Mask Editor Starting")
    logger.info(f"Executed script: {os.path.abspath(__file__)}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("="*50)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create Qt application FIRST
    app = QApplication(sys.argv)
    
    # Start with a blank frame - user will load content from within the editor
    frames = [np.zeros((512, 512, 3), dtype=np.uint8)]
    
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
            height: 6px;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #5a5a5a;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #6a6a6a;
        }
        
        /* Scrollbars */
        QScrollBar:vertical {
            background: #2b2b2b;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #4a4a4a;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background: #5a5a5a;
        }
        
        /* Frames and groups */
        QFrame, QGroupBox {
            background-color: #2b2b2b;
            border: 1px solid #3a3a3a;
            border-radius: 4px;
            padding: 4px;
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
            left: 10px;
            padding: 0 5px 0 5px;
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
    
    # Create and configure the enhanced mask editor
    editor = EnhancedMaskEditor(frames)
    editor.setWindowTitle("WAN2.1 VACE Mask Editor - ComfyUI")
    
    # Set output directory for session auto-save
    editor.output_dir = config.get("output_dir", "output")
    editor.config_path = args.config
    
    # Load previous project data if available
    project_data = config.get("project_data", "")
    logger.info(f"Project data present: {bool(project_data)}")
    print(f"[ComfyUI Mask Editor] Project data present: {bool(project_data)}")
    sys.stdout.flush()
    
    if project_data:
        logger.info(f"Project data type: {type(project_data)}")
        print(f"[ComfyUI Mask Editor] Project data type: {type(project_data)}")
        if isinstance(project_data, str):
            logger.info(f"Project data length: {len(project_data)} chars")
            print(f"[ComfyUI Mask Editor] Project data length: {len(project_data)} chars")
        sys.stdout.flush()
        try:
            project = json.loads(project_data) if isinstance(project_data, str) else project_data
            print(f"[ComfyUI Mask Editor] Successfully parsed project data")
            
            # Load video frames for session restore
            loaded_frames = []
            
            # Debug: print what's in the project
            logger.info(f"Project keys: {list(project.keys())}")
            print(f"[ComfyUI Mask Editor] Project keys: {list(project.keys())}")
            if "source_video" in project:
                logger.info(f"source_video content: {project['source_video']}")
                print(f"[ComfyUI Mask Editor] source_video content: {project['source_video']}")
            # Check for video_info (from ComfyUI node)
            if "video_info" in project:
                video_info = project["video_info"]
                logger.info(f"video_info content: {video_info}")
                print(f"[ComfyUI Mask Editor] video_info content: {video_info}")
                # Set source path on editor from video_info
                if video_info.get("path"):
                    editor.source_video_path = video_info["path"]
                    editor.source_type = video_info.get("type", "video")
                    logger.info(f"Set source_video_path from video_info: {editor.source_video_path}")
                    print(f"[ComfyUI Mask Editor] Set source_video_path from video_info: {editor.source_video_path}")
                    # Trigger reload check after setting source path
                    QTimer.singleShot(200, editor.check_and_reload_video)
            
            # First try to load from source video path (new format)
            if "source_video" in project and project["source_video"].get("path"):
                source_path = project["source_video"]["path"]
                source_type = project["source_video"]["type"]
                print(f"[ComfyUI Mask Editor] Loading video from source: {source_path} (type: {source_type})")
            # Also check for flat format from auto-save
            elif "source_video_path" in project and project["source_video_path"]:
                source_path = project["source_video_path"]
                source_type = project.get("source_type", "video")
                print(f"[ComfyUI Mask Editor] Loading video from auto-save source: {source_path} (type: {source_type})")
            else:
                source_path = None
                source_type = None
                print(f"[ComfyUI Mask Editor] No source video path found in project data")
                print(f"[ComfyUI Mask Editor] Project keys: {list(project.keys())}")
            
            # Store source info in editor IMMEDIATELY after determining it
            # This ensures the editor has the source path even if video loading fails
            if source_path:
                # Check if source file exists before setting
                if not os.path.exists(source_path):
                    print(f"[ComfyUI Mask Editor] Source file not found during session load: {source_path}")
                    new_path = prompt_for_missing_file(source_path, source_type, editor)
                    if new_path:
                        print(f"[ComfyUI Mask Editor] User selected new path during session load: {new_path}")
                        source_path = new_path
                        # Update the project data with new path
                        if "source_video" in project and isinstance(project["source_video"], dict):
                            project["source_video"]["path"] = new_path
                        elif "source_video_path" in project:
                            project["source_video_path"] = new_path
                        # Also update video_info if present
                        if "video_info" in project and isinstance(project["video_info"], dict):
                            project["video_info"]["path"] = new_path
                    else:
                        print(f"[ComfyUI Mask Editor] User cancelled file browse during session load")
                        source_path = None
                
                if source_path:
                    editor.source_video_path = source_path
                    editor.source_type = source_type
                    print(f"[ComfyUI Mask Editor] Set source video path on editor: {source_path} (type: {source_type})")
                    # Trigger reload check after setting source path
                    QTimer.singleShot(200, editor.check_and_reload_video)
                
            if source_path and source_type == "video" and os.path.exists(source_path):
                # Load video
                cap = cv2.VideoCapture(source_path)
                if cap.isOpened():
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Keep frames in BGR format (OpenCV default)
                        loaded_frames.append(frame)
                    cap.release()
                    print(f"[ComfyUI Mask Editor] Loaded {len(loaded_frames)} frames from video: {source_path}")
            elif source_path and source_type == "image_sequence" and os.path.exists(source_path):
                # Use optimized parallel loading
                print(f"[ComfyUI Mask Editor] Loading image sequence from folder: {source_path}")
                loaded_frames = load_image_sequence_optimized_mask_editor(source_path)
                if loaded_frames and len(loaded_frames) > 0:
                    print(f"[ComfyUI Mask Editor] First image shape: {loaded_frames[0].shape}")
                print(f"[ComfyUI Mask Editor] Loaded {len(loaded_frames)} frames from image sequence: {source_path}")
            else:
                if source_path:
                    print(f"[ComfyUI Mask Editor] Warning: Could not load video from source: {source_path}")
                    if source_path and not os.path.exists(source_path):
                        print(f"[ComfyUI Mask Editor] File does not exist: {source_path}")
                    elif source_type not in ["video", "image_sequence"]:
                        print(f"[ComfyUI Mask Editor] Unknown source type: {source_type}")
            
            # Try to load from base64 data (old format)
            if "video_frames_data" in project and project["video_frames_data"] and not loaded_frames:
                import time
                print(f"[ComfyUI Mask Editor] Loading video from base64 data...")
                frame_data_start = time.time()
                
                frame_data_list = project["video_frames_data"]
                for i, encoded_frame in enumerate(frame_data_list):
                    try:
                        # Decode base64 back to image
                        buffer = base64.b64decode(encoded_frame)
                        nparr = np.frombuffer(buffer, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            loaded_frames.append(frame)
                        
                        if i % 100 == 0:
                            print(f"[ComfyUI Mask Editor] Decoded frame {i}/{len(frame_data_list)}")
                    except Exception as e:
                        print(f"[ComfyUI Mask Editor] Error decoding frame {i}: {e}")
                
                print(f"[ComfyUI Mask Editor] Frame decoding took {time.time() - frame_data_start:.2f}s")
                
            # Fallback to old file-based loading
            if "frame_files" in project and editor.output_dir and not loaded_frames:
                frame_files = project.get("frame_files", [])
                
                for frame_file in frame_files:
                    frame_path = os.path.join(editor.output_dir, frame_file)
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        if frame is not None:
                            loaded_frames.append(frame)
            
            if loaded_frames:
                print(f"[ComfyUI Mask Editor] Loaded {len(loaded_frames)} frames from session")
                editor.reinitialize_with_frames(loaded_frames)
            else:
                print(f"[ComfyUI Mask Editor] No video frames found in session data")
                # If we have a source path but no frames, just reload the fucking video
                if hasattr(editor, 'source_video_path') and editor.source_video_path and os.path.exists(editor.source_video_path):
                    print(f"[ComfyUI Mask Editor] Reloading video from: {editor.source_video_path}")
                    reload_frames = []
                    if editor.source_type == "video":
                        cap = cv2.VideoCapture(editor.source_video_path)
                        if cap.isOpened():
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                reload_frames.append(frame)
                            cap.release()
                    if reload_frames:
                        editor.current_frames = reload_frames
                        editor.reinitialize_with_frames(reload_frames)
                        print(f"[ComfyUI Mask Editor] Reloaded {len(reload_frames)} frames")
                    else:
                        print(f"[ComfyUI Mask Editor] Failed to reload video")
            
            # Restore shape keyframes with proper type conversion
            if "shape_keyframes" in project:
                logger.info(f"Found shape_keyframes in project data")
                print(f"[ComfyUI Mask Editor] Found shape_keyframes in project data")
                sys.stdout.flush()
                # Convert string keys to integers
                shape_keyframes = {}
                for frame_str, shapes in project["shape_keyframes"].items():
                    frame_num = int(frame_str)
                    shape_keyframes[frame_num] = shapes
                
                print(f"[ComfyUI Mask Editor] Loaded {len(shape_keyframes)} keyframes: {list(shape_keyframes.keys())}")
                
                # Set the shape keyframes in the mask widget
                if hasattr(editor, 'mask_widget'):
                    print(f"[ComfyUI Mask Editor] Setting shape_keyframes on mask_widget")
                    editor.mask_widget.shape_keyframes = shape_keyframes
                    editor.mask_widget.update_mask_from_shapes()
                    print(f"[ComfyUI Mask Editor] mask_widget.shape_keyframes now has {len(editor.mask_widget.shape_keyframes)} frames")
                    
                    # Force immediate timeline update
                    if hasattr(editor, 'update_mask_frame_tracking'):
                        print(f"[ComfyUI Mask Editor] Calling update_mask_frame_tracking immediately")
                        # Store current mode and temporarily set to shape
                        original_mode = getattr(editor, 'drawing_mode', None)
                        editor.drawing_mode = "shape"
                        editor.update_mask_frame_tracking()
                        if original_mode and original_mode != "shape":
                            editor.drawing_mode = original_mode
                    
                    # Timeline will be updated later after all data is loaded
                else:
                    print(f"[ComfyUI Mask Editor] ERROR: editor.mask_widget not found!")
            
            # Restore other session data
            if "current_frame" in project:
                editor.current_frame_index = project["current_frame"]
                if hasattr(editor, 'on_frame_changed'):
                    editor.on_frame_changed(editor.current_frame_index)
                    
            if "drawing_mode" in project and hasattr(editor, 'set_drawing_mode'):
                editor.set_drawing_mode(project["drawing_mode"])
                
            if "current_project_path" in project:
                editor.current_project_path = project["current_project_path"]
            
            print("[ComfyUI Mask Editor] Loaded project data and restored session")
            
            # DON'T update timeline here - save it for after the dialog is shown
            print("[ComfyUI Mask Editor] Deferring timeline update until after dialog is shown")
            
            # Store the keyframes for later
            editor._restored_keyframes = {}
            if hasattr(editor.mask_widget, 'shape_keyframes'):
                editor._restored_keyframes = editor.mask_widget.shape_keyframes.copy()
            
            # Set flag for update after shown
            editor._needs_timeline_update = True
                
        except Exception as e:
            print(f"[ComfyUI Mask Editor] Failed to load project data: {e}")
    
    # Set up the timeline fix if needed
    needs_timeline_fix = hasattr(editor, '_needs_timeline_update') and editor._needs_timeline_update
    logger.info(f"needs_timeline_fix: {needs_timeline_fix}")
    print(f"[ComfyUI Mask Editor] needs_timeline_fix: {needs_timeline_fix}")
    sys.stdout.flush()
    
    # Show and exec the dialog
    editor.show()
    
    # Check if we need to reload the video after showing
    print(f"[ComfyUI Mask Editor] Checking video reload: has source_video_path={hasattr(editor, 'source_video_path')}, source_video_path={getattr(editor, 'source_video_path', None)}")
    print(f"[ComfyUI Mask Editor] Current frames: {len(editor.current_frames) if hasattr(editor, 'current_frames') and editor.current_frames else 'None or empty'}")
    
    if hasattr(editor, 'source_video_path') and editor.source_video_path and (not hasattr(editor, 'current_frames') or not editor.current_frames):
        print(f"[ComfyUI Mask Editor] No frames loaded but have source path, reloading video...")
        def reload_video():
            frames = []
            if editor.source_type == "video":
                cap = cv2.VideoCapture(editor.source_video_path)
                if cap.isOpened():
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    cap.release()
                    if frames:
                        print(f"[ComfyUI Mask Editor] Reloaded {len(frames)} frames")
                        editor.current_frames = frames
                        editor.reinitialize_with_frames(frames)
        QTimer.singleShot(100, reload_video)
    
    # If we need to fix the timeline, do it after the dialog is shown
    if needs_timeline_fix:
        logger.info("Setting up timeline fix timer...")
        print(f"[ComfyUI Mask Editor] Setting up timeline fix timer...")
        sys.stdout.flush()
        from PyQt5.QtCore import QTimer
        
        def do_timeline_fix():
            print(f"[ComfyUI Mask Editor] === TIMELINE FIX START ===")
            print(f"[ComfyUI Mask Editor] Has update_mask_frame_tracking: {hasattr(editor, 'update_mask_frame_tracking')}")
            print(f"[ComfyUI Mask Editor] Has timeline_widget: {hasattr(editor, 'timeline_widget')}")
            print(f"[ComfyUI Mask Editor] Has mask_widget: {hasattr(editor, 'mask_widget')}")
            
            if hasattr(editor, 'mask_widget') and hasattr(editor.mask_widget, 'shape_keyframes'):
                print(f"[ComfyUI Mask Editor] mask_widget.shape_keyframes has {len(editor.mask_widget.shape_keyframes)} frames: {list(editor.mask_widget.shape_keyframes.keys())}")
            
            # First, call the proper update method if it exists
            if hasattr(editor, 'update_mask_frame_tracking'):
                print(f"[ComfyUI Mask Editor] Current drawing mode: {getattr(editor, 'drawing_mode', 'unknown')}")
                print(f"[ComfyUI Mask Editor] Calling update_mask_frame_tracking()")
                
                # Store current mode
                original_mode = getattr(editor, 'drawing_mode', None)
                
                # If we have shape keyframes, temporarily set mode to shape to ensure they show
                if hasattr(editor.mask_widget, 'shape_keyframes') and editor.mask_widget.shape_keyframes:
                    print(f"[ComfyUI Mask Editor] Temporarily setting mode to 'shape' to show keyframes")
                    editor.drawing_mode = "shape"
                
                editor.update_mask_frame_tracking()
                
                # Restore original mode if we changed it
                if original_mode is not None and editor.drawing_mode != original_mode:
                    editor.drawing_mode = original_mode
                    print(f"[ComfyUI Mask Editor] Restored drawing mode to: {original_mode}")
                
                # Check the result
                if hasattr(editor, 'timeline_widget') and hasattr(editor.timeline_widget, 'mask_frames'):
                    print(f"[ComfyUI Mask Editor] After update: timeline mask_frames = {editor.timeline_widget.mask_frames}")
            elif hasattr(editor, 'timeline_widget') and hasattr(editor.mask_widget, 'shape_keyframes'):
                # Fallback: manually update if the method doesn't exist
                print(f"[ComfyUI Mask Editor] Manual timeline update (fallback)")
                # Set the mask frames
                editor.timeline_widget.mask_frames = set()
                for frame_idx in editor.mask_widget.shape_keyframes:
                    if editor.mask_widget.shape_keyframes[frame_idx]:
                        editor.timeline_widget.mask_frames.add(frame_idx)
                
                print(f"[ComfyUI Mask Editor] Timeline mask_frames = {editor.timeline_widget.mask_frames}")
                
                # Force a repaint
                editor.timeline_widget.update()
                
                # 4. Force parent to update
                parent = editor.timeline_widget.parent()
                if parent:
                    parent.update()
                
                # 5. Process all events
                QApplication.processEvents()
                
                print(f"[ComfyUI Mask Editor] Forced paint event complete")
            else:
                print(f"[ComfyUI Mask Editor] WARNING: No valid timeline data to update!")
            
            print(f"[ComfyUI Mask Editor] === TIMELINE FIX END ===")
        
        # Execute after a longer delay to ensure dialog is FULLY ready
        QTimer.singleShot(3000, do_timeline_fix)  # 3 seconds should be enough
    
    result = editor.exec_()
    
    if result == editor.Accepted:
        import time
        save_start = time.time()
        print(f"[ComfyUI Mask Editor] Starting save process...")
        
        output_dir = config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save both original frames and masks from the enhanced editor
        frames_start = time.time()
        current_frames = editor.get_frames_for_export()
        print(f"[ComfyUI Mask Editor] get_frames_for_export took {time.time() - frames_start:.2f}s")
        
        masks = []
        saved_keyframes = []
        
        if hasattr(editor, 'mask_widget'):
            # Save current state
            current_frame = editor.mask_widget.current_frame
            
            # Save a reference frame for ComfyUI
            print(f"[ComfyUI Mask Editor] Saving reference frame...")
            
            # Save just the first frame as reference
            if len(current_frames) > 0:
                frame_path = os.path.join(output_dir, "frame_0000.png")
                cv2.imwrite(frame_path, current_frames[0])
                saved_frames = ["frame_0000.png"]
            
            # Get keyframes with shapes
            keyframes = []
            if hasattr(editor.mask_widget, 'shape_keyframes'):
                keyframes = list(editor.mask_widget.shape_keyframes.keys())
            
            print(f"[ComfyUI Mask Editor] Found {len(keyframes)} keyframes with shapes: {keyframes}")
            
            # Save a dummy mask file so server knows we succeeded
            mask_path = os.path.join(output_dir, "mask_0000.png")
            if len(current_frames) > 0:
                dummy_mask = np.zeros((current_frames[0].shape[0], current_frames[0].shape[1]), dtype=np.uint8)
                cv2.imwrite(mask_path, dummy_mask)
            saved_masks = ["mask_0000.png"]
            saved_keyframes = keyframes
            
            # Restore current frame
            editor.mask_widget.current_frame = current_frame
        
        # Save project data - Only save essential data
        project_data = {
            "shape_keyframes": {},
            "settings": {
                "drawing_mode": editor.drawing_mode if hasattr(editor, 'drawing_mode') else "brush",
                "brush_size": getattr(editor, 'brush_size', 30),
            },
            "video_info": {
                "path": editor.source_video_path if hasattr(editor, 'source_video_path') else None,
                "type": editor.source_type if hasattr(editor, 'source_type') else None,
                "total_frames": len(current_frames),
                "width": current_frames[0].shape[1] if current_frames else 512,
                "height": current_frames[0].shape[0] if current_frames else 512,
            },
            "current_frame": editor.current_frame_index,
        }
        
        if hasattr(editor, 'mask_widget') and hasattr(editor.mask_widget, 'shape_keyframes'):
            # Convert integer keys to strings for JSON
            shapes_serialize_start = time.time()
            for frame, shapes in editor.mask_widget.shape_keyframes.items():
                project_data["shape_keyframes"][str(frame)] = shapes
            print(f"[ComfyUI Mask Editor] Saved {len(editor.mask_widget.shape_keyframes)} shape keyframes")
            if editor.mask_widget.shape_keyframes:
                first_frame = list(editor.mask_widget.shape_keyframes.keys())[0]
                print(f"[ComfyUI Mask Editor] Example - Frame {first_frame}: {len(editor.mask_widget.shape_keyframes[first_frame])} shapes")
            print(f"[ComfyUI Mask Editor] Shape serialization took {time.time() - shapes_serialize_start:.3f}s")
        
        project_save_start = time.time()
        project_file = os.path.join(output_dir, "project_data.json")
        with open(project_file, 'w') as f:
            json.dump(project_data, f, indent=2)
        print(f"[ComfyUI Mask Editor] Project data save took {time.time() - project_save_start:.3f}s")

        # Apply working changes to current state (user accepted)
        editor.apply_working_to_current_state()  # Copies working_autosave.json → current_state.json

        total_save_time = time.time() - save_start
        print(f"[ComfyUI Mask Editor] TOTAL SAVE TIME: {total_save_time:.2f}s")
        print(f"[ComfyUI Mask Editor] Saved shape keyframes data and applied to current state")
        return 0
    else:
        print("[ComfyUI Mask Editor] Cancelled - cleaning up session data")
        # User cancelled - clean up any auto-saved session data
        if hasattr(editor, 'clean_up_enhanced_session_data'):
            editor.clean_up_enhanced_session_data()
        return 1

if __name__ == "__main__":
    sys.exit(main())