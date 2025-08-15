"""
OutpaintingPreviewWidget - Exact copy from wan21_vace_prepsuite.py
"""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, QRectF
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPixmap, QImage
import cv2
import numpy as np


class OutpaintingPreviewWidget(QWidget):
    frameChanged = pyqtSignal(int)  # Signal emitted when frame changes
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        
        # Video properties
        self.video_frame = None
        self.video_frames = []  # Store all frames for playback
        self.current_frame_index = 0
        self.video_width = 0
        self.video_height = 0
        self.video_fps = 30
        self.is_playing = False
        self.frames_are_rgb = True  # Track color format - default to RGB
        
        # Canvas properties
        self.canvas_width = 0
        self.canvas_height = 0
        self.video_x = 0
        self.video_y = 0
        
        # Interaction properties
        self.dragging = False
        self.drag_handle = None  # Which handle is being dragged
        self.drag_start = None
        self.aspect_locked = True
        self.original_aspect = 1.0
        self.preset_aspect = None  # Aspect ratio from preset selection
        
        # View properties
        self.show_mask = False
        self.feather_amount = 0
        
        # Cropping properties
        self.allow_cropping = False
        
        # Handle size
        self.handle_size = 8
        
        # Playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.next_frame)
        
        self.setMouseTracking(True)
        
    def setVideoFrame(self, frame):
        """Set the video frame to display"""
        if frame is not None:
            self.video_height, self.video_width = frame.shape[:2]
            self.video_frame = frame
            self.original_aspect = self.video_width / self.video_height
            
            # Initialize canvas to video size
            self.canvas_width = int(self.video_width)
            self.canvas_height = int(self.video_height)
            self.video_x = 0
            self.video_y = 0
            
            self.update()
            
    def setVideoFrames(self, frames, fps=30):
        """Set all video frames for playback"""
        self.video_frames = frames
        self.video_fps = fps
        self.current_frame_index = 0
        self.frames_are_rgb = True  # Frames loaded in launcher are RGB
        if frames:
            self.setVideoFrame(frames[0])
            
    def set_frames(self, frames, fps=30):
        """Alternative method name for compatibility"""
        self.setVideoFrames(frames, fps)
            
    def setCurrentFrame(self, index):
        """Set the current frame by index"""
        if 0 <= index < len(self.video_frames):
            self.current_frame_index = index
            self.video_frame = self.video_frames[index]
            self.update()
            self.frameChanged.emit(index)
            
    def play(self):
        """Start video playback"""
        if self.video_frames:
            self.is_playing = True
            interval = int(1000 / self.video_fps)
            self.playback_timer.start(interval)
            
    def pause(self):
        """Pause video playback"""
        self.is_playing = False
        self.playback_timer.stop()
        
    def stop(self):
        """Stop video playback and reset to first frame"""
        self.pause()
        self.setCurrentFrame(0)
        
    def next_frame(self):
        """Advance to next frame"""
        if self.video_frames:
            next_index = (self.current_frame_index + 1) % len(self.video_frames)
            self.setCurrentFrame(next_index)
            
    def setAspectLocked(self, locked):
        """Set whether aspect ratio should be locked"""
        self.aspect_locked = locked
        # Clear preset aspect when manually toggling aspect lock
        if not locked:
            self.preset_aspect = None
            
    def setPresetAspect(self, aspect_ratio):
        """Set the aspect ratio from a preset selection"""
        self.preset_aspect = aspect_ratio
        if aspect_ratio is not None:
            self.aspect_locked = True
        
    def setShowMask(self, show_mask):
        """Set whether to show mask preview"""
        self.show_mask = show_mask
        self.update()
        
    def setFeatherAmount(self, amount):
        """Set the feather amount in pixels"""
        self.feather_amount = amount
        self.update()
        
    def setAllowCropping(self, allow):
        """Set whether cropping is allowed"""
        self.allow_cropping = allow
        if not allow:
            # Snap back to original video boundaries
            self.canvas_width = max(self.canvas_width, self.video_width)
            self.canvas_height = max(self.canvas_height, self.video_height)
            self.video_x = max(0, min(self.video_x, self.canvas_width - self.video_width))
            self.video_y = max(0, min(self.video_y, self.canvas_height - self.video_height))
            self.update()
        
    def createMask(self, width, height):
        """Create a mask with proper inward feathering"""
        # Ensure all dimensions are integers
        video_x = int(self.video_x)
        video_y = int(self.video_y)
        video_width = int(self.video_width)
        video_height = int(self.video_height)
        width = int(width)
        height = int(height)
        
        # Create base mask (white for padding, black for video)
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Calculate the intersection between video and canvas
        # This handles cases where video is partially outside canvas (negative video_x/y)
        x1 = max(0, video_x)
        y1 = max(0, video_y)
        x2 = min(width, video_x + video_width)
        y2 = min(height, video_y + video_height)
        
        # Only set black region if there's a valid intersection
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
        
        # Apply feathering if needed
        if self.feather_amount > 0 and x2 > x1 and y2 > y1:
            # Create a distance transform from the inside of the video region
            # This gives us the distance from each pixel to the nearest edge
            video_region = np.zeros((height, width), dtype=np.uint8)
            video_region[y1:y2, x1:x2] = 255
            
            # Calculate distance from edges inside the video region
            dist_transform = cv2.distanceTransform(video_region, cv2.DIST_L2, 5)
            
            # Normalize and clip the distance transform to create feather
            feather_pixels = self.feather_amount
            feather_mask = np.clip(dist_transform / feather_pixels, 0, 1)
            
            # Apply a power curve for faster falloff (higher power = faster falloff)
            # Using power of 3 for a nice curve that starts slow then drops fast
            feather_mask = np.power(feather_mask, 3)
            
            # Invert so edges are white (masked) and center is black (video)
            feather_mask = 1.0 - feather_mask
            
            # Apply the feather only to the video region
            final_mask = np.ones((height, width), dtype=np.float32)
            final_mask[y1:y2, x1:x2] = feather_mask[y1:y2, x1:x2]
            
            # Convert back to uint8
            mask = (final_mask * 255).astype(np.uint8)
            
        return mask
        
    def getCanvasSize(self):
        """Get the current canvas size and video position"""
        return {
            'canvas_width': int(self.canvas_width),
            'canvas_height': int(self.canvas_height),
            'video_x': int(self.video_x),
            'video_y': int(self.video_y),
            'video_width': int(self.video_width),
            'video_height': int(self.video_height),
            'feather_amount': int(self.feather_amount)
        }
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.video_frame is None:
            # Draw placeholder
            painter.fillRect(self.rect(), QColor(50, 50, 50))
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Load a video to preview")
            return
            
        # Calculate scale to fit widget
        if self.canvas_width == 0 or self.canvas_height == 0:
            return
        scale = min(self.width() / self.canvas_width, self.height() / self.canvas_height) * 0.9
        
        # Canvas position in widget
        canvas_display_width = self.canvas_width * scale
        canvas_display_height = self.canvas_height * scale
        canvas_x = (self.width() - canvas_display_width) / 2
        canvas_y = (self.height() - canvas_display_height) / 2
        
        # Draw black background
        painter.fillRect(self.rect(), Qt.black)
        
        # Draw canvas area
        painter.fillRect(QRect(int(canvas_x), int(canvas_y), 
                              int(canvas_display_width), int(canvas_display_height)), 
                        QColor(30, 30, 30))
        
        # Draw video frame or mask
        if self.video_frame is not None:
            video_display_x = canvas_x + self.video_x * scale
            video_display_y = canvas_y + self.video_y * scale
            video_display_width = self.video_width * scale
            video_display_height = self.video_height * scale
            
            if self.show_mask:
                # Create and show mask
                mask = self.createMask(self.canvas_width, self.canvas_height)
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                h, w, ch = mask_rgb.shape
                bytes_per_line = ch * w
                qt_image = QPixmap.fromImage(
                    QImage(mask_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                )
                
                # Draw scaled mask
                painter.drawPixmap(int(canvas_x), int(canvas_y),
                                 int(canvas_display_width), int(canvas_display_height),
                                 qt_image)
            else:
                # Convert frame to QPixmap (check if conversion needed)
                if self.frames_are_rgb:
                    frame_rgb = self.video_frame  # Already RGB
                else:
                    frame_rgb = cv2.cvtColor(self.video_frame, cv2.COLOR_BGR2RGB)
                    
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_image = QPixmap.fromImage(
                    QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                )
                
                # Draw scaled video
                painter.drawPixmap(int(video_display_x), int(video_display_y),
                                 int(video_display_width), int(video_display_height),
                                 qt_image)
            
        # Draw canvas border
        painter.setPen(QPen(Qt.white, 2, Qt.DashLine))
        painter.drawRect(QRect(int(canvas_x), int(canvas_y), 
                              int(canvas_display_width), int(canvas_display_height)))
        
        # Draw resize handles
        painter.setPen(QPen(Qt.white, 2))
        painter.setBrush(QBrush(Qt.white))
        
        # Corner handles
        handles = [
            ('tl', canvas_x, canvas_y),
            ('tr', canvas_x + canvas_display_width, canvas_y),
            ('bl', canvas_x, canvas_y + canvas_display_height),
            ('br', canvas_x + canvas_display_width, canvas_y + canvas_display_height),
            ('t', canvas_x + canvas_display_width/2, canvas_y),
            ('b', canvas_x + canvas_display_width/2, canvas_y + canvas_display_height),
            ('l', canvas_x, canvas_y + canvas_display_height/2),
            ('r', canvas_x + canvas_display_width, canvas_y + canvas_display_height/2)
        ]
        
        for handle_id, x, y in handles:
            rect = QRect(int(x - self.handle_size/2), int(y - self.handle_size/2),
                        self.handle_size, self.handle_size)
            painter.drawRect(rect)
            
        # Draw info
        painter.setPen(Qt.white)
        info_text = f"Canvas: {self.canvas_width}x{self.canvas_height} | Video: {self.video_width}x{self.video_height}"
        if self.aspect_locked:
            info_text += " | Aspect Locked"
        if self.show_mask:
            info_text += " | MASK VIEW"
            if self.feather_amount > 0:
                info_text += f" | Feather: {self.feather_amount}px"
        painter.drawText(10, 20, info_text)
        
    def getHandleAt(self, pos):
        """Get which handle is at the given position"""
        if self.canvas_width == 0 or self.canvas_height == 0:
            return None
            
        # Calculate scale
        scale = min(self.width() / self.canvas_width, self.height() / self.canvas_height) * 0.9
        
        canvas_display_width = self.canvas_width * scale
        canvas_display_height = self.canvas_height * scale
        canvas_x = (self.width() - canvas_display_width) / 2
        canvas_y = (self.height() - canvas_display_height) / 2
        
        handles = [
            ('tl', canvas_x, canvas_y),
            ('tr', canvas_x + canvas_display_width, canvas_y),
            ('bl', canvas_x, canvas_y + canvas_display_height),
            ('br', canvas_x + canvas_display_width, canvas_y + canvas_display_height),
            ('t', canvas_x + canvas_display_width/2, canvas_y),
            ('b', canvas_x + canvas_display_width/2, canvas_y + canvas_display_height),
            ('l', canvas_x, canvas_y + canvas_display_height/2),
            ('r', canvas_x + canvas_display_width, canvas_y + canvas_display_height/2)
        ]
        
        for handle_id, x, y in handles:
            if abs(pos.x() - x) < self.handle_size and abs(pos.y() - y) < self.handle_size:
                return handle_id
        return None
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            handle = self.getHandleAt(event.pos())
            if handle:
                self.dragging = True
                self.drag_handle = handle
                self.drag_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                
    def mouseMoveEvent(self, event):
        if not self.dragging:
            handle = self.getHandleAt(event.pos())
            if handle:
                if handle in ['tl', 'br']:
                    self.setCursor(Qt.SizeFDiagCursor)
                elif handle in ['tr', 'bl']:
                    self.setCursor(Qt.SizeBDiagCursor)
                elif handle in ['t', 'b']:
                    self.setCursor(Qt.SizeVerCursor)
                elif handle in ['l', 'r']:
                    self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        else:
            # Handle dragging
            if self.canvas_width == 0 or self.canvas_height == 0:
                return
            scale = min(self.width() / self.canvas_width, self.height() / self.canvas_height) * 0.9
            dx = (event.x() - self.drag_start.x()) / scale
            dy = (event.y() - self.drag_start.y()) / scale
            
            new_canvas_width = self.canvas_width
            new_canvas_height = self.canvas_height
            new_video_x = self.video_x
            new_video_y = self.video_y
            
            # Update based on which handle is dragged
            if 'l' in self.drag_handle:
                new_canvas_width -= dx
                # When dragging left edge, canvas expands leftward
                # So we need to move the video right by the same amount to keep it in place
                new_video_x -= dx
            if 'r' in self.drag_handle:
                new_canvas_width += dx
            if 't' in self.drag_handle:
                new_canvas_height -= dy
                # When dragging top edge, canvas expands upward
                # So we need to move the video down by the same amount to keep it in place
                new_video_y -= dy
            if 'b' in self.drag_handle:
                new_canvas_height += dy
                
            # Apply aspect ratio lock if needed
            if self.aspect_locked:
                # Use preset aspect ratio if available, otherwise use original
                target_aspect = self.preset_aspect if self.preset_aspect else self.original_aspect
                
                # Store original positions before aspect ratio adjustments
                original_video_x = new_video_x
                original_video_y = new_video_y
                
                if self.drag_handle in ['tl', 'tr', 'bl', 'br']:
                    # Corner handles - maintain aspect ratio based on larger movement
                    if abs(dx) > abs(dy):
                        new_canvas_height = new_canvas_width / target_aspect
                    else:
                        new_canvas_width = new_canvas_height * target_aspect
                elif self.drag_handle in ['l', 'r', 't', 'b']:
                    # Edge handles - maintain aspect ratio when locked
                    if self.drag_handle in ['l', 'r']:
                        # Horizontal edge - adjust height to maintain ratio
                        old_canvas_height = self.canvas_height
                        new_canvas_height = new_canvas_width / target_aspect
                        # Always distribute the height change equally above and below
                        # This ensures the canvas expands symmetrically
                        height_change = new_canvas_height - old_canvas_height
                        new_video_y = self.video_y + (height_change / 2)
                    else:
                        # Vertical edge - adjust width to maintain ratio
                        old_canvas_width = self.canvas_width
                        new_canvas_width = new_canvas_height * target_aspect
                        # Always distribute the width change equally left and right
                        # This ensures the canvas expands symmetrically
                        width_change = new_canvas_width - old_canvas_width
                        new_video_x = self.video_x + (width_change / 2)
                
                # Note: We removed the position restoration for cropping mode
                # because it was preventing proper symmetrical expansion when
                # using aspect ratio presets
                    
            # Apply constraints based on cropping mode
            if self.allow_cropping:
                # Allow canvas to be smaller than video (cropping)
                # Minimum canvas size is 10x10 pixels
                if new_canvas_width >= 10 and new_canvas_height >= 10:
                    self.canvas_width = int(new_canvas_width)
                    self.canvas_height = int(new_canvas_height)
                    # Allow video position to be negative for cropping
                    self.video_x = int(new_video_x)
                    self.video_y = int(new_video_y)
            else:
                # Ensure canvas is at least as big as video (no cropping)
                if new_canvas_width >= self.video_width and new_canvas_height >= self.video_height:
                    self.canvas_width = int(new_canvas_width)
                    self.canvas_height = int(new_canvas_height)
                    self.video_x = int(max(0, min(new_video_x, self.canvas_width - self.video_width)))
                    self.video_y = int(max(0, min(new_video_y, self.canvas_height - self.video_height)))
                
            self.drag_start = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.drag_handle = None
        self.setCursor(Qt.ArrowCursor)
        
    # Compatibility methods
    def set_frames(self, frames):
        """Compatibility method - redirects to setVideoFrames"""
        self.setVideoFrames(frames)
        
    def set_frame(self, index):
        """Compatibility method - redirects to setCurrentFrame"""
        self.setCurrentFrame(index)
        
    def set_canvas_size(self, width, height):
        """Set the canvas dimensions"""
        self.canvas_width = width
        self.canvas_height = height
        # Re-center video if it fits
        if self.video_width > 0 and self.video_height > 0:
            if self.video_width <= width and self.video_height <= height:
                self.video_x = (width - self.video_width) // 2
                self.video_y = (height - self.video_height) // 2
        self.update()
        
    def set_aspect_locked(self, locked):
        """Compatibility method - redirects to setAspectLocked"""
        self.setAspectLocked(locked)
        
    def set_allow_cropping(self, allow):
        """Compatibility method - redirects to setAllowCropping"""
        self.setAllowCropping(allow)
        
    def set_feather_amount(self, amount):
        """Compatibility method - redirects to setFeatherAmount"""
        self.setFeatherAmount(amount)