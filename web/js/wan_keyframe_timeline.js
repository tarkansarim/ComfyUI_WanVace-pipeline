import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// Complete Keyframe Timeline Widget with all PyQt5 features
class WANKeyframeTimelineWidget {
    constructor(node) {
        this.node = node;
        this.name = "keyframe_timeline";
        this.type = "custom";  // Important: must be "custom" for ComfyUI
        
        // Timeline properties
        this.keyframes = {}; // {frame: {path: string, enabled: bool, hold: int, id: string}} - NO IMAGE DATA HERE
        this.keyframeImages = {}; // {frame: base64} - Image data stored separately
        this.keyframeImagesByID = {}; // {id: base64} - Image data stored by unique ID for undo robustness
        this.timeline_frames = 81;
        this.selectedKeyframe = null;
        this.selectedKeyframes = new Set();
        this.isDragging = false;
        this.dragStartX = 0;
        this.dragStartFrame = 0;
        this.dragTargetFrame = 0;
        this.nextKeyframeID = 1; // Counter for unique IDs
        
        // Playback properties
        this.currentFrame = 0;
        this.isPlaying = false;
        this.fps = 15;
        this.lastPlayTime = 0;
        this.isScrubbing = false;
        
        // Visual properties
        this.height = 545;  // Increased to accommodate hold indicators
        this.margin = 15;
        this.hoverKeyframe = null;
        this.mousePos = {x: 0, y: 0};
        
        // CENTRALIZED LAYOUT SYSTEM - Single source of truth for all positioning
        this.layout = {
            // Base Y positions (relative to widget Y)
            controls: 10,
            deleteRow: 45,  // controls + buttonHeight + gap
            playback: 105,  // Calculated dynamically: deleteRow + deleteRowHeight + 35
            zoom: 170,      // playback + buttonHeight + 40 (increased gap)  
            scroll: 195,    // zoom + 20 (zoom height) + 5 (small gap)
            timeline: 215,  // scroll + 15 (scroll height) + 5 (small gap for visual separation)
            preview: 310,   // timeline + timelineHeight + 35 (space for hold indicators)
            info: 440,      // preview + previewHeight + 10
            
            // Heights
            buttonHeight: 25,
            deleteRowHeight: 25,
            timelineHeight: 60,
            previewHeight: 120,
            
            // Gaps and spacing
            standardGap: 10,
            sectionGap: 20
        };
        
        // Zoom and scroll
        this.zoomLevel = 1.0;  // 1.0 = fully zoomed out, 10.0 = max zoom in
        this.scrollOffset = 0;
        this.visibleFrames = 270;
        this.isDraggingZoom = false;
        this.isDraggingScroll = false;
        this.scrollDragStartX = 0;
        this.scrollDragStartOffset = 0; // Store initial scroll position when drag starts
        this.minZoom = 1.0;
        this.maxZoom = 10.0;
        
        // Batch boundaries
        this.batchSize = 81;
        
        // Preview cache
        this.previewCache = {};
        this.imageCache = {};  // Store loaded Image objects
        
        // Edge scrolling state
        this.edgeScrollDirection = 0; // -1 for left, 0 for none, 1 for right
        this.edgeScrollSpeed = 0;
        this.edgeScrollInterval = null;
        
        // Box selection state
        this.isBoxSelecting = false;
        this.boxSelectStart = {x: 0, y: 0};
        this.boxSelectEnd = {x: 0, y: 0};
        
        // Block selection state
        this.blockBounds = null; // Stores the visual block bounds for click detection
        this.isDraggingBlock = false;
        
        // Skewing/scaling state
        this.isDraggingLeftHandle = false;
        this.isDraggingRightHandle = false;
        this.skewStartX = 0;
        this.skewStartPositions = {}; // Store original positions for calculation
        this.skewStartMinFrame = 0; // Leftmost frame when scaling started
        this.skewStartMaxFrame = 0; // Rightmost frame when scaling started
        this.skewStartSpan = 0; // Original span when scaling started
        this.handleSize = 20; // Size of arrow handles
        this.leftHandleBounds = null;
        this.rightHandleBounds = null;
        this.scaleFactor = 1.0; // Current scale factor for display
        
        // Mask settings
        this.ignore_held_frames_mask = false; // Whether to ignore held frames in mask output
        
        // Folder memory for batch operations
        this.lastUsedFolder = null;
        
        // Reload prompt state
        this.showReloadPrompt = false;
        
        // Initialize
        if (!node.properties) {
            node.properties = {};
        }
        if (!node.properties.keyframeData) {
            node.properties.keyframeData = {
                frames: this.timeline_frames,
                keyframes: {}
            };
        }
    }
    
    draw(ctx, node, widget_width, y, widget_height) {
        // Store the y position for mouse event handling
        this.last_y = y;
        
        const width = widget_width - this.margin * 2;
        
        // Background
        ctx.fillStyle = "#1a1a1a";
        ctx.fillRect(this.margin, y, width, this.height);
        
        // Show reload prompt if we have keyframes but no images
        if (this.showReloadPrompt && Object.keys(this.keyframes).length > 0) {
            // Draw overlay
            ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
            ctx.fillRect(this.margin, y, width, this.height);
            
            // Draw message
            ctx.fillStyle = "#ffcc00";
            ctx.font = "16px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Keyframe images need to be reloaded", this.margin + width/2, y + this.height/2 - 20);
            
            // Draw reload button
            const reloadButtonWidth = 150;
            const reloadButtonHeight = 40;
            const reloadButtonX = this.margin + (width - reloadButtonWidth) / 2;
            const reloadButtonY = y + this.height/2;
            
            this.drawButton(ctx, reloadButtonX, reloadButtonY, reloadButtonWidth, reloadButtonHeight, "Select Image Folder", "#00cc00");
            
            // Draw "New Timeline" button below the reload button
            const newTimelineButtonY = reloadButtonY + reloadButtonHeight + 10;
            this.drawButton(ctx, reloadButtonX, newTimelineButtonY, reloadButtonWidth, reloadButtonHeight, "New Timeline", "#ff6600");
            
            // Store button bounds for click detection
            // Convert absolute Y to widget-relative Y by subtracting the widget's Y position
            this.reloadButtonBounds = {
                x: reloadButtonX,
                y: reloadButtonY - y,  // Convert to widget-relative coordinates
                width: reloadButtonWidth,
                height: reloadButtonHeight
            };
            
            this.newTimelineButtonBounds = {
                x: reloadButtonX,
                y: newTimelineButtonY - y,  // Convert to widget-relative coordinates
                width: reloadButtonWidth,
                height: reloadButtonHeight
            };
            
            return this.height;
        }
        
        // 1. CONTROL BUTTONS
        const controlsY = y + this.layout.controls;
        const buttonHeight = this.layout.buttonHeight;
        const buttonSpacing = 5;
        let buttonX = this.margin + 10;
        
        // Add Keyframes button
        this.drawButton(ctx, buttonX, controlsY, 100, buttonHeight, "Add Keyframes", "#0099ff");
        buttonX += 105;
        
        // Replace Selected button
        this.drawButton(ctx, buttonX, controlsY, 100, buttonHeight, "Replace Selected", "#0099ff", this.selectedKeyframe === null);
        buttonX += 105;
        
        // Batch Replace button
        this.drawButton(ctx, buttonX, controlsY, 110, buttonHeight, "Batch Replace", "#0099ff");
        buttonX += 115;

        // Clear All button (adjacent)
        this.drawButton(ctx, buttonX, controlsY, 90, buttonHeight, "Clear All", "#cc0000");
        buttonX += 95;
        
        // (No Delete Selected button in this row)
        
        // Save Config button (adjacent to Clear All, no separator)
        this.drawButton(ctx, buttonX, controlsY, 90, buttonHeight, "Save Config", "#666");
        buttonX += 95;
        
        // Load Config button (keep within node width)
        this.drawButton(ctx, buttonX, controlsY, 90, buttonHeight, "Load Config", "#666");
        
        // --- NEW ROW: Delete Selected Keyframes ---
        const deleteRowY = y + this.layout.deleteRow;
        const deleteRowHeight = this.layout.deleteRowHeight;
        const deleteRowButtonX = this.margin + 10;
        const deleteRowButtonColor = "#cc2222";
        const deleteRowButtonDisabled = this.selectedKeyframes.size === 0;
        this.drawButton(
            ctx,
            deleteRowButtonX,
            deleteRowY,
            110,
            deleteRowHeight,
            "Delete Selected",
            deleteRowButtonColor,
            deleteRowButtonDisabled
        );
        
        // Key Hold Duration control - works on selected keyframes
        const holdControlX = deleteRowButtonX + 120;
        const holdControlWidth = 150;
        const hasSelectedKeyframes = this.selectedKeyframes.size > 0 || this.selectedKeyframe !== null;
        
        // Get hold duration from selected keyframe(s) or default
        let displayHoldDuration = 1;
        if (this.selectedKeyframe !== null && this.keyframes[this.selectedKeyframe]) {
            displayHoldDuration = this.keyframes[this.selectedKeyframe].hold || 1;
        } else if (this.selectedKeyframes.size > 0) {
            // Show the hold duration of the first selected keyframe
            const firstSelected = Array.from(this.selectedKeyframes)[0];
            displayHoldDuration = this.keyframes[firstSelected]?.hold || 1;
        }
        
        // Draw label with selection context
        ctx.fillStyle = hasSelectedKeyframes ? "#ffffff" : "#888";
        ctx.font = "12px Arial";
        const labelText = hasSelectedKeyframes ? "Hold Duration:" : "Hold (select key):";
        ctx.fillText(labelText, holdControlX, deleteRowY + 15);
        
        // Draw value box
        const valueBoxX = holdControlX + 100;
        const valueBoxWidth = 40;
        ctx.fillStyle = hasSelectedKeyframes ? "#333" : "#222";
        ctx.fillRect(valueBoxX, deleteRowY, valueBoxWidth, deleteRowHeight);
        ctx.strokeStyle = hasSelectedKeyframes ? "#666" : "#444";
        ctx.strokeRect(valueBoxX, deleteRowY, valueBoxWidth, deleteRowHeight);
        
        // Draw value text
        ctx.fillStyle = hasSelectedKeyframes ? "#ffffff" : "#666";
        ctx.textAlign = "center";
        ctx.fillText(displayHoldDuration.toString(), valueBoxX + valueBoxWidth/2, deleteRowY + 16);
        ctx.textAlign = "left";
        
        // Draw +/- buttons
        const buttonSize = deleteRowHeight;
        const minusButtonX = valueBoxX + valueBoxWidth + 5;
        const plusButtonX = minusButtonX + buttonSize + 2;
        
        // Buttons disabled if no selection or at limits
        const minusDisabled = !hasSelectedKeyframes || displayHoldDuration <= 1;
        const plusDisabled = !hasSelectedKeyframes || displayHoldDuration >= 100;
        
        // Minus button
        this.drawButton(ctx, minusButtonX, deleteRowY, buttonSize, buttonSize, "-", "#666", minusDisabled);
        
        // Plus button  
        this.drawButton(ctx, plusButtonX, deleteRowY, buttonSize, buttonSize, "+", "#666", plusDisabled);
        
        // Checkbox for ignoring held frames in mask
        const checkboxX = plusButtonX + buttonSize + 15;
        const checkboxSize = 15;
        const checkboxY = deleteRowY + (deleteRowHeight - checkboxSize) / 2;
        
        // Draw checkbox
        ctx.fillStyle = "#333";
        ctx.fillRect(checkboxX, checkboxY, checkboxSize, checkboxSize);
        ctx.strokeStyle = "#666";
        ctx.lineWidth = 1;
        ctx.strokeRect(checkboxX, checkboxY, checkboxSize, checkboxSize);
        
        // Draw check mark if enabled
        if (this.ignore_held_frames_mask) {
            ctx.strokeStyle = "#0099ff";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(checkboxX + 3, checkboxY + 7);
            ctx.lineTo(checkboxX + 6, checkboxY + 10);
            ctx.lineTo(checkboxX + 12, checkboxY + 4);
            ctx.stroke();
        }
        
        // Label
        ctx.fillStyle = "#ccc";
        ctx.font = "11px Arial";
        ctx.textAlign = "left";
        ctx.fillText("Ignore held frames for mask", checkboxX + checkboxSize + 5, deleteRowY + 16);
        
        // --- END NEW ROW ---
        
        // 2. PLAYBACK CONTROLS
        const playbackY = y + this.layout.playback;
        buttonX = this.margin + 10;
        
        // Play/Pause button with modern design
        const playPauseColor = this.isPlaying ? "#ff6600" : "#00cc00";
        this.drawPlaybackButton(ctx, buttonX, playbackY, buttonHeight, buttonHeight, 
                               this.isPlaying ? "pause" : "play", playPauseColor);
        buttonX += buttonHeight + 5;
        
        // Stop button with modern design
        this.drawPlaybackButton(ctx, buttonX, playbackY, buttonHeight, buttonHeight, 
                               "stop", "#666666");
        buttonX += buttonHeight + 10;
        
        // Frame counter
        ctx.fillStyle = "#fff";
        ctx.font = "14px Arial";
        ctx.textAlign = "left";
        const frameText = `Frame: ${Math.floor(this.currentFrame)} / ${this.timeline_frames}`;
        ctx.fillText(frameText, buttonX, playbackY + 17);
        
        // Time display
        const currentTime = (this.currentFrame / this.fps).toFixed(2);
        const totalTime = (this.timeline_frames / this.fps).toFixed(2);
        ctx.fillText(`${currentTime}s / ${totalTime}s`, buttonX + 150, playbackY + 17);
        
        // 3. TIMELINE DURATION CONTROLS
        const durationY = y + 80;
        ctx.fillStyle = "#888";
        ctx.font = "12px Arial";
        ctx.fillText("Timeline Duration:", this.margin + 10, durationY + 15);
        ctx.fillStyle = "#fff";
        ctx.fillText(`${(this.timeline_frames / this.fps).toFixed(1)}s (${this.timeline_frames} frames)`, this.margin + 120, durationY + 15);
        
        // --- Shift everything below this line down by 35px ---
        const shiftY = 25 + 10; // deleteRowHeight + spacing
        
        // 4. ZOOM SLIDER
        const zoomY = y + this.layout.zoom;
        const zoomWidth = width - 20;
        
        // Zoom track background
        ctx.fillStyle = "#222";
        ctx.fillRect(this.margin + 10, zoomY, zoomWidth, 20);
        
        // Zoom track border
        ctx.strokeStyle = "#444";
        ctx.lineWidth = 1;
        ctx.strokeRect(this.margin + 10, zoomY, zoomWidth, 20);
        
        // Zoom slider position (1.0-10.0 mapped to 0-1)
        const zoomNormalized = (this.zoomLevel - this.minZoom) / (this.maxZoom - this.minZoom);
        const zoomX = this.margin + 10 + zoomNormalized * (zoomWidth - 20);
        
        // Zoom slider handle
        ctx.save();
        // Handle shadow
        ctx.shadowBlur = 4;
        ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
        ctx.shadowOffsetY = 2;
        
        // Handle gradient
        const handleGradient = ctx.createLinearGradient(zoomX - 10, zoomY - 2, zoomX - 10, zoomY + 22);
        handleGradient.addColorStop(0, "#0099ff");
        handleGradient.addColorStop(0.5, "#0077dd");
        handleGradient.addColorStop(1, "#0055bb");
        ctx.fillStyle = handleGradient;
        ctx.fillRect(zoomX - 10, zoomY - 2, 20, 24);
        
        // Handle highlight
        ctx.strokeStyle = "#00aaff";
        ctx.lineWidth = 1;
        ctx.strokeRect(zoomX - 10, zoomY - 2, 20, 24);
        ctx.restore();
        
        // Zoom percentage
        ctx.fillStyle = "#fff";
        ctx.font = "11px Arial";
        ctx.textAlign = "right";
        const zoomPercent = Math.round((this.zoomLevel / this.maxZoom) * 1000);
        ctx.fillText(`${zoomPercent}%`, this.margin + width - 10, zoomY - 5);
        
        // 4b. HORIZONTAL SCROLLBAR (always present but subtle when not needed)
        const scrollY = y + this.layout.scroll;
        const scrollHeight = 15;
        
        // Always draw scrollbar track
        ctx.fillStyle = this.zoomLevel > 1.0 ? "#222" : "#1a1a1a";
        ctx.fillRect(this.margin + 10, scrollY, zoomWidth, scrollHeight);
        ctx.strokeStyle = this.zoomLevel > 1.0 ? "#444" : "#333";
        ctx.lineWidth = 1;
        ctx.strokeRect(this.margin + 10, scrollY, zoomWidth, scrollHeight);
        
        if (this.zoomLevel > 1.0) {
            // Calculate visible portion of timeline
            const visibleFrames = this.timeline_frames / this.zoomLevel;
            const maxScroll = this.timeline_frames - visibleFrames;
            
            // Scrollbar thumb
            const thumbWidth = Math.max(30, (visibleFrames / this.timeline_frames) * zoomWidth);
            const thumbX = this.margin + 10 + (this.scrollOffset / maxScroll) * (zoomWidth - thumbWidth);
            
            // Thumb gradient
            const thumbGradient = ctx.createLinearGradient(0, scrollY, 0, scrollY + scrollHeight);
            thumbGradient.addColorStop(0, "#666");
            thumbGradient.addColorStop(0.5, "#555");
            thumbGradient.addColorStop(1, "#444");
            ctx.fillStyle = thumbGradient;
            ctx.fillRect(thumbX, scrollY, thumbWidth, scrollHeight);
            
            // Thumb border
            ctx.strokeStyle = "#777";
            ctx.lineWidth = 1;
            ctx.strokeRect(thumbX, scrollY, thumbWidth, scrollHeight);
        } else {
            // When fully zoomed out, show a subtle full-width thumb
            ctx.fillStyle = "#2a2a2a";
            ctx.fillRect(this.margin + 10, scrollY, zoomWidth, scrollHeight);
        }
        
        // 5. MAIN TIMELINE WITH SCRUBBER
        const timelineY = y + this.layout.timeline;
        const timelineHeight = this.layout.timelineHeight;
        const trackWidth = width - 20;
        
        // Premium timeline track with metallic finish
        ctx.save();
        
        // Outer bezel shadow
        ctx.shadowBlur = 8;
        ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
        ctx.shadowOffsetY = 2;
        ctx.fillStyle = "#000";
        ctx.fillRect(this.margin + 10 - 2, timelineY - 2, trackWidth + 4, timelineHeight + 4);
        ctx.shadowBlur = 0;
        
        // Timeline background with premium gradient
        const trackGradient = ctx.createLinearGradient(0, timelineY, 0, timelineY + timelineHeight);
        trackGradient.addColorStop(0, "#0a0a0a");
        trackGradient.addColorStop(0.1, "#1a1a1a");
        trackGradient.addColorStop(0.5, "#242424");
        trackGradient.addColorStop(0.9, "#1a1a1a");
        trackGradient.addColorStop(1, "#0a0a0a");
        ctx.fillStyle = trackGradient;
        ctx.fillRect(this.margin + 10, timelineY, trackWidth, timelineHeight);
        
        // Metallic sheen overlay
        const sheenGradient = ctx.createLinearGradient(this.margin + 10, timelineY, this.margin + 10 + trackWidth, timelineY);
        sheenGradient.addColorStop(0, "transparent");
        sheenGradient.addColorStop(0.3, "rgba(255, 255, 255, 0.02)");
        sheenGradient.addColorStop(0.5, "rgba(255, 255, 255, 0.04)");
        sheenGradient.addColorStop(0.7, "rgba(255, 255, 255, 0.02)");
        sheenGradient.addColorStop(1, "transparent");
        ctx.fillStyle = sheenGradient;
        ctx.fillRect(this.margin + 10, timelineY, trackWidth, timelineHeight);
        
        // Inner shadow for depth
        const innerShadow = ctx.createLinearGradient(0, timelineY, 0, timelineY + 8);
        innerShadow.addColorStop(0, "rgba(0, 0, 0, 0.6)");
        innerShadow.addColorStop(1, "transparent");
        ctx.fillStyle = innerShadow;
        ctx.fillRect(this.margin + 10, timelineY, trackWidth, 8);
        
        // Bottom highlight
        const bottomHighlight = ctx.createLinearGradient(0, timelineY + timelineHeight - 4, 0, timelineY + timelineHeight);
        bottomHighlight.addColorStop(0, "transparent");
        bottomHighlight.addColorStop(1, "rgba(255, 255, 255, 0.05)");
        ctx.fillStyle = bottomHighlight;
        ctx.fillRect(this.margin + 10, timelineY + timelineHeight - 4, trackWidth, 4);
        
        // Track border with metallic finish
        ctx.strokeStyle = "#555";
        ctx.lineWidth = 1;
        ctx.strokeRect(this.margin + 10, timelineY, trackWidth, timelineHeight);
        
        // Inner border highlight
        ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(this.margin + 10 + 1, timelineY + 1, trackWidth - 2, timelineHeight - 2);
        
        ctx.restore();
        
        // Set up clipping region for timeline content
        ctx.save();
        ctx.beginPath();
        ctx.rect(this.margin + 10, timelineY - 40, trackWidth, timelineHeight + 80);
        ctx.clip();
        
        // Calculate visible frame range when zoomed
        const timeline_frames_int = Math.round(this.timeline_frames); // Ensure integer
        const visibleFrames = timeline_frames_int / this.zoomLevel;
        const startFrame = Math.floor(this.scrollOffset);
        const endFrame = Math.ceil(this.scrollOffset + visibleFrames);
        
        // Helper function to convert frame to screen X position with zoom
        const frameToScreenX = (frame) => {
            const normalizedFrame = (frame - this.scrollOffset) / visibleFrames;
            return this.margin + 10 + normalizedFrame * trackWidth;
        };
        
        // Draw batch boundaries at the END of each batch
        if (this.batchSize > 0) {
            ctx.strokeStyle = "#ff880044";
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            
            for (let batch = 0; batch * this.batchSize < this.timeline_frames; batch++) {
                // Draw line at the END of each batch (frames 80, 161, 242, etc.)
                const frame = (batch + 1) * this.batchSize - 1;
                if (frame >= this.timeline_frames) break;
                
                // Only draw if visible
                if (frame >= startFrame - 10 && frame <= endFrame + 10) {
                    const x = frameToScreenX(frame);
                    
                    // Only draw if on screen
                    if (x >= this.margin && x <= this.margin + width) {
                        ctx.beginPath();
                        ctx.moveTo(x, timelineY);
                        ctx.lineTo(x, timelineY + timelineHeight);
                        ctx.stroke();
                        
                        // Batch label (B1|B2 showing the boundary between batches)
                        ctx.fillStyle = "#ff8800";
                        ctx.font = "10px Arial";
                        ctx.textAlign = "center";
                        ctx.fillText(`B${batch + 1}|B${batch + 2}`, x, timelineY - 5);
                    }
                }
            }
            ctx.setLineDash([]);
        }
        
        // Frame markers (adaptive 1–2–5 scale)
        ctx.strokeStyle = "#555";
        ctx.lineWidth = 1;
        ctx.font = "10px Arial";
        ctx.fillStyle = "#888";
        ctx.textAlign = "center";

        const trackPixelWidth = width - (this.margin + 10) * 2; // inner track width
        const visibleFramesForTicks = Math.max(1, endFrame - startFrame);
        const framesPerPixel = visibleFramesForTicks / Math.max(1, trackPixelWidth);

        // Aim for ~20 major ticks when fully zoomed out
        const targetMajorPx = Math.max(40, trackPixelWidth / 20);
        const targetFrames = targetMajorPx * framesPerPixel;

        const niceStep = (raw) => {
            const pow10 = Math.pow(10, Math.floor(Math.log10(raw)));
            const n = raw / pow10;
            let base;
            if (n <= 1) base = 1;
            else if (n <= 2) base = 2;
            else if (n <= 5) base = 5;
            else base = 10;
            return base * pow10;
        };

        const majorStep = Math.max(1, Math.round(niceStep(targetFrames)));
        const minorStep = Math.max(1, Math.floor(majorStep / 5));

        // Draw minor ticks
        const firstMinor = Math.floor(startFrame / minorStep) * minorStep;
        let minorCount = 0;
        for (let f = firstMinor; f <= endFrame && f <= this.timeline_frames; f += minorStep) {
            // Skip if this is also a major
            if (f % majorStep === 0) continue;
            const x = frameToScreenX(f);
            if (x < this.margin || x > this.margin + width) continue;
            ctx.beginPath();
            ctx.moveTo(x, timelineY + timelineHeight);
            ctx.lineTo(x, timelineY + timelineHeight + 6);
            ctx.stroke();
            if (++minorCount > 1000) break; // safety
        }

        // Draw major ticks with labels, avoid overcrowding
        const firstMajor = Math.floor(startFrame / majorStep) * majorStep;
        let lastLabelX = -Infinity;
        const minLabelPx = 40; // minimum spacing between labels
        let majorCount = 0;
        for (let f = firstMajor; f <= endFrame && f <= this.timeline_frames; f += majorStep) {
            const x = frameToScreenX(f);
            if (x < this.margin || x > this.margin + width) continue;
            // tick
            ctx.beginPath();
            ctx.moveTo(x, timelineY + timelineHeight);
            ctx.lineTo(x, timelineY + timelineHeight + 10);
            ctx.stroke();
            // label
            if (x - lastLabelX >= minLabelPx) {
                ctx.fillText(f.toString(), x, timelineY + timelineHeight + 20);
                lastLabelX = x;
            }
            if (++majorCount > 500) break; // safety
        }
        
        // Draw timeline range selection as solid 3D block BEFORE keyframes so they appear on top
        if (this.isBoxSelecting || (this.selectedKeyframes.size > 0 && !this.isDragging) || this.isDraggingBlock || this.isDraggingLeftHandle || this.isDraggingRightHandle) {
            let minX, maxX;
            
            if (this.isBoxSelecting) {
                // During selection
                minX = Math.min(this.boxSelectStart.x, this.boxSelectEnd.x);
                maxX = Math.max(this.boxSelectStart.x, this.boxSelectEnd.x);
            } else if ((this.isDraggingLeftHandle || this.isDraggingRightHandle)) {
                // During scaling - show block at scaled position (or original if not yet scaling)
                if (this.scaleFactor !== 1.0 && this.skewStartSpan > 0) {
                    let minFrame, maxFrame;
                    
                    if (this.isDraggingLeftHandle) {
                        // Scale from right pivot
                        const newSpan = this.skewStartSpan * this.scaleFactor;
                        minFrame = this.skewStartMaxFrame - newSpan;
                        maxFrame = this.skewStartMaxFrame;
                    } else {
                        // Scale from left pivot
                        const newSpan = this.skewStartSpan * this.scaleFactor;
                        minFrame = this.skewStartMinFrame;
                        maxFrame = this.skewStartMinFrame + newSpan;
                    }
                    
                    minX = frameToScreenX(minFrame) - 15;
                    maxX = frameToScreenX(maxFrame) + 15;
                } else {
                    // Not yet scaling, show at original position
                    const selectedFrames = Array.from(this.selectedKeyframes);
                    if (selectedFrames.length > 0) {
                        const minFrame = Math.min(...selectedFrames);
                        const maxFrame = Math.max(...selectedFrames);
                        minX = frameToScreenX(minFrame) - 15;
                        maxX = frameToScreenX(maxFrame) + 15;
                    }
                }
            } else if (this.isDraggingBlock && this.dragTargetFrame !== undefined && this.dragStartFrame !== undefined) {
                // During block dragging - show at dragged position
                const offset = this.dragTargetFrame - this.dragStartFrame;
                const selectedFrames = Array.from(this.selectedKeyframes);
                if (selectedFrames.length > 0) {
                    const minFrame = Math.min(...selectedFrames) + offset;
                    const maxFrame = Math.max(...selectedFrames) + offset;
                    minX = frameToScreenX(minFrame) - 15; // Add padding around keyframes
                    maxX = frameToScreenX(maxFrame) + 15;
                }
            } else {
                // Show block for selected keyframes at original position
                const selectedFrames = Array.from(this.selectedKeyframes);
                if (selectedFrames.length > 0) {
                    const minFrame = Math.min(...selectedFrames);
                    const maxFrame = Math.max(...selectedFrames);
                    minX = frameToScreenX(minFrame) - 15; // Add padding around keyframes
                    maxX = frameToScreenX(maxFrame) + 15;
                }
            }
            
            if (minX !== undefined && maxX !== undefined) {
                const timelineY = y + this.layout.timeline;  // Use centralized timeline position
                const timelineHeight = this.layout.timelineHeight;
                const blockTop = timelineY + 1; // Properly centered on timeline
                const blockHeight = timelineHeight - 2; // 2px shorter (1px from top, 1px from bottom)
                const blockWidth = maxX - minX;
                
                // For click detection, we need widget-relative coordinates
                const blockTopRelative = this.layout.timeline + 1; // Uses centralized position
                
                // Draw 3D wooden block with shading (more transparent so keyframes appear in front)
                ctx.save();
                ctx.globalAlpha = 0.4; // More transparent so keyframes appear in front of the block
                
                // Main block body - gradient for 3D shiny red wood effect
                const blockGradient = ctx.createLinearGradient(minX, blockTop, minX, blockTop + blockHeight);
                blockGradient.addColorStop(0, "#e74c3c"); // Bright red at top
                blockGradient.addColorStop(0.2, "#c0392b"); // Medium red
                blockGradient.addColorStop(0.8, "#a93226"); // Darker red
                blockGradient.addColorStop(1, "#922b21"); // Dark red wood at bottom
                
                ctx.fillStyle = blockGradient;
                ctx.fillRect(minX, blockTop, blockWidth, blockHeight);
                
                // Top highlight (simulating light hitting the top)
                const topGradient = ctx.createLinearGradient(minX, blockTop, minX, blockTop + 8);
                topGradient.addColorStop(0, "#ffffff44"); // Light highlight
                topGradient.addColorStop(1, "#ffffff00"); // Fade to transparent
                ctx.fillStyle = topGradient;
                ctx.fillRect(minX, blockTop, blockWidth, 8);
                
                // Left edge highlight
                const leftGradient = ctx.createLinearGradient(minX, blockTop, minX + 6, blockTop);
                leftGradient.addColorStop(0, "#ffffff33");
                leftGradient.addColorStop(1, "#ffffff00");
                ctx.fillStyle = leftGradient;
                ctx.fillRect(minX, blockTop, 6, blockHeight);
                
                // Right edge shadow
                const rightGradient = ctx.createLinearGradient(maxX - 6, blockTop, maxX, blockTop);
                rightGradient.addColorStop(0, "#00000000");
                rightGradient.addColorStop(1, "#00000033");
                ctx.fillStyle = rightGradient;
                ctx.fillRect(maxX - 6, blockTop, 6, blockHeight);
                
                // Bottom shadow
                const bottomGradient = ctx.createLinearGradient(minX, blockTop + blockHeight - 6, minX, blockTop + blockHeight);
                bottomGradient.addColorStop(0, "#00000000");
                bottomGradient.addColorStop(1, "#00000033");
                ctx.fillStyle = bottomGradient;
                ctx.fillRect(minX, blockTop + blockHeight - 6, blockWidth, 6);
                
                // Subtle wood grain texture lines
                ctx.strokeStyle = "#00000011";
                ctx.lineWidth = 1;
                for (let i = 1; i < 4; i++) {
                    const lineY = blockTop + (blockHeight / 4) * i;
                    ctx.beginPath();
                    ctx.moveTo(minX + 4, lineY);
                    ctx.lineTo(maxX - 4, lineY);
                    ctx.stroke();
                }
                
                // Border with darker red
                ctx.strokeStyle = "#922b21";
                ctx.lineWidth = 2;
                ctx.strokeRect(minX, blockTop, blockWidth, blockHeight);
                
                // Store block bounds for click detection (using widget-relative coordinates)
                this.blockBounds = {minX, maxX, blockTop: blockTopRelative, blockHeight};
                
                // Store handle bounds with extra hit area
                const hitAreaPadding = 6; // Extra pixels for easier clicking
                this.leftHandleBounds = {
                    x: minX - this.handleSize/2 - hitAreaPadding,
                    y: blockTopRelative - hitAreaPadding,
                    width: this.handleSize + hitAreaPadding * 2,
                    height: blockHeight + hitAreaPadding * 2
                };
                this.rightHandleBounds = {
                    x: maxX - this.handleSize/2 - hitAreaPadding,
                    y: blockTopRelative - hitAreaPadding,
                    width: this.handleSize + hitAreaPadding * 2,
                    height: blockHeight + hitAreaPadding * 2
                };
                
                ctx.restore();
                
                // Block label removed for cleaner appearance
                
                // Store handle drawing info for later (after keyframes are drawn)
                if (!this.isBoxSelecting && blockWidth > 40 && this.selectedKeyframes.size > 1) {
                    this.handleDrawInfo = {
                        minX,
                        maxX,
                        blockTop,
                        blockHeight,
                        handleCenterY: blockTop + blockHeight/2,
                        showHandles: true
                    };
                } else {
                    this.handleDrawInfo = null;
                }
            }
        } else {
            // Clear handle bounds when we don't have multiple selections
            this.leftHandleBounds = null;
            this.rightHandleBounds = null;
            this.blockBounds = null;
        }
        
        // Draw keyframes
        const sortedFrames = Object.keys(this.keyframes).map(Number).sort((a, b) => a - b);
        sortedFrames.forEach((frame, index) => {
            // Skip drawing any selected keyframes at their original positions when dragging or scaling
            if ((this.isDragging || this.isDraggingBlock || 
                ((this.isDraggingLeftHandle || this.isDraggingRightHandle) && this.scaleFactor !== 1.0)) && 
                this.selectedKeyframes.has(frame)) {
                return;
            }
            
            // Only draw keyframes that are visible
            if (frame < startFrame - 5 || frame > endFrame + 5) {
                return;
            }
            
            const kf = this.keyframes[frame];
            if (!kf) return; // Skip if keyframe data is missing
            const x = frameToScreenX(frame);
            
            // Skip if off screen
            if (x < this.margin - 20 || x > this.margin + width + 20) {
                return;
            }
            // Check if this is the keyframe being dragged
            const isBeingDragged = this.isDragging && frame === this.dragStartFrame;
            const isSelected = (!this.isDragging && this.selectedKeyframe === frame) || 
                              (isBeingDragged) || 
                              this.selectedKeyframes.has(frame);
            const isHovered = this.hoverKeyframe === frame;
            const isEnabled = kf.enabled !== false;
            const isOnBoundary = this.isExactlyOnBoundary(frame);
            
            // Premium glass-like keyframe diamond design - taller and skinnier
            const baseSize = isSelected ? 24 : (isHovered ? 20 : 17);
            const centerY = timelineY + timelineHeight/2;
            
            // Save context for shadow effects
            ctx.save();
            
            // Enhanced shadow for depth when appearing in front of block
            ctx.shadowBlur = isSelected ? 20 : 15; // Stronger shadow for selected keyframes
            // Green shadow if on boundary, otherwise normal colors  
            if (isOnBoundary && isEnabled) {
                ctx.shadowColor = isSelected ? "#00ff00" : "#00cc00";
            } else {
                ctx.shadowColor = isSelected ? "#ffcc00" : (isEnabled ? "#0099ff" : "#666");
            }
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 0; // Remove vertical offset to prevent visual shift
            
            // Function to draw tall skinny diamond shape
            // Use fixed horizontal width to prevent shifting when scaling
            const fixedWidth = 4; // Fixed width in pixels
            const drawDiamond = (centerX, centerY, size) => {
                ctx.beginPath();
                ctx.moveTo(centerX, centerY - size); // Top
                ctx.lineTo(centerX + fixedWidth, centerY); // Right (fixed width)
                ctx.lineTo(centerX, centerY + size); // Bottom
                ctx.lineTo(centerX - fixedWidth, centerY); // Left (fixed width)
                ctx.closePath();
            };
            
            // Base layer - metallic diamond with gradient
            // Create gradient centered on the diamond for proper alignment
            const gradientSize = baseSize * 0.7; // Smaller gradient size for better centering
            const baseGradient = ctx.createLinearGradient(x - gradientSize, centerY - gradientSize, x + gradientSize, centerY + gradientSize);
            if (isOnBoundary && isEnabled) {
                // Green gradient for keyframes on boundaries
                if (isSelected) {
                    baseGradient.addColorStop(0, "#88ff88");
                    baseGradient.addColorStop(0.3, "#00ff00");
                    baseGradient.addColorStop(0.7, "#00cc00");
                    baseGradient.addColorStop(1, "#008800");
                } else {
                    baseGradient.addColorStop(0, "#66ff66");
                    baseGradient.addColorStop(0.3, "#00dd00");
                    baseGradient.addColorStop(0.7, "#00aa00");
                    baseGradient.addColorStop(1, "#007700");
                }
            } else if (isSelected) {
                baseGradient.addColorStop(0, "#ffee44");
                baseGradient.addColorStop(0.3, "#ffcc00");
                baseGradient.addColorStop(0.7, "#ff9900");
                baseGradient.addColorStop(1, "#cc6600");
            } else if (isEnabled) {
                baseGradient.addColorStop(0, "#66ddff");
                baseGradient.addColorStop(0.3, "#0099ff");
                baseGradient.addColorStop(0.7, "#0066cc");
                baseGradient.addColorStop(1, "#003388");
            } else {
                baseGradient.addColorStop(0, "#999999");
                baseGradient.addColorStop(0.3, "#666666");
                baseGradient.addColorStop(0.7, "#444444");
                baseGradient.addColorStop(1, "#222222");
            }
            
            ctx.fillStyle = baseGradient;
            drawDiamond(x, centerY, baseSize);
            ctx.fill();
            
            // Reset shadow for inner layers
            ctx.shadowBlur = 0;
            
            // Glass effect - top highlight on diamond
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            const glassGradient = ctx.createLinearGradient(x, centerY - baseSize, x, centerY);
            glassGradient.addColorStop(0, "rgba(255, 255, 255, 0.3)");
            glassGradient.addColorStop(0.5, "rgba(255, 255, 255, 0.1)");
            glassGradient.addColorStop(1, "transparent");
            
            ctx.fillStyle = glassGradient;
            drawDiamond(x, centerY, baseSize * 0.9);
            ctx.fill();
            ctx.restore();
            
            // Inner border for depth
            let borderColor, accentColor;
            if (isOnBoundary && isEnabled) {
                borderColor = isSelected ? "#00ff0066" : "#00cc0044";
                accentColor = isSelected ? "#00ff00" : "#00dd00";
            } else {
                borderColor = isSelected ? "#ffcc0066" : (isEnabled ? "#0099ff44" : "#44444466");
                accentColor = isSelected ? "#ffcc00" : (isEnabled ? "#00ccff" : "#666");
            }
            
            ctx.strokeStyle = borderColor;
            ctx.lineWidth = 1.5;
            drawDiamond(x, centerY, baseSize * 0.85);
            ctx.stroke();
            
            // Small center accent line (vertical)
            ctx.shadowBlur = isSelected ? 8 : 4;
            ctx.shadowColor = accentColor;
            
            ctx.strokeStyle = accentColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x, centerY - baseSize * 0.3);
            ctx.lineTo(x, centerY + baseSize * 0.3);
            ctx.stroke();
            
            // Restore context
            ctx.restore();
            
            // Keyframe number - positioned below (use fixed offset, not baseSize)
            ctx.fillStyle = accentColor;
            ctx.font = "bold 10px Arial";
            ctx.textAlign = "center";
            const fixedTextOffset = 29; // Fixed offset for text
            ctx.fillText((index + 1).toString(), x, timelineY + timelineHeight + fixedTextOffset);
            
            // Hold duration indicator - using per-keyframe hold duration
            const keyHoldDuration = kf.hold || 1;
            if (keyHoldDuration > 1) {
                const holdEndX = frameToScreenX(Math.min(frame + keyHoldDuration - 1, this.timeline_frames - 1));
                ctx.strokeStyle = "#00cc6666"; // Semi-transparent green
                ctx.lineWidth = 4;
                ctx.beginPath();
                // Position right at the timeline bottom edge
                const holdY = timelineY + timelineHeight;
                ctx.moveTo(x, holdY);
                ctx.lineTo(holdEndX, holdY);
                ctx.stroke();
                
                // Hold duration text
                ctx.fillStyle = "#00cc66";
                ctx.font = "bold 10px Arial";
                ctx.textAlign = "center";
                ctx.fillText(`${keyHoldDuration}f`, (x + holdEndX) / 2, holdY + 12);
            }
            
        });
        
        // Draw scaled keyframes at their new positions during scaling
        if ((this.isDraggingLeftHandle || this.isDraggingRightHandle) && this.scaleFactor !== 1.0) {
            if (this.skewStartSpan > 0) {
                // Calculate new positions for all keyframes
                for (const [frameStr, originalPos] of Object.entries(this.skewStartPositions)) {
                    const frame = parseInt(frameStr);
                    const kf = this.keyframes[frame];
                    if (!kf) continue;
                    
                    let newFrame;
                    const relativePos = (originalPos - this.skewStartMinFrame) / this.skewStartSpan;
                    
                    if (this.isDraggingLeftHandle) {
                        // Scale from right pivot
                        const newSpan = this.skewStartSpan * this.scaleFactor;
                        const newMin = this.skewStartMaxFrame - newSpan;
                        newFrame = Math.round(newMin + relativePos * newSpan);
                    } else {
                        // Scale from left pivot
                        const newSpan = this.skewStartSpan * this.scaleFactor;
                        newFrame = Math.round(this.skewStartMinFrame + relativePos * newSpan);
                    }
                    
                    // Clamp to timeline bounds
                    newFrame = Math.max(0, Math.min(this.timeline_frames - 1, newFrame));
                    
                    // Draw ghost keyframe at new position
                    const x = frameToScreenX(newFrame);
                    if (x >= this.margin - 20 && x <= this.margin + width + 20) {
                        const isEnabled = kf.enabled !== false;
                        const centerY = timelineY + timelineHeight/2;
                        const baseSize = 24; // Always show as selected size
                        
                        ctx.save();
                        ctx.globalAlpha = 0.6; // Semi-transparent for ghost effect
                        
                        // Enhanced shadow for ghost keyframes
                        ctx.shadowBlur = 20;
                        ctx.shadowColor = "#ffcc00"; // Gold shadow for scaling
                        ctx.shadowOffsetX = 0;
                        ctx.shadowOffsetY = 0;
                        
                        // Draw diamond shape
                        const fixedWidth = 4;
                        ctx.beginPath();
                        ctx.moveTo(x, centerY - baseSize); // Top
                        ctx.lineTo(x + fixedWidth, centerY); // Right
                        ctx.lineTo(x, centerY + baseSize); // Bottom
                        ctx.lineTo(x - fixedWidth, centerY); // Left
                        ctx.closePath();
                        
                        // Gradient fill
                        const gradientSize = baseSize * 0.7;
                        const gradient = ctx.createLinearGradient(x - gradientSize, centerY - gradientSize, x + gradientSize, centerY + gradientSize);
                        gradient.addColorStop(0, "#ffee88");
                        gradient.addColorStop(0.3, "#ffcc00");
                        gradient.addColorStop(0.7, "#ff9900");
                        gradient.addColorStop(1, "#cc6600");
                        
                        ctx.fillStyle = gradient;
                        ctx.fill();
                        
                        // Border
                        ctx.strokeStyle = "#ffffff";
                        ctx.lineWidth = 2;
                        ctx.stroke();
                        
                        ctx.restore();
                    }
                }
            }
        }
        
        // Draw all dragged keyframes at their new positions
        if ((this.isDragging || this.isDraggingBlock) && this.dragStartFrame !== undefined && this.dragTargetFrame !== undefined) {
            const offset = this.dragTargetFrame - this.dragStartFrame;
            const selectedFrames = Array.from(this.selectedKeyframes);
            
            // Check for collisions first
            let hasCollision = false;
            const targetPositions = new Set();
            
            for (const frame of selectedFrames) {
                const newFrame = frame + offset;
                
                // Check bounds
                if (newFrame < 0 || newFrame >= this.timeline_frames) {
                    hasCollision = true;
                    break;
                }
                
                // Check for collision with non-selected keyframes
                if (this.keyframes[newFrame] && !this.selectedKeyframes.has(newFrame)) {
                    hasCollision = true;
                    break;
                }
                
                // Check for internal collisions
                if (targetPositions.has(newFrame)) {
                    hasCollision = true;
                    break;
                }
                
                targetPositions.add(newFrame);
            }
            
            // Draw each selected keyframe at its new position
            selectedFrames.forEach(frame => {
                const kf = this.keyframes[frame];
                if (kf) {
                    const newFrame = frame + offset;
                    const x = frameToScreenX(newFrame);
                    const isEnabled = kf.enabled !== false;
                
                // Premium diamond keyframe design for dragging - even taller
                const baseSize = 28; // Larger when dragging
                const centerY = timelineY + timelineHeight/2;
                
                ctx.save();
                
                // Function to draw tall skinny diamond shape
                // Use fixed horizontal width to prevent shifting when scaling
                const fixedWidth = 4; // Fixed width in pixels
                const drawDiamond = (centerX, centerY, size) => {
                    ctx.beginPath();
                    ctx.moveTo(centerX, centerY - size); // Top
                    ctx.lineTo(centerX + fixedWidth, centerY); // Right (fixed width)
                    ctx.lineTo(centerX, centerY + size); // Bottom
                    ctx.lineTo(centerX - fixedWidth, centerY); // Left (fixed width)
                    ctx.closePath();
                };
                
                // Check if the current drag position is on a boundary
                const isExactlyOnBoundary = this.isExactlyOnBoundary(this.dragTargetFrame);
                
                // Enhanced shadow for dragged keyframe - red if collision, green if boundary, yellow normal
                ctx.shadowBlur = 30;
                ctx.shadowColor = hasCollision ? "#ff0000" : (isExactlyOnBoundary ? "#00ff00" : "#ffcc00");
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 5;
                
                // Base layer - glowing metallic diamond (green tint if exactly on boundary)
                // Create gradient centered on the diamond for proper alignment
                const gradientSize = baseSize * 0.7; // Smaller gradient size for better centering
                const baseGradient = ctx.createLinearGradient(x - gradientSize, centerY - gradientSize, x + gradientSize, centerY + gradientSize);
                if (isExactlyOnBoundary) {
                    // Green gradient when near boundary
                    baseGradient.addColorStop(0, "#66ff66");
                    baseGradient.addColorStop(0.3, "#00ff00");
                    baseGradient.addColorStop(0.7, "#00cc00");
                    baseGradient.addColorStop(1, "#008800");
                } else {
                    // Normal yellow gradient
                    baseGradient.addColorStop(0, "#ffff66");
                    baseGradient.addColorStop(0.3, "#ffcc00");
                    baseGradient.addColorStop(0.7, "#ff9900");
                    baseGradient.addColorStop(1, "#ff6600");
                }
                
                ctx.fillStyle = baseGradient;
                drawDiamond(x, centerY, baseSize);
                ctx.fill();
                
                // Reset shadow for inner layers
                ctx.shadowBlur = 0;
                
                // Glass effect overlay
                ctx.save();
                ctx.globalCompositeOperation = "screen";
                const glassGradient = ctx.createLinearGradient(x, centerY - baseSize, x, centerY);
                glassGradient.addColorStop(0, "rgba(255, 255, 255, 0.5)");
                glassGradient.addColorStop(0.5, "rgba(255, 255, 255, 0.2)");
                glassGradient.addColorStop(1, "transparent");
                
                ctx.fillStyle = glassGradient;
                drawDiamond(x, centerY, baseSize * 0.9);
                ctx.fill();
                ctx.restore();
                
                // Pulsing glow border for dragged state (green if exactly on boundary)
                const borderColor = isExactlyOnBoundary ? "#00ff00" : "#ffcc00";
                ctx.strokeStyle = borderColor + "aa";
                ctx.lineWidth = 3;
                ctx.shadowBlur = 15;
                ctx.shadowColor = borderColor;
                drawDiamond(x, centerY, baseSize * 0.9);
                ctx.stroke();
                
                // Center accent line - extra bright when dragging (green if near boundary)
                ctx.shadowBlur = 15;
                ctx.shadowColor = borderColor;
                
                ctx.strokeStyle = borderColor;
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(x, centerY - baseSize * 0.4);
                ctx.lineTo(x, centerY + baseSize * 0.4);
                ctx.stroke();
                
                ctx.restore();
                
                // Keyframe number (green if near boundary)
                const originalIndex = sortedFrames.indexOf(frame);
                ctx.fillStyle = borderColor;
                ctx.font = "bold 10px Arial";
                ctx.textAlign = "center";
                ctx.fillText((originalIndex + 1).toString(), x, timelineY + timelineHeight + baseSize + 12);
                }
            });
        }
        
        // Draw batch boundary indicators AFTER keyframes to ensure visibility
        if (this.batchSize > 0) {
            ctx.save();
            
            // Calculate boundary indicators for each batch boundary
            const boundaryIndicators = [];
            
            // Add frame 0 as the first boundary (start of first batch)
            const frame0x = frameToScreenX(0);
            const hasKeyframeAt0 = this.hasKeyframeNearBoundary(0);
            // Only add if visible
            if (0 >= startFrame - 5 && 0 <= endFrame + 5) {
                boundaryIndicators.push({
                    frame: 0,
                    x: frame0x,
                    hasKeyframe: hasKeyframeAt0,
                    batchIndex: 0,
                    isStart: true
                });
            }
            
            // For each batch end boundary
            for (let batch = 0; batch * this.batchSize < this.timeline_frames; batch++) {
                // Calculate the last frame of this batch (0-indexed)
                // Batch 0: frames 0-80 (last frame is 80), Batch 1: frames 81-161 (last frame is 161)
                const boundaryFrame = Math.min((batch + 1) * this.batchSize - 1, this.timeline_frames - 1);
                
                // Only process if boundary is visible
                if (boundaryFrame < startFrame - 5 || boundaryFrame > endFrame + 5) {
                    continue;
                }
                
                // Position the indicator at the actual boundary frame position
                const x = frameToScreenX(boundaryFrame);
                
                // Additional bounds check - ensure indicator stays within widget bounds
                if (x < this.margin || x > this.margin + width) {
                    continue;
                }
                
                // Check if there's a keyframe near this boundary position (more forgiving)
                const hasKeyframe = this.hasKeyframeNearBoundary(boundaryFrame);
                
                boundaryIndicators.push({
                    frame: boundaryFrame,
                    x: x,
                    hasKeyframe: hasKeyframe,
                    batchIndex: batch,
                    isStart: false
                });
            }
            
            // Draw boundary indicators ONLY for boundaries that have keyframes
            boundaryIndicators.forEach(indicator => {
                // Check if we're currently dragging a keyframe
                const isDraggingToThisBoundary = this.isDragging && 
                    this.dragTargetFrame === indicator.frame;
                
                // Only draw indicators for boundaries that actually have keyframes
                // BUT NOT if we're currently dragging a keyframe to this position
                if (indicator.hasKeyframe && !isDraggingToThisBoundary) {
                    // Green glow effect around the boundary position
                    ctx.save();
                    ctx.globalCompositeOperation = "screen";
                    
                    // Multi-layer green glow
                    const glowColor = "rgba(0, 255, 0, 0.4)";
                    for (let radius = 25; radius >= 10; radius -= 5) {
                        ctx.beginPath();
                        ctx.arc(indicator.x, timelineY + timelineHeight / 2, radius, 0, 2 * Math.PI);
                        ctx.fillStyle = `rgba(0, 255, 0, ${0.1 + (25 - radius) * 0.02})`;
                        ctx.fill();
                    }
                    ctx.restore();
                    
                    // Draw green outline box CENTERED on the boundary line
                    ctx.strokeStyle = "rgba(0, 200, 0, 0.8)";
                    ctx.lineWidth = 3;
                    ctx.setLineDash([]);
                    
                    // Center the box on the boundary line
                    const boxWidth = 24;
                    const boxHeight = timelineHeight + 10;
                    const boxX = indicator.x - boxWidth / 2;
                    const boxY = timelineY - 5;
                    
                    ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
                    
                    // Draw "OK" checkmark above, centered on boundary
                    ctx.strokeStyle = "rgba(0, 200, 0, 1)";
                    ctx.lineWidth = 2;
                    ctx.lineCap = "round";
                    ctx.lineJoin = "round";
                    
                    // Checkmark path centered on indicator.x
                    ctx.beginPath();
                    ctx.moveTo(indicator.x - 6, timelineY - 18);
                    ctx.lineTo(indicator.x - 2, timelineY - 14);
                    ctx.lineTo(indicator.x + 6, timelineY - 22);
                    ctx.stroke();
                } else if (!indicator.hasKeyframe && this.batchSize > 0) {
                    // Red indicators for missing boundary keyframes
                    ctx.save();
                    
                    // Pulsing red circle effect using time-based animation
                    const time = Date.now() * 0.003;
                    const pulseIntensity = 0.3 + 0.2 * Math.sin(time);
                    
                    ctx.globalCompositeOperation = "screen";
                    ctx.fillStyle = `rgba(255, 0, 0, ${pulseIntensity})`;
                    ctx.beginPath();
                    ctx.arc(indicator.x, timelineY + timelineHeight / 2, 15, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.restore();
                    
                    // Draw warning triangle
                    ctx.fillStyle = "rgba(255, 0, 0, 0.9)";
                    ctx.beginPath();
                    ctx.moveTo(indicator.x, timelineY - 25);
                    ctx.lineTo(indicator.x - 8, timelineY - 10);
                    ctx.lineTo(indicator.x + 8, timelineY - 10);
                    ctx.closePath();
                    ctx.fill();
                    
                    // Warning exclamation mark
                    ctx.fillStyle = "white";
                    ctx.font = "bold 12px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("!", indicator.x, timelineY - 13);
                    
                    // "MISSING KF" text below
                    ctx.fillStyle = "rgba(255, 0, 0, 1)";
                    ctx.font = "bold 8px Arial";
                    ctx.fillText("MISSING KF", indicator.x, timelineY - 30);
                }
            });
            
            ctx.restore();
        }
        
        // Restore context to remove clipping for scrubber and other UI elements
        ctx.restore();
        
        // Current frame indicator (scrubber) - always draw but clamp to visible bounds
        const scrubberX = frameToScreenX(this.currentFrame);
        
        // Determine if scrubber should be clamped to edges
        let displayX = scrubberX;
        let isOffscreen = false;
        
        if (this.currentFrame < startFrame) {
            // Scrubber is off the left edge
            displayX = this.margin + 10;
            isOffscreen = true;
        } else if (this.currentFrame > endFrame) {
            // Scrubber is off the right edge
            displayX = this.margin + 10 + trackWidth;
            isOffscreen = true;
        }
        
        // Always draw the scrubber
        // Scrubber line
        ctx.strokeStyle = isOffscreen ? "#ff000080" : "#ff0000"; // Semi-transparent when offscreen
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(displayX, timelineY - 20);
        ctx.lineTo(displayX, timelineY + timelineHeight + 30);
        ctx.stroke();
        
        // Scrubber handle
        ctx.fillStyle = isOffscreen ? "#ff000080" : "#ff0000";
        ctx.beginPath();
        ctx.moveTo(displayX - 8, timelineY - 20);
        ctx.lineTo(displayX + 8, timelineY - 20);
        ctx.lineTo(displayX + 8, timelineY - 10);
        ctx.lineTo(displayX, timelineY);
        ctx.lineTo(displayX - 8, timelineY - 10);
        ctx.closePath();
        ctx.fill();
        
        // Diamond on timeline
        const diamondSize = 8;
        ctx.save();
        ctx.translate(displayX, timelineY + timelineHeight/2);
        ctx.rotate(Math.PI / 4);
        ctx.fillStyle = isOffscreen ? "#ff000080" : "#ff0000";
        ctx.fillRect(-diamondSize/2, -diamondSize/2, diamondSize, diamondSize);
        ctx.restore();
        
        // Current frame number
        ctx.fillStyle = isOffscreen ? "#ffffff80" : "#ffffff";
        ctx.font = "bold 11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`F${Math.floor(this.currentFrame)}`, displayX, timelineY - 25);
        
        // Add directional arrow if offscreen
        if (isOffscreen) {
            ctx.fillStyle = "#ff0000";
            ctx.beginPath();
            if (this.currentFrame < startFrame) {
                // Left arrow
                ctx.moveTo(displayX + 15, timelineY + timelineHeight/2);
                ctx.lineTo(displayX + 25, timelineY + timelineHeight/2 - 5);
                ctx.lineTo(displayX + 25, timelineY + timelineHeight/2 + 5);
            } else {
                // Right arrow
                ctx.moveTo(displayX - 15, timelineY + timelineHeight/2);
                ctx.lineTo(displayX - 25, timelineY + timelineHeight/2 - 5);
                ctx.lineTo(displayX - 25, timelineY + timelineHeight/2 + 5);
            }
            ctx.closePath();
            ctx.fill();
        }
        
        // 6. PREVIEW AREA
        const previewY = y + this.layout.preview;
        const previewHeight = this.layout.previewHeight;
        
        // Preview background
        ctx.fillStyle = "#222";
        ctx.fillRect(this.margin + 10, previewY, width - 20, previewHeight);
        
        // Set up clipping for preview area
        ctx.save();
        ctx.beginPath();
        ctx.rect(this.margin + 10, previewY, width - 20, previewHeight);
        ctx.clip();
        
        // Draw preview thumbnails for visible keyframes
        const visibleKeyframes = sortedFrames.filter(frame => {
            // Check if frame is in visible range with zoom
            return frame >= startFrame - 5 && frame <= endFrame + 5;
        });
        
        visibleKeyframes.forEach(frame => {
            // Skip the original position if dragging
            if (this.isDragging && frame === this.dragStartFrame) {
                return;
            }
            
            const kf = this.keyframes[frame];
            const x = frameToScreenX(frame);
            
            // Skip if off screen
            if (x < this.margin - 30 || x > this.margin + width + 30) {
                return;
            }
            
            
            // Draw thumbnail
            const thumbSize = 15; // 25% of original 60px
            
            // Background
            ctx.fillStyle = "#333";
            ctx.fillRect(x - thumbSize/2, previewY + 10, thumbSize, thumbSize);
            
            // Draw image if available
            if (this.keyframeImages[frame] && this.imageCache[frame]) {
                const img = this.imageCache[frame];
                if (img.complete && img.naturalWidth > 0) {
                    // Calculate aspect ratio correct drawing
                    const imgAspect = img.naturalWidth / img.naturalHeight;
                    let drawWidth = thumbSize;
                    let drawHeight = thumbSize;
                    let drawX = x - thumbSize/2;
                    let drawY = previewY + 10;
                    
                    if (imgAspect > 1) {
                        // Wider than tall
                        drawHeight = thumbSize / imgAspect;
                        drawY = previewY + 10 + (thumbSize - drawHeight) / 2;
                    } else {
                        // Taller than wide
                        drawWidth = thumbSize * imgAspect;
                        drawX = x - drawWidth/2;
                    }
                    
                    ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
                }
            } else if (this.keyframeImages[frame] && !this.imageCache[frame]) {
                // Image exists but not in cache - check if it's being dragged
                if (this.isDragging && this.dragStartFrame !== undefined) {
                    // While dragging, use the original frame's image if available
                    const draggedImg = this.imageCache[this.dragStartFrame];
                    if (draggedImg && draggedImg.complete) {
                        // Calculate aspect ratio correct drawing
                        const imgAspect = draggedImg.naturalWidth / draggedImg.naturalHeight;
                        let drawWidth = thumbSize;
                        let drawHeight = thumbSize;
                        let drawX = x - thumbSize/2;
                        let drawY = previewY + 10;
                        
                        if (imgAspect > 1) {
                            drawHeight = thumbSize / imgAspect;
                            drawY = previewY + 10 + (thumbSize - drawHeight) / 2;
                        } else {
                            drawWidth = thumbSize * imgAspect;
                            drawX = x - drawWidth/2;
                        }
                        
                        ctx.drawImage(draggedImg, drawX, drawY, drawWidth, drawHeight);
                    }
                } else {
                    // Not dragging - reload the image
                    this.loadImageToCache(frame, this.keyframeImages[frame]);
                    
                    // Show loading placeholder - removed overlay text
                }
            } else {
                // No image at all - removed overlay text
            }
            
            // Frame border
            ctx.strokeStyle = this.selectedKeyframe === frame ? "#ffcc00" : "#555";
            ctx.lineWidth = this.selectedKeyframe === frame ? 2 : 1;
            ctx.strokeRect(x - thumbSize/2, previewY + 10, thumbSize, thumbSize);
            
            // Frame number
            ctx.fillStyle = "#fff";
            ctx.font = "10px Arial";
            ctx.textAlign = "center";
            ctx.fillText(`F${frame}`, x, previewY + 30);
        });
        
        // Draw the dragged keyframe preview at its new position
        if ((this.isDragging || this.isDraggingBlock) && this.dragStartFrame !== undefined && this.dragTargetFrame !== undefined) {
            const frame = this.dragStartFrame;
            const kf = this.keyframes[frame];
            if (kf) {
                const x = frameToScreenX(this.dragTargetFrame);
                const thumbSize = 15; // 25% of original 60px
                
                // Background
                ctx.fillStyle = "#333";
                ctx.fillRect(x - thumbSize/2, previewY + 10, thumbSize, thumbSize);
                
                // Draw image if available
                // First check if we have the image in cache
                if (this.imageCache[frame]) {
                    const img = this.imageCache[frame];
                    if (img.complete && img.naturalWidth > 0) {
                        // Calculate aspect ratio correct drawing
                        const imgAspect = img.naturalWidth / img.naturalHeight;
                        let drawWidth = thumbSize;
                        let drawHeight = thumbSize;
                        let drawX = x - thumbSize/2;
                        let drawY = previewY + 10;
                        
                        if (imgAspect > 1) {
                            drawHeight = thumbSize / imgAspect;
                            drawY = previewY + 10 + (thumbSize - drawHeight) / 2;
                        } else {
                            drawWidth = thumbSize * imgAspect;
                            drawX = x - drawWidth/2;
                        }
                        
                        ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
                    }
                } else if (this.keyframeImages[frame]) {
                    // Image data exists but not loaded to cache yet - reload it
                    this.loadImageToCache(frame, this.keyframeImages[frame]);
                    
                    // Show loading placeholder - removed overlay text
                } else {
                    // No image at all - removed overlay text
                }
                
                // Frame border (highlighted for dragging)
                ctx.strokeStyle = "#ffcc00";
                ctx.lineWidth = 2;
                ctx.strokeRect(x - thumbSize/2, previewY + 10, thumbSize, thumbSize);
                
                // Frame number
                ctx.fillStyle = "#fff";
                ctx.font = "10px Arial";
                ctx.textAlign = "center";
                ctx.fillText(`F${this.dragTargetFrame}`, x, previewY + 30);
            }
        }
        
        // Restore context to remove preview clipping
        ctx.restore();
        
        // Current frame preview
        const currentKF = this.findKeyframeAtPosition(Math.floor(this.currentFrame));
        if (currentKF) {
            ctx.fillStyle = "#0099ff";
            ctx.fillRect(this.margin + width/2 - 45, previewY + 90, 90, 20);
            ctx.fillStyle = "#fff";
            ctx.font = "11px Arial";
            ctx.textAlign = "center";
            ctx.fillText(currentKF.isExact ? `Keyframe at F${currentKF.frame}` : `Holding F${currentKF.frame}`, this.margin + width/2, previewY + 103);
        }
        
        // DRAW SCALING HANDLES AFTER KEYFRAMES (so they appear on top)
        if (this.handleDrawInfo && this.handleDrawInfo.showHandles) {
            const { minX, maxX, blockTop, blockHeight, handleCenterY } = this.handleDrawInfo;
            
            // Check hover state for handles
            const leftHovered = this.leftHandleBounds && this.mousePos.x >= this.leftHandleBounds.x && 
                               this.mousePos.x <= this.leftHandleBounds.x + this.leftHandleBounds.width &&
                               this.mousePos.y >= this.leftHandleBounds.y && 
                               this.mousePos.y <= this.leftHandleBounds.y + this.leftHandleBounds.height;
            const rightHovered = this.rightHandleBounds && this.mousePos.x >= this.rightHandleBounds.x && 
                                this.mousePos.x <= this.rightHandleBounds.x + this.rightHandleBounds.width &&
                                this.mousePos.y >= this.rightHandleBounds.y && 
                                this.mousePos.y <= this.rightHandleBounds.y + this.rightHandleBounds.height;
            
            // Left handle (arrow pointing right)
            ctx.save();
            ctx.globalAlpha = this.isDraggingLeftHandle ? 0.9 : (leftHovered ? 0.8 : 0.6);
            ctx.fillStyle = this.isDraggingLeftHandle ? "#ffcc00" : (leftHovered ? "#ffffff" : "#dddddd");
            ctx.beginPath();
            ctx.moveTo(minX - 10, handleCenterY - 8); // Left point
            ctx.lineTo(minX - 2, handleCenterY);      // Right point (arrow tip)
            ctx.lineTo(minX - 10, handleCenterY + 8); // Bottom point
            ctx.closePath();
            ctx.fill();
            
            // Outline for better visibility
            ctx.strokeStyle = "#333333";
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();
            
            // Right handle (arrow pointing left)
            ctx.save();
            ctx.globalAlpha = this.isDraggingRightHandle ? 0.9 : (rightHovered ? 0.8 : 0.6);
            ctx.fillStyle = this.isDraggingRightHandle ? "#ffcc00" : (rightHovered ? "#ffffff" : "#dddddd");
            ctx.beginPath();
            ctx.moveTo(maxX + 10, handleCenterY - 8); // Right point
            ctx.lineTo(maxX + 2, handleCenterY);      // Left point (arrow tip)
            ctx.lineTo(maxX + 10, handleCenterY + 8); // Bottom point
            ctx.closePath();
            ctx.fill();
            
            // Outline for better visibility
            ctx.strokeStyle = "#333333";
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();
            
            // Show scale factor if actively scaling
            if ((this.isDraggingLeftHandle || this.isDraggingRightHandle) && this.scaleFactor !== 1.0) {
                ctx.save();
                ctx.fillStyle = "#ffcc00";
                ctx.font = "bold 14px Arial";
                ctx.textAlign = "center";
                ctx.strokeStyle = "#000000";
                ctx.lineWidth = 3;
                const scaleText = `${Math.round(this.scaleFactor * 100)}%`;
                const textX = (minX + maxX) / 2;
                const textY = blockTop - 10;
                ctx.strokeText(scaleText, textX, textY);
                ctx.fillText(scaleText, textX, textY);
                ctx.restore();
            }
        }
        
        // 7. INFO AND INSTRUCTIONS
        const infoY = y + this.layout.info;
        
        // Keyframe count with selection info
        ctx.fillStyle = "#fff";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        const keyframeCount = Object.keys(this.keyframes).length;
        const selectedCount = this.selectedKeyframes.size;
        
        if (selectedCount > 0) {
            ctx.fillText(`Keyframes: ${keyframeCount} (${selectedCount} selected)`, this.margin + 10, infoY);
        } else {
            ctx.fillText(`Keyframes: ${keyframeCount}`, this.margin + 10, infoY);
        }
        
        // Instructions
        ctx.fillStyle = "#666";
        ctx.font = "11px Arial";
        ctx.fillText("Drop images • Click+drag timeline scrub • Shift+drag range select • Shift+click move keyframes", 
                     this.margin + 10, infoY + 20);
        
        // Drop zone indicator when dragging
        if (this.isDraggingFile) {
            // Draw overlay focused on the timeline area
            const timelineY = y + this.layout.timeline;
            const timelineHeight = this.layout.timelineHeight;
            const overlayPadding = 20; // Extra padding around timeline
            
            // Semi-transparent blue overlay around timeline area
            ctx.fillStyle = "#0099ff33";
            ctx.fillRect(this.margin, timelineY - overlayPadding, width, timelineHeight + overlayPadding * 2);
            
            // Dashed border around timeline area
            ctx.strokeStyle = "#0099ff";
            ctx.lineWidth = 3;
            ctx.setLineDash([10, 5]);
            ctx.strokeRect(this.margin + 10, timelineY - overlayPadding + 5, width - 20, timelineHeight + overlayPadding * 2 - 10);
            ctx.setLineDash([]);
            
            // Center text in timeline area
            ctx.fillStyle = "#0099ff";
            ctx.font = "24px Arial";
            ctx.textAlign = "center";
            const textY = timelineY + timelineHeight / 2 + 8; // Center vertically in timeline (+8 for font baseline)
            ctx.fillText("Drop Images Here", this.margin + width/2, textY);
        }
        
        // Schedule animation refresh for pulsing boundary indicators
        if (this.batchSize > 0) {
            // Check if we have any missing boundary keyframes that need animation
            let hasMissingBoundaries = false;
            
            // Check frame 0 boundary
            if (!this.hasKeyframeNearBoundary(0)) {
                hasMissingBoundaries = true;
            }
            
            // Check batch end boundaries
            if (!hasMissingBoundaries) {
                for (let batch = 0; batch * this.batchSize < this.timeline_frames; batch++) {
                    const boundaryFrame = Math.min((batch + 1) * this.batchSize - 1, this.timeline_frames - 1);
                    if (!this.hasKeyframeNearBoundary(boundaryFrame)) {
                        hasMissingBoundaries = true;
                        break;
                    }
                }
            }
            
            // Only schedule animation if we have missing boundaries to animate
            if (hasMissingBoundaries) {
                if (!this._animationScheduled) {
                    this._animationScheduled = true;
                    requestAnimationFrame(() => {
                        this._animationScheduled = false;
                        // Trigger a redraw by updating the widget
                        if (this.node && this.node.graph && this.node.graph.canvas) {
                            this.node.graph.canvas.setDirty(true);
                        }
                    });
                }
            }
        }
        
        
        return this.height;
    }
    
    // Helper method to check if there's a keyframe near a boundary (within snap tolerance)
    hasKeyframeNearBoundary(boundaryFrame) {
        const snapTolerance = 0; // Only exact boundary frame counts
        
        for (let frame = boundaryFrame - snapTolerance; frame <= boundaryFrame + snapTolerance; frame++) {
            if (frame >= 0 && frame < this.timeline_frames && 
                this.keyframes[frame] && (this.keyframes[frame].enabled !== false)) {
                return true;
            }
        }
        return false;
    }
    
    // Helper method to check if a frame position is EXACTLY on any boundary
    isExactlyOnBoundary(frame) {
        // Check frame 0 boundary (start of first batch)
        if (frame === 0) {
            return true;
        }
        
        // Check batch end boundaries (last frame of each batch)
        // For batchSize=81: boundaries at frames 80, 161, 242, etc.
        if (this.batchSize > 0) {
            // Calculate which batch this would be the end of
            // Frame 80 = end of batch 0 (frames 0-80)
            // Frame 161 = end of batch 1 (frames 81-161)
            for (let batchNumber = 0; batchNumber < 20; batchNumber++) { // limit to 20 batches max
                const batchEndFrame = (batchNumber + 1) * this.batchSize - 1;
                if (batchEndFrame >= this.timeline_frames) {
                    break; // beyond timeline
                }
                if (frame === batchEndFrame) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    drawButton(ctx, x, y, width, height, text, color, disabled = false) {
        ctx.save();
        
        // Button shadow
        ctx.shadowBlur = disabled ? 0 : 4;
        ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
        ctx.shadowOffsetY = 2;
        
        // Button gradient background
        const buttonGradient = ctx.createLinearGradient(x, y, x, y + height);
        if (disabled) {
            buttonGradient.addColorStop(0, "#3a3a3a");
            buttonGradient.addColorStop(0.5, "#2a2a2a");
            buttonGradient.addColorStop(1, "#222222");
        } else {
            // Create gradient based on the button color
            const rgb = this.hexToRgb(color);
            buttonGradient.addColorStop(0, this.adjustBrightness(color, 30));
            buttonGradient.addColorStop(0.5, color);
            buttonGradient.addColorStop(1, this.adjustBrightness(color, -30));
        }
        
        ctx.fillStyle = buttonGradient;
        ctx.fillRect(x, y, width, height);
        
        // Reset shadow
        ctx.shadowBlur = 0;
        
        // Glass effect overlay
        const glassGradient = ctx.createLinearGradient(x, y, x, y + height * 0.5);
        glassGradient.addColorStop(0, "rgba(255, 255, 255, 0.2)");
        glassGradient.addColorStop(1, "transparent");
        ctx.fillStyle = glassGradient;
        ctx.fillRect(x, y, width, height * 0.5);
        
        // Button border
        ctx.strokeStyle = disabled ? "#555" : this.adjustBrightness(color, 40);
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);
        
        // Inner highlight
        ctx.strokeStyle = disabled ? "rgba(255, 255, 255, 0.05)" : "rgba(255, 255, 255, 0.15)";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x + 1, y + 1, width - 2, height - 2);
        
        // Text with subtle shadow
        ctx.shadowBlur = 2;
        ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
        ctx.shadowOffsetY = 1;
        
        ctx.fillStyle = disabled ? "#888" : "#fff";
        ctx.font = "bold 12px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, x + width/2, y + height/2);
        
        ctx.restore();
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }
    
    adjustBrightness(hexColor, percent) {
        const rgb = this.hexToRgb(hexColor);
        if (!rgb) return hexColor;
        
        const r = Math.max(0, Math.min(255, rgb.r + (rgb.r * percent / 100)));
        const g = Math.max(0, Math.min(255, rgb.g + (rgb.g * percent / 100)));
        const b = Math.max(0, Math.min(255, rgb.b + (rgb.b * percent / 100)));
        
        return `#${Math.round(r).toString(16).padStart(2, '0')}${Math.round(g).toString(16).padStart(2, '0')}${Math.round(b).toString(16).padStart(2, '0')}`;
    }
    
    drawPlaybackButton(ctx, x, y, width, height, type, color) {
        ctx.save();
        
        // Modern rounded button background
        const radius = 6;
        
        // Button shadow
        ctx.shadowBlur = 6;
        ctx.shadowColor = "rgba(0, 0, 0, 0.3)";
        ctx.shadowOffsetY = 2;
        
        // Create gradient background
        const buttonGradient = ctx.createLinearGradient(x, y, x, y + height);
        const rgb = this.hexToRgb(color);
        buttonGradient.addColorStop(0, this.adjustBrightness(color, 20));
        buttonGradient.addColorStop(0.5, color);
        buttonGradient.addColorStop(1, this.adjustBrightness(color, -20));
        
        // Draw rounded rectangle
        ctx.fillStyle = buttonGradient;
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        ctx.fill();
        
        // Reset shadow
        ctx.shadowBlur = 0;
        
        // Subtle inner highlight
        const highlightGradient = ctx.createLinearGradient(x, y, x, y + height * 0.4);
        highlightGradient.addColorStop(0, "rgba(255, 255, 255, 0.25)");
        highlightGradient.addColorStop(1, "transparent");
        ctx.fillStyle = highlightGradient;
        ctx.fill();
        
        // Draw border with subtle glow
        ctx.strokeStyle = this.adjustBrightness(color, 40);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Draw icon
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        const iconSize = width * 0.4;
        
        if (type === "play") {
            // Modern play triangle
            ctx.beginPath();
            ctx.moveTo(centerX - iconSize * 0.3, centerY - iconSize * 0.5);
            ctx.lineTo(centerX + iconSize * 0.4, centerY);
            ctx.lineTo(centerX - iconSize * 0.3, centerY + iconSize * 0.5);
            ctx.closePath();
            ctx.fill();
        } else if (type === "pause") {
            // Modern pause bars
            const barWidth = iconSize * 0.25;
            const barHeight = iconSize;
            const gap = iconSize * 0.3;
            
            ctx.fillRect(centerX - gap/2 - barWidth, centerY - barHeight/2, barWidth, barHeight);
            ctx.fillRect(centerX + gap/2, centerY - barHeight/2, barWidth, barHeight);
        } else if (type === "stop") {
            // Modern stop square with rounded corners
            const squareSize = iconSize * 0.7;
            const squareRadius = 2;
            
            ctx.beginPath();
            ctx.moveTo(centerX - squareSize/2 + squareRadius, centerY - squareSize/2);
            ctx.lineTo(centerX + squareSize/2 - squareRadius, centerY - squareSize/2);
            ctx.quadraticCurveTo(centerX + squareSize/2, centerY - squareSize/2, 
                                centerX + squareSize/2, centerY - squareSize/2 + squareRadius);
            ctx.lineTo(centerX + squareSize/2, centerY + squareSize/2 - squareRadius);
            ctx.quadraticCurveTo(centerX + squareSize/2, centerY + squareSize/2, 
                                centerX + squareSize/2 - squareRadius, centerY + squareSize/2);
            ctx.lineTo(centerX - squareSize/2 + squareRadius, centerY + squareSize/2);
            ctx.quadraticCurveTo(centerX - squareSize/2, centerY + squareSize/2, 
                                centerX - squareSize/2, centerY + squareSize/2 - squareRadius);
            ctx.lineTo(centerX - squareSize/2, centerY - squareSize/2 + squareRadius);
            ctx.quadraticCurveTo(centerX - squareSize/2, centerY - squareSize/2, 
                                centerX - squareSize/2 + squareRadius, centerY - squareSize/2);
            ctx.closePath();
            ctx.fill();
        }
        
        ctx.restore();
    }
    
    findKeyframeAtPosition(frame) {
        // Check exact match
        if (this.keyframes[frame]) {
            return { frame: frame, keyframe: this.keyframes[frame], isExact: true };
        }
        
        // Find previous keyframe for hold
        const frames = Object.keys(this.keyframes).map(Number).sort((a, b) => a - b);
        for (let i = frames.length - 1; i >= 0; i--) {
            if (frames[i] <= frame) {
                const kf = this.keyframes[frames[i]];
                if (kf.hold && frames[i] + kf.hold > frame) {
                    return { frame: frames[i], keyframe: kf, isExact: false };
                }
                return { frame: frames[i], keyframe: kf, isExact: false };
            }
        }
        
        return null;
    }
    
    clearAllStates() {
        // Clear all interaction states - used for recovery from frozen state
        this.isDragging = false;
        this.isBoxSelecting = false;
        this.isScrubbing = false;
        this.isDraggingZoom = false;
        this.isDraggingScroll = false;
        this.isDraggingLeftHandle = false;
        this.isDraggingRightHandle = false;
        this.isDraggingBlock = false;
        this.dragStartFrame = 0;
        this.dragTargetFrame = 0;
        this.dragStartX = 0;
        this.boxSelectStart = {x: 0, y: 0};
        this.boxSelectEnd = {x: 0, y: 0};
        this.edgeScrollDirection = 0;
        this.edgeScrollSpeed = 0;
        if (this.edgeScrollInterval) {
            clearInterval(this.edgeScrollInterval);
            this.edgeScrollInterval = null;
        }
        console.log("[clearAllStates] All interaction states cleared");
    }
    
    mouse(event, pos, node) {
        // Log ALL events to see what's happening
        if (event && (event.type === "mousedown" || event.type === "pointerdown" || event.type === "mouseup" || event.type === "pointerup")) {
            console.log(`[Mouse] ENTRY - Event type: ${event.type}, pos: ${pos}, event object exists: ${!!event}`);
        }
        
        const absoluteX = pos[0];
        const absoluteY = pos[1];
        
        // Get widget bounds - with safety check
        const widgetY = this.last_y !== undefined ? this.last_y : 0;
        const margin = this.margin;
        const width = node.size[0] - margin * 2;
        
        // COORDINATE VALIDATION: Check if last_y seems reasonable
        if (this.last_y !== undefined && Math.abs(this.last_y) > 10000) {
            console.warn(`[Mouse] WARNING: last_y=${this.last_y} seems invalid - using fallback coordinate calculation`);
            // Fallback: try to estimate widget Y from node position
            const estimatedY = node.pos ? node.pos[1] + 100 : 0; // rough estimate
            console.warn(`[Mouse] Using estimated widget Y: ${estimatedY}`);
            this.last_y = estimatedY;
        }
        
        // Convert to widget-relative coordinates
        const x = absoluteX;
        const y = absoluteY - widgetY;
        
        // Log mouse events for debugging
        if (event.type === "mousedown" || event.type === "pointerdown") {
            console.log(`[Mouse] Event: ${event.type}, pos=[${absoluteX}, ${absoluteY}], widget-relative=[${x}, ${y}]`);
            console.log(`[Mouse] Shift key: event.shiftKey=${event.shiftKey}, LiteGraph.shift_key=${LiteGraph.shift_key}`);
            console.log(`[Mouse] Event object keys: ${Object.keys(event).join(', ')}`);
            console.log(`[Mouse] Widget Y: ${widgetY}, last_y: ${this.last_y}, height: ${this.height}`);
            console.log(`[Mouse] Widget bounds: y=${widgetY}, height=${this.height}, in bounds=${absoluteY >= widgetY && absoluteY <= widgetY + this.height}`);
            
            // Check if we have keyframes to select
            const keyframeCount = Object.keys(this.keyframes).length;
            console.log(`[Mouse] Available keyframes: ${keyframeCount}, frames: ${Object.keys(this.keyframes).slice(0, 5).join(', ')}${keyframeCount > 5 ? '...' : ''}`);
        }
        
        // Check if mouse is within widget bounds
        if (absoluteY < widgetY || absoluteY > widgetY + this.height) {
            if (event.type === "mousedown" || event.type === "pointerdown") {
                console.log(`[Mouse] OUT OF BOUNDS - returning false`);
            }
            // Clear hover state when mouse leaves widget
            if (this.hoverKeyframe !== null) {
                this.hoverKeyframe = null;
                this.node.setDirtyCanvas(true);
            }
            return false;
        }
        
        // Ensure widget has focus for proper event handling
        if (event.type === "mousedown" || event.type === "pointerdown") {
            console.log("[Mouse] Ensuring widget focus");
            // Force focus to this widget and clear any stuck states
            if (node.graph && node.graph.canvas) {
                // Store previous widget to detect focus issues
                const prevWidget = node.graph.canvas.node_widget;
                node.graph.canvas.node_widget = [node, this];
                console.log("[Mouse] Set node_widget focus");
                
                // If we had a different widget focused, ensure we're fully taking over
                if (prevWidget && (prevWidget[0] !== node || prevWidget[1] !== this)) {
                    console.log("[Mouse] Taking focus from another widget - clearing all states");
                    // Clear ALL states when taking focus from another widget
                    this.clearAllStates();
                }
            }
            
            // Reset any stuck drag states that might interfere with selection
            if (!this.isDragging && !this.isDraggingBlock && !this.isScrubbing) {
                this.dragStartFrame = 0;
                this.dragTargetFrame = 0;
                console.log("[Mouse] Reset drag states");
            }
            
            // Track mouse down time to detect frozen states
            this.lastMouseDownTime = Date.now();
        }
        
        // Handle different mouse events
        if (event.type === "mousedown" || event.type === "pointerdown") {
            console.log("[Mouse] About to call handleMouseDown");
            const result = this.handleMouseDown(x, y, width, event);
            console.log(`[Mouse] handleMouseDown returned: ${result}`);
            
            // Additional recovery: If handleMouseDown returns false but we're in the timeline area,
            // force event capture to prevent the event from being lost
            if (!result) {
                const trackY = this.layout.timeline;
                const trackHeight = this.layout.timelineHeight;
                if (y >= trackY - 10 && y <= trackY + trackHeight + 20) {
                    console.log("[Mouse] RECOVERY: Mouse down in timeline area but not handled - forcing capture");
                    if (event.stopPropagation) event.stopPropagation();
                    if (event.preventDefault) event.preventDefault();
                    // Force clear all states and try to handle as timeline click
                    this.clearAllStates();
                    this.isScrubbing = true;
                    this.dragStartX = x;
                    const visibleFrames = this.timeline_frames / this.zoomLevel;
                    const frameX = ((x - this.margin) / (width - this.margin * 2)) * visibleFrames + this.scrollOffset;
                    this.currentFrame = Math.max(0, Math.min(this.timeline_frames - 1, Math.round(frameX)));
                    this.node.setDirtyCanvas(true);
                    return true;
                }
            }
            
            return result;
        } else if (event.type === "mousemove" || event.type === "pointermove") {
            // Don't log mouse move as it's too verbose
            return this.handleMouseMove(x, y, width);
        } else if (event.type === "mouseup" || event.type === "pointerup") {
            console.log("[Mouse] About to call handleMouseUp");
            const result = this.handleMouseUp(x, y, width);
            console.log(`[Mouse] handleMouseUp returned: ${result}`);
            return result;
        } else if (event.type === "mousewheel" || event.type === "wheel") {
            return this.handleMouseWheel(event, x, y, width);
        }
        
        console.log(`[Mouse] Unhandled event type: ${event.type}`);
        return false;
    }
    
    canPlaceKeyframe(frame, excludeFrame = null) {
        if (frame < 0 || frame >= this.timeline_frames) return false;
        
        // No gap enforcement - always allow placement
        return true;
    }
    
    togglePlayback() {
        this.isPlaying = !this.isPlaying;
        if (this.isPlaying) {
            this.lastPlayTime = performance.now();
            this.startPlayback();
        }
        this.node.setDirtyCanvas(true);
    }
    
    stopPlayback() {
        this.isPlaying = false;
        // Reset to start of visible range when zoomed, or 0 when not zoomed
        if (this.zoomLevel > 1.0) {
            this.currentFrame = this.scrollOffset;
        } else {
            this.currentFrame = 0;
        }
        this.node.setDirtyCanvas(true);
    }
    
    startPlayback() {
        const animate = () => {
            if (!this.isPlaying) return;
            
            const now = performance.now();
            const deltaTime = (now - this.lastPlayTime) / 1000;
            this.lastPlayTime = now;
            
            // Calculate visible frame range when zoomed
            const visibleFrames = this.timeline_frames / this.zoomLevel;
            const startFrame = this.scrollOffset;
            const endFrame = Math.min(this.timeline_frames, startFrame + visibleFrames);
            
            this.currentFrame += deltaTime * this.fps;
            
            // Loop within visible range when zoomed, or full timeline when not zoomed
            if (this.zoomLevel > 1.0) {
                // Zoomed in - loop within visible range
                if (this.currentFrame >= endFrame) {
                    this.currentFrame = startFrame;
                } else if (this.currentFrame < startFrame) {
                    this.currentFrame = startFrame;
                }
            } else {
                // Not zoomed - loop entire timeline
                if (this.currentFrame >= this.timeline_frames) {
                    this.currentFrame = 0;
                }
            }
            
            this.node.setDirtyCanvas(true);
            requestAnimationFrame(animate);
        };
        
        requestAnimationFrame(animate);
    }
    
    // showFileDialog replaced by handleAddKeyframes
    
    // async loadKeyframeImages replaced by the non-async version below
    
    // These old methods are replaced by the handle* versions
    
    removeKeyframe(frame) {
        delete this.keyframes[frame];
        if (this.selectedKeyframe === frame) {
            this.selectedKeyframe = null;
        }
        this.selectedKeyframes.delete(frame);
        this.updateNodeProperty();
        this.node.setDirtyCanvas(true);
    }
    
    deleteSelectedKeyframes() {
        if (this.selectedKeyframe !== null) {
            this.removeKeyframe(this.selectedKeyframe);
        }
        
        for (const frame of this.selectedKeyframes) {
            delete this.keyframes[frame];
        }
        
        this.selectedKeyframes.clear();
        this.updateNodeProperty();
        this.node.setDirtyCanvas(true);
    }
    
    clearTimeline() {
        console.log("[Timeline] Clearing entire timeline - starting fresh");
        
        // Clear all keyframes and images
        this.keyframes = {};
        this.keyframeImages = {};
        
        // Clear global image cache for this node
        if (window._comfyui_wan_keyframes && window._comfyui_wan_keyframes[this.node.id]) {
            window._comfyui_wan_keyframes[this.node.id] = {};
        }
        
        // Clear selections
        this.selectedKeyframe = null;
        this.selectedKeyframes.clear();
        
        // Hide reload prompt
        this.showReloadPrompt = false;
        
        // Update node and redraw
        this.updateNodeProperty();
        this.node.setDirtyCanvas(true);
        
        console.log("[Timeline] Timeline cleared successfully");
    }
    
    toggleSelectedKeyframes() {
        if (this.selectedKeyframe !== null) {
            const kf = this.keyframes[this.selectedKeyframe];
            if (kf) {
                kf.enabled = !kf.enabled;
            }
        }
        
        for (const frame of this.selectedKeyframes) {
            const kf = this.keyframes[frame];
            if (kf) {
                kf.enabled = !kf.enabled;
            }
        }
        
        this.updateNodeProperty();
        this.node.setDirtyCanvas(true);
    }
    
    setHoldDuration() {
        if (this.selectedKeyframe === null) return;
        
        const current = this.keyframes[this.selectedKeyframe].hold || 1;
        const duration = prompt("Hold duration (frames):", current);
        
        if (duration && !isNaN(duration)) {
            this.keyframes[this.selectedKeyframe].hold = Math.max(1, parseInt(duration));
            this.updateNodeProperty();
            this.node.setDirtyCanvas(true);
        }
    }
    
    saveConfig() {
        const config = {
            frames: this.timeline_frames,
            fps: this.fps,
            batch_size: this.batchSize,
            ignore_held_frames_mask: !!this.ignore_held_frames_mask,
            nextKeyframeID: this.nextKeyframeID,
            keyframes: this.keyframes
        };
        // Include frame_darkness if widget exists
        const fdWidget = this.node.widgets.find(w => w.name === "frame_darkness");
        if (fdWidget && typeof fdWidget.value !== 'undefined') {
            config.frame_darkness = fdWidget.value;
        }
        
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'keyframe_timeline.json';
        a.click();
        URL.revokeObjectURL(url);
    }
    
    loadConfig() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (ev) => {
                    try {
                        const config = JSON.parse(ev.target.result);
                        // LEGACY FLOAT SUPPORT: Convert float frames to integer
                        const framesValue = Math.round(config.frames || 81);
                        if (config.frames && framesValue !== config.frames) {
                            console.log(`[loadConfig] Converting legacy float frames ${config.frames} to integer ${framesValue}`);
                        }
                        this.timeline_frames = framesValue;
                        
                        // LEGACY FLOAT SUPPORT: Convert float fps to integer  
                        const fpsValue = Math.round(config.fps || 15);
                        if (config.fps && fpsValue !== config.fps) {
                            console.log(`[loadConfig] Converting legacy float fps ${config.fps} to integer ${fpsValue}`);
                        }
                        this.fps = fpsValue;

                        // Optional settings
                        if (typeof config.batch_size !== 'undefined') {
                            this.batchSize = Math.round(config.batch_size);
                        }
                        if (typeof config.ignore_held_frames_mask !== 'undefined') {
                            this.ignore_held_frames_mask = !!config.ignore_held_frames_mask;
                        }
                        if (typeof config.nextKeyframeID !== 'undefined') {
                            this.nextKeyframeID = config.nextKeyframeID;
                        }
                        
                        this.keyframes = config.keyframes || {};
                        
                        // Update node widgets
                        const framesWidget = this.node.widgets.find(w => w.name === "timeline_frames");
                        if (framesWidget) {
                            framesWidget.value = this.timeline_frames;
                            if (typeof framesWidget.callback === 'function') framesWidget.callback(this.timeline_frames);
                        }
                        
                        const fpsWidget = this.node.widgets.find(w => w.name === "fps");
                        if (fpsWidget) {
                            fpsWidget.value = this.fps;
                            if (typeof fpsWidget.callback === 'function') fpsWidget.callback(this.fps);
                        }

                        const batchWidget = this.node.widgets.find(w => w.name === "batch_size");
                        if (batchWidget && typeof this.batchSize !== 'undefined') {
                            batchWidget.value = this.batchSize;
                            if (typeof batchWidget.callback === 'function') batchWidget.callback(this.batchSize);
                        }

                        const fdWidget = this.node.widgets.find(w => w.name === "frame_darkness");
                        if (fdWidget && typeof config.frame_darkness !== 'undefined') {
                            fdWidget.value = config.frame_darkness;
                            if (typeof fdWidget.callback === 'function') fdWidget.callback(config.frame_darkness);
                        }
                        
                        this.updateNodeProperty();
                        this.node.setDirtyCanvas(true);
                    } catch (err) {
                        alert("Failed to load config: " + err.message);
                    }
                };
                reader.readAsText(file);
            }
        };
        
        input.click();
    }
    
    updateNodeProperty() {
        // Debounce updates to reduce lag
        if (this._updateTimeout) {
            clearTimeout(this._updateTimeout);
        }
        
        this._updateTimeout = setTimeout(() => {
            this._doUpdateNodeProperty();
        }, 100); // Delay updates by 100ms to batch multiple changes
    }
    
    _doUpdateNodeProperty() {
        // Store only metadata in properties (no image data)
        const keyframeMetadata = {
            frames: this.timeline_frames,
            keyframes: this.keyframes,  // This now contains only metadata, no images
            ignore_held_frames_mask: this.ignore_held_frames_mask
        };
        
        this.node.properties.keyframeData = keyframeMetadata;
        
        // Don't save images to localStorage - we'll reload from file paths
        
        // Store image data in a global variable to avoid localStorage quota
        if (!window._comfyui_wan_keyframes) {
            window._comfyui_wan_keyframes = {};
        }
        if (!window._comfyui_wan_keyframes[this.node.id]) {
            window._comfyui_wan_keyframes[this.node.id] = {};
        }
        
        const globalStorage = window._comfyui_wan_keyframes[this.node.id];
        
        // Clear old entries that are no longer valid
        const validKeys = new Set();
        
        // Store all current images by both frame AND ID
        for (const frame in this.keyframeImages) {
            if (this.keyframeImages[frame]) {
                // Store by frame
                globalStorage[frame] = this.keyframeImages[frame];
                validKeys.add(frame);
                
                // Store by ID if keyframe has one
                const kf = this.keyframes[frame];
                if (kf && kf.id) {
                    globalStorage[kf.id] = this.keyframeImages[frame];
                    validKeys.add(kf.id);
                }
            }
        }
        
        // IMPORTANT: Do NOT clean up old entries from global storage
        // This was causing issues with undo/redo where valid images were being deleted
        // The global storage should persist across undo operations
        // Images may be referenced after undo even if not in current validKeys
        
        // Log what we have stored for debugging
        console.log(`[updateNodeProperty] Stored ${validKeys.size} images in global storage for node ${this.node.id}`);
        console.log(`  - Total entries in global storage: ${Object.keys(globalStorage).length}`);
        console.log(`  - Valid keys: ${Array.from(validKeys).slice(0, 5).join(', ')}${validKeys.size > 5 ? '...' : ''}`);
        
        // Update widget with ONLY METADATA (no images) to avoid localStorage quota issues
        let keyframeDataWidget = this.keyframeDataWidget || this.node.widgets.find(w => w.name === "keyframe_data");
        if (keyframeDataWidget) {
            // Store only metadata in widget (no image data)
            const metadataOnly = {
                frames: this.timeline_frames,
                keyframes: this.keyframes,  // This contains only path, enabled, hold - no images
                ignore_held_frames_mask: this.ignore_held_frames_mask
            };
            
            const jsonData = JSON.stringify(metadataOnly);
            
            // Set the widget value directly
            keyframeDataWidget.value = jsonData;
            
            // Also try setting internal properties as backup
            if (keyframeDataWidget._value !== undefined) {
                keyframeDataWidget._value = jsonData;
            }
            if (keyframeDataWidget._internalValue !== undefined) {
                keyframeDataWidget._internalValue = jsonData;
            }
            
            console.log(`Updated keyframe_data widget with ${Object.keys(this.keyframes).length} keyframes metadata (images stored separately)`);
        } else {
            console.warn("keyframe_data widget not found!");
        }
    }
    
    computePosition(node) {
        let y = LiteGraph.NODE_TITLE_HEIGHT + 10;
        
        for (const w of node.widgets) {
            if (w === this) break;
            
            if (w.computeSize) {
                y += w.computeSize(node.size[0])[1] + 4;
            } else if (w.size) {
                y += w.size[1] + 4;
            } else {
                y += LiteGraph.NODE_WIDGET_HEIGHT + 4;
            }
        }
        
        return y;
    }
    
    computeSize(width) {
        return [width, this.height];
    }
    
    handleMouseDown(x, y, width, event = null) {
        console.log(`[handleMouseDown] Entry: x=${x}, y=${y}, width=${width}`);
        
        // FIRST: Check if clicking reload button when prompt is shown
        if (this.showReloadPrompt && this.reloadButtonBounds) {
            // The reload button bounds are stored relative to the widget origin (already include this.last_y)
            // The incoming x,y are already widget-relative, so we just need to adjust x by margin
            console.log(`[handleMouseDown] Checking reload button: bounds=${JSON.stringify(this.reloadButtonBounds)}, x=${x}, y=${y}`);
            console.log(`[handleMouseDown] Adjusted bounds: x=${this.reloadButtonBounds.x - this.margin}, y=${this.reloadButtonBounds.y}, w=${this.reloadButtonBounds.width}, h=${this.reloadButtonBounds.height}`);
            
            if (this.isPointInRect(x, y, this.reloadButtonBounds.x - this.margin, this.reloadButtonBounds.y, 
                                    this.reloadButtonBounds.width, this.reloadButtonBounds.height)) {
                console.log("[Timeline] Reload Images button clicked!");
                // First try automatic reload
                this.attemptAutoReload().then(success => {
                    if (!success) {
                        // If auto-reload failed, show file browser
                        this.handleReloadImages();
                    }
                });
                return true;
            }
            
            // Check if clicking "New Timeline" button
            if (this.newTimelineButtonBounds && this.isPointInRect(x, y, this.newTimelineButtonBounds.x - this.margin, this.newTimelineButtonBounds.y, 
                                    this.newTimelineButtonBounds.width, this.newTimelineButtonBounds.height)) {
                console.log("[Timeline] New Timeline button clicked!");
                this.clearTimeline();
                return true;
            }
            // If reload prompt is shown but click is outside button, ignore the click
            return false;
        }
        
        const buttonHeight = this.layout.buttonHeight;
        const buttonY = this.layout.controls;
        const trackY = this.layout.timeline;
        const trackHeight = this.layout.timelineHeight;
        
        console.log(`[handleMouseDown] Timeline track bounds: trackY=${trackY}, trackHeight=${trackHeight}, check range: ${trackY - 10} to ${trackY + trackHeight + 20}`);
        
        // Use centralized layout positions
        const deleteRowYPos = this.layout.deleteRow;
        const deleteRowHeightSize = this.layout.deleteRowHeight;
        const playbackY = this.layout.playback;
        
        // Check control buttons
        let buttonX = this.margin + 10;
        
        // Add Keyframes button
        if (this.isPointInRect(x, y, buttonX, buttonY, 100, buttonHeight)) {
            console.log("Add Keyframes button area clicked!");
            this.handleAddKeyframes();
            return true;
        }
        buttonX += 105;
        
        // Replace Selected button
        if (this.isPointInRect(x, y, buttonX, buttonY, 100, buttonHeight) && this.selectedKeyframe !== null) {
            this.handleReplaceSelected();
            return true;
        }
        buttonX += 105;
        
        // Batch Replace button
        if (this.isPointInRect(x, y, buttonX, buttonY, 110, buttonHeight)) {
            this.handleBatchReplace();
            return true;
        }
        buttonX += 115;

        // Removed Reload button and spacing
        
        // Clear All button
        if (this.isPointInRect(x, y, buttonX, buttonY, 90, buttonHeight)) {
            this.handleClearAll();
            return true;
        }
        buttonX += 95;
        
        // --- NEW ROW: Delete Selected Keyframes ---
        const deleteRowHeight = 25;
        const deleteRowButtonX = this.margin + 10;
        const deleteRowButtonWidth = 110;
        if (this.isPointInRect(x, y, deleteRowButtonX, deleteRowYPos, deleteRowButtonWidth, deleteRowHeight)) {
            if (this.selectedKeyframes && this.selectedKeyframes.size > 0) {
                this.deleteSelectedKeyframes();
            }
            return true;
        }
        
        // Key Hold Duration +/- buttons - work on selected keyframes
        const holdControlX = deleteRowButtonX + 120;
        const valueBoxX = holdControlX + 100;
        const valueBoxWidth = 40;
        const buttonSize = deleteRowHeight;
        const minusButtonX = valueBoxX + valueBoxWidth + 5;
        const plusButtonX = minusButtonX + buttonSize + 2;
        
        // Get selected keyframes to modify
        const selectedFrames = [];
        if (this.selectedKeyframe !== null) {
            selectedFrames.push(this.selectedKeyframe);
        }
        if (this.selectedKeyframes.size > 0) {
            selectedFrames.push(...Array.from(this.selectedKeyframes));
        }
        // Remove duplicates
        const uniqueSelectedFrames = [...new Set(selectedFrames)];
        
        // Minus button click
        if (this.isPointInRect(x, y, minusButtonX, deleteRowYPos, buttonSize, buttonSize)) {
            if (uniqueSelectedFrames.length > 0) {
                let modified = false;
                uniqueSelectedFrames.forEach(frame => {
                    if (this.keyframes[frame]) {
                        const currentHold = this.keyframes[frame].hold || 1;
                        if (currentHold > 1) {
                            this.keyframes[frame].hold = currentHold - 1;
                            modified = true;
                        }
                    }
                });
                if (modified) {
                    this.updateNodeProperty();
                    this.node.setDirtyCanvas(true);
                }
            }
            return true;
        }
        
        // Plus button click  
        if (this.isPointInRect(x, y, plusButtonX, deleteRowYPos, buttonSize, buttonSize)) {
            if (uniqueSelectedFrames.length > 0) {
                let modified = false;
                uniqueSelectedFrames.forEach(frame => {
                    if (this.keyframes[frame]) {
                        const currentHold = this.keyframes[frame].hold || 1;
                        if (currentHold < 100) {
                            this.keyframes[frame].hold = currentHold + 1;
                            modified = true;
                        }
                    }
                });
                if (modified) {
                    this.updateNodeProperty();
                    this.node.setDirtyCanvas(true);
                }
            }
            return true;
        }
        
        // Checkbox click - calculate checkbox position same as in draw
        const checkboxX = plusButtonX + buttonSize + 15;
        const checkboxSize = 15;
        const checkboxY = deleteRowYPos + (deleteRowHeight - checkboxSize) / 2;
        
        if (this.isPointInRect(x, y, checkboxX, checkboxY, checkboxSize, checkboxSize)) {
            this.ignore_held_frames_mask = !this.ignore_held_frames_mask;
            this.updateNodeProperty();
            this.node.setDirtyCanvas(true);
            return true;
        }
        // --- END NEW ROW ---
        
        // Save Config button
        if (this.isPointInRect(x, y, buttonX, buttonY, 80, buttonHeight)) {
            this.handleSaveConfig();
            return true;
        }
        buttonX += 85;
        
        // Display last used folder info if available
        if (this.lastUsedFolder) {
            ctx.save();
            ctx.font = "10px Arial";
            ctx.fillStyle = "#888";
            ctx.textAlign = "left";
            ctx.fillText(this.lastUsedFolder, this.margin + 10, buttonY + buttonHeight + 20);
            ctx.restore();
        }
        
        // Load Config button
        if (this.isPointInRect(x, y, buttonX, buttonY, 80, buttonHeight)) {
            this.handleLoadConfig();
            return true;
        }
        
        // Check playback controls
        const playX = this.margin + 10;
        
        // Play/Pause button
        if (this.isPointInRect(x, y, playX, playbackY, 30, 30)) {
            this.togglePlayback();
            return true;
        }
        
        // Stop button
        if (this.isPointInRect(x, y, playX + 35, playbackY, 30, 30)) {
            this.stopPlayback();
            return true;
        }
        
        // Check zoom slider - use centralized position
        const zoomY = this.layout.zoom;
        const zoomWidth = width - 20;
        const zoomNormalized = (this.zoomLevel - this.minZoom) / (this.maxZoom - this.minZoom);
        const zoomX = this.margin + 10 + zoomNormalized * (zoomWidth - 20);
        
        // Check if clicking on zoom slider handle
        if (this.isPointInRect(x, y, zoomX - 10, zoomY - 2, 20, 24)) {
            this.isDraggingZoom = true;
            return true;
        }
        
        // Check if clicking on zoom track
        if (this.isPointInRect(x, y, this.margin + 10, zoomY, zoomWidth, 20)) {
            // Jump zoom to clicked position
            const clickNormalized = (x - this.margin - 10) / (zoomWidth - 20);
            const newZoom = this.minZoom + clickNormalized * (this.maxZoom - this.minZoom);
            this.setZoomLevel(newZoom);
            this.isDraggingZoom = true;
            return true;
        }
        
        // Check horizontal scrollbar - use centralized position
        const scrollY = this.layout.scroll;
        const scrollHeight = 15;
        
        if (this.zoomLevel > 1.0) {
            const visibleFrames = this.timeline_frames / this.zoomLevel;
            const maxScroll = this.timeline_frames - visibleFrames;
            const thumbWidth = Math.max(30, (visibleFrames / this.timeline_frames) * zoomWidth);
            const thumbX = this.margin + 10 + (this.scrollOffset / maxScroll) * (zoomWidth - thumbWidth);
            
            // Check if clicking on scroll thumb
            if (this.isPointInRect(x, y, thumbX, scrollY, thumbWidth, scrollHeight)) {
                this.isDraggingScroll = true;
                this.scrollDragStartX = x;
                this.scrollDragStartOffset = this.scrollOffset;
                return true;
            }
            
            // Check if clicking on scroll track - don't jump, just start dragging from current position
            if (this.isPointInRect(x, y, this.margin + 10, scrollY, zoomWidth, scrollHeight)) {
                // Start dragging but don't change scroll position yet
                this.isDraggingScroll = true;
                this.scrollDragStartX = x;
                this.scrollDragStartOffset = this.scrollOffset;
                this.node.setDirtyCanvas(true);
                return true;
            }
        }
        
        // Check timeline track
        const trackX = this.margin + 10;
        const trackWidth = width - 20;
        
        // Get modifier keys first to determine priority
        const shiftPressed = (event && event.shiftKey) || 
                            LiteGraph.shift_key || 
                            (event && event.modifiers && event.modifiers.shift) ||
                            (window.event && window.event.shiftKey);
        
        // If shift is NOT pressed and we have multiple keyframes selected, check handles first
        // If shift IS pressed, skip handle checks to prioritize keyframe selection
        if (!shiftPressed && this.selectedKeyframes.size >= 2) {
            // Check if clicking on arrow handles (only when not shift-clicking)
            if (this.leftHandleBounds && this.isPointInRect(x, y, this.leftHandleBounds.x, this.leftHandleBounds.y, 
                                                             this.leftHandleBounds.width, this.leftHandleBounds.height)) {
                console.log("[Timeline] Left handle clicked!");
                // Clicking on left handle - start scaling from right pivot
                this.isDraggingLeftHandle = true;
                this.skewStartX = x;
                
                // Store original positions of all selected keyframes
                this.skewStartPositions = {};
                const frames = [];
                for (const frame of this.selectedKeyframes) {
                    this.skewStartPositions[frame] = frame;
                    frames.push(frame);
                }
                
                // Store initial bounds
                this.skewStartMinFrame = Math.min(...frames);
                this.skewStartMaxFrame = Math.max(...frames);
                this.skewStartSpan = this.skewStartMaxFrame - this.skewStartMinFrame;
                
                this.node.setDirtyCanvas(true);
                return true;
            }
            
            if (this.rightHandleBounds && this.isPointInRect(x, y, this.rightHandleBounds.x, this.rightHandleBounds.y, 
                                                              this.rightHandleBounds.width, this.rightHandleBounds.height)) {
                console.log("[Timeline] Right handle clicked!");
                // Clicking on right handle - start scaling from left pivot
                this.isDraggingRightHandle = true;
                this.skewStartX = x;
                
                // Store original positions of all selected keyframes
                this.skewStartPositions = {};
                const frames = [];
                for (const frame of this.selectedKeyframes) {
                    this.skewStartPositions[frame] = frame;
                    frames.push(frame);
                }
                
                // Store initial bounds
                this.skewStartMinFrame = Math.min(...frames);
                this.skewStartMaxFrame = Math.max(...frames);
                this.skewStartSpan = this.skewStartMaxFrame - this.skewStartMinFrame;
                
                this.node.setDirtyCanvas(true);
                return true;
            }
        }
        
        // Check if clicking on the wooden block (but not when shift is pressed for selection)
        if (!shiftPressed && this.blockBounds && this.isPointInRect(x, y, this.blockBounds.minX, this.blockBounds.blockTop, 
                                                    this.blockBounds.maxX - this.blockBounds.minX, this.blockBounds.blockHeight)) {
            // Clicking on the block - start dragging the entire block
            this.isDraggingBlock = true;
            this.dragStartX = x;
            
            // Helper function to convert screen X to frame with zoom (same as in mouse move)
            const screenXToFrame = (screenX) => {
                const visibleFrames = this.timeline_frames / this.zoomLevel;
                const normalizedX = (screenX - trackX) / trackWidth;
                return Math.round(this.scrollOffset + normalizedX * visibleFrames);
            };
            
            // Calculate the drag start frame based on where we actually clicked
            this.dragStartFrame = screenXToFrame(x);
            this.dragTargetFrame = this.dragStartFrame;
            
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        // Remove this duplicate check - it's handled below with proper priority
        
        // Helper function to convert screen X to frame with zoom
        const screenXToFrame = (screenX) => {
            const visibleFrames = this.timeline_frames / this.zoomLevel;
            const normalizedX = (screenX - trackX) / trackWidth;
            return Math.round(this.scrollOffset + normalizedX * visibleFrames);
        };
        
        // Helper function to convert frame to screen X position with zoom
        const frameToScreenX = (frame) => {
            const visibleFrames = this.timeline_frames / this.zoomLevel;
            const normalizedFrame = (frame - this.scrollOffset) / visibleFrames;
            return trackX + normalizedFrame * trackWidth;
        };
        
        // Check if clicking on the timeline track - but handle shift mode first
        console.log(`[handleMouseDown] About to check timeline click: x=${x}, y=${y}, trackX=${trackX}, trackY=${trackY}, trackWidth=${trackWidth}, trackHeight=${trackHeight}`);
        console.log(`[handleMouseDown] Timeline bounds check: ${trackX} <= ${x} <= ${trackX + trackWidth} && ${trackY - 10} <= ${y} <= ${trackY + trackHeight + 20}`);
        
        if (this.isPointInRect(x, y, trackX, trackY - 10, trackWidth, trackHeight + 20)) {
            // Get modifier keys - check multiple sources for shift key state
            const shiftPressed = (event && event.shiftKey) || 
                                LiteGraph.shift_key || 
                                (event && event.modifiers && event.modifiers.shift) ||
                                (window.event && window.event.shiftKey);
            
            console.log(`[Timeline Click] REACHED! x=${x}, y=${y}, shiftPressed=${shiftPressed}`);
            console.log(`[Timeline Click] Shift key sources: event.shiftKey=${event && event.shiftKey}, LiteGraph.shift_key=${LiteGraph.shift_key}, window.event.shiftKey=${window.event && window.event.shiftKey}`);
            console.log(`[Timeline State] isDragging=${this.isDragging}, isBoxSelecting=${this.isBoxSelecting}, isScrubbing=${this.isScrubbing}`);
            console.log(`[Timeline State] selectedKeyframes.size=${this.selectedKeyframes.size}, selectedKeyframe=${this.selectedKeyframe}`);
            
            // Force recalculation of keyframe positions when shift is pressed
            // This helps prevent the "bug out" issue
            if (shiftPressed) {
                this.node.setDirtyCanvas(true);
            }
            
            if (shiftPressed) {
                console.log("[Timeline] SHIFT MODE: Entering selection mode");
                // SHIFT MODE: Selection operations take priority
                const clickedKeyframe = this.getKeyframeAt(x, y, trackX, trackY - 10, trackWidth);
                console.log(`[Timeline] Clicked keyframe: ${clickedKeyframe}`);
                
                if (clickedKeyframe !== null) {
                    // Shift+Click on keyframe: Better selection logic
                    if (this.selectedKeyframes.size === 0 && this.selectedKeyframe === null) {
                        // No existing selection - just select this keyframe
                        console.log(`[Timeline] No existing selection - selecting keyframe ${clickedKeyframe}`);
                        this.selectedKeyframe = clickedKeyframe;
                        this.selectedKeyframes.add(clickedKeyframe);
                    } else if (this.selectedKeyframes.has(clickedKeyframe)) {
                        // Already selected - remove from selection
                        console.log(`[Timeline] Keyframe ${clickedKeyframe} already selected - removing from selection`);
                        this.selectedKeyframes.delete(clickedKeyframe);
                        if (this.selectedKeyframe === clickedKeyframe) {
                            // If it was the primary selection, set a new primary
                            this.selectedKeyframe = this.selectedKeyframes.size > 0 ? 
                                Array.from(this.selectedKeyframes)[0] : null;
                            console.log(`[Timeline] New primary selection: ${this.selectedKeyframe}`);
                        }
                    } else {
                        // Not selected - add to selection
                        console.log(`[Timeline] Adding keyframe ${clickedKeyframe} to selection`);
                        this.selectedKeyframes.add(clickedKeyframe);
                        // Make it the primary selection
                        this.selectedKeyframe = clickedKeyframe;
                    }
                    console.log(`[Timeline] After selection: selectedKeyframes=${Array.from(this.selectedKeyframes)}, selectedKeyframe=${this.selectedKeyframe}`);
                    
                    // Log detailed state for debugging
                    console.log(`[Timeline] Detailed state after selection:`);
                    console.log(`  - this.selectedKeyframes.size = ${this.selectedKeyframes.size}`);
                    console.log(`  - this.selectedKeyframe = ${this.selectedKeyframe}`);
                    console.log(`  - this.isDragging = ${this.isDragging}`);
                    console.log(`  - this.isBoxSelecting = ${this.isBoxSelecting}`);
                } else {
                    // Shift+Click in empty space: Start box selection
                    console.log("[Timeline] Shift+Click in empty space - starting box selection");
                    // First ensure no stuck states
                    this.isDragging = false;
                    this.isDraggingBlock = false;
                    this.isScrubbing = false;
                    
                    this.isBoxSelecting = true;
                    this.boxSelectStart = {x, y};
                    this.boxSelectEnd = {x, y};
                    // Store original scrubber position to restore after selection
                    this.preSelectionCurrentFrame = this.currentFrame;
                    // Clear any existing selection - we'll select what's in the box
                    this.selectedKeyframe = null;
                    this.selectedKeyframes.clear();
                    console.log(`[Timeline] Box selection started at x=${x}, y=${y}`);
                }
                this.node.setDirtyCanvas(true);
                return true;
            } else {
                console.log("[Timeline] NORMAL MODE: Entering scrub mode");
                // NORMAL MODE: Always scrub, never select keyframes
                
                // Clear any existing selection
                if ((this.selectedKeyframes.size > 0 || this.selectedKeyframe !== null)) {
                    console.log("[Timeline] Clearing selection in normal mode");
                    this.selectedKeyframe = null;
                    this.selectedKeyframes.clear();
                    // Clear scaling handles
                    this.handleDrawInfo = null;
                    this.leftHandleBounds = null;
                    this.rightHandleBounds = null;
                    this.node.setDirtyCanvas(true);
                }
                
                // Check scrubber
                const scrubberX = frameToScreenX(this.currentFrame);
                console.log(`[Timeline] Scrubber check: scrubberX=${scrubberX}, x=${x}, diff=${Math.abs(x - scrubberX)}`);
                
                // Check if clicking on scrubber handle (more generous hit area)
                if (Math.abs(x - scrubberX) < 15 && y >= trackY - 30 && y <= trackY + trackHeight + 20) {
                    console.log("[Timeline] Clicking on scrubber handle - starting scrub");
                    this.isScrubbing = true;
                    this.stopPlayback();
                    return true;
                } else {
                    // Start scrubbing from clicked position with zoom
                    const frame = screenXToFrame(x);
                    console.log(`[Timeline] Starting scrub from clicked position - frame=${frame}`);
                    this.isScrubbing = true;
                    this.stopPlayback();
                    this.currentFrame = Math.max(0, Math.min(this.timeline_frames - 1, frame));
                    this.node.setDirtyCanvas(true);
                    return true;
                }
            }
        }
        
        // Check if clicking in dark area below timeline to deselect keys
        const timelineEnd = this.layout.timeline + this.layout.timelineHeight;
        const previewStart = this.layout.preview;
        if (y >= timelineEnd && y <= previewStart && (this.selectedKeyframes.size > 0 || this.selectedKeyframe !== null)) {
            this.selectedKeyframe = null;
            this.selectedKeyframes.clear();
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        return false;
    }
    
    handleMouseMove(x, y, width) {
        const trackX = this.margin + 10;
        const trackWidth = width - 20;
        const trackY = this.layout.timeline;
        
        // Store mouse position
        this.mousePos = {x, y};
        
        // Handle zoom slider dragging
        if (this.isDraggingZoom) {
            const zoomWidth = width - 20;
            const clickNormalized = Math.max(0, Math.min(1, (x - this.margin - 10) / (zoomWidth - 20)));
            const newZoom = this.minZoom + clickNormalized * (this.maxZoom - this.minZoom);
            this.setZoomLevel(newZoom);
            return true;
        }
        
        // Handle scrollbar dragging
        if (this.isDraggingScroll && this.zoomLevel > 1.0) {
            const zoomWidth = width - 20;
            const visibleFrames = this.timeline_frames / this.zoomLevel;
            const maxScroll = this.timeline_frames - visibleFrames;
            const thumbWidth = Math.max(30, (visibleFrames / this.timeline_frames) * zoomWidth);
            
            // Calculate delta movement from start position
            const deltaX = x - this.scrollDragStartX;
            // Convert pixel delta to scroll units
            const scrollDelta = (deltaX / (zoomWidth - thumbWidth)) * maxScroll;
            // Apply delta to initial scroll position
            this.scrollOffset = Math.max(0, Math.min(maxScroll, this.scrollDragStartOffset + scrollDelta));
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        // Handle box selection dragging
        if (this.isBoxSelecting) {
            this.boxSelectEnd = {x, y};
            
            // Update current frame to follow mouse during selection
            if (x >= trackX && x <= trackX + trackWidth) {
                const visibleFrames = this.timeline_frames / this.zoomLevel;
                const normalizedX = (x - trackX) / trackWidth;
                const frame = this.scrollOffset + normalizedX * visibleFrames;
                this.currentFrame = Math.round(Math.max(0, Math.min(this.timeline_frames - 1, frame)));
            }
            
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        // Calculate visible frame range when zoomed
        const visibleFrames = this.timeline_frames / this.zoomLevel;
        const startFrame = Math.floor(this.scrollOffset);
        const endFrame = Math.ceil(this.scrollOffset + visibleFrames);
        
        // Helper function to convert screen X to frame with zoom
        const screenXToFrame = (screenX) => {
            const normalizedX = (screenX - trackX) / trackWidth;
            return Math.round(this.scrollOffset + normalizedX * visibleFrames);
        };
        
        // Check for hover over keyframes
        const prevHover = this.hoverKeyframe;
        this.hoverKeyframe = this.getKeyframeAt(x, y, trackX, trackY, trackWidth);
        if (prevHover !== this.hoverKeyframe) {
            this.node.setDirtyCanvas(true);
        }
        
        // Handle skewing/scaling
        if (this.isDraggingLeftHandle || this.isDraggingRightHandle) {
            const selectedFrames = Array.from(this.selectedKeyframes);
            if (selectedFrames.length < 2 || this.skewStartSpan === 0) {
                // Need at least 2 keyframes with span to scale
                return true;
            }
            
            // Calculate delta movement in pixels and convert to frames
            const deltaX = x - this.skewStartX;
            const visibleFrames = this.timeline_frames / this.zoomLevel;
            const deltaFrames = (deltaX / trackWidth) * visibleFrames;
            
            if (this.isDraggingLeftHandle) {
                // Left handle: scale from right edge (max stays fixed)
                const newMin = Math.max(0, this.skewStartMinFrame + deltaFrames);
                let newSpan = this.skewStartMaxFrame - newMin;
                
                // Ensure minimum span where all keyframes are at least 1 frame apart
                const minRequiredSpan = this.selectedKeyframes.size - 1;
                if (newSpan < minRequiredSpan) {
                    newSpan = minRequiredSpan;
                    this.scaleFactor = newSpan / this.skewStartSpan;
                } else {
                    this.scaleFactor = newSpan / this.skewStartSpan;
                }
                
                // Check for internal collisions and limit scale if needed
                const newPositionMap = new Map();
                let hasInternalCollision = false;
                
                for (const [originalFrameStr, originalPos] of Object.entries(this.skewStartPositions)) {
                    const originalFrame = parseInt(originalFrameStr);
                    const relativePos = (originalPos - this.skewStartMinFrame) / this.skewStartSpan;
                    const newFrame = Math.round((this.skewStartMaxFrame - newSpan) + relativePos * newSpan);
                    
                    // Check for collision with non-selected keyframe
                    if (newFrame >= 0 && newFrame < this.timeline_frames) {
                        const existingKf = this.keyframes[newFrame];
                        if (existingKf && !this.selectedKeyframes.has(newFrame) && originalFrame !== newFrame) {
                            hasInternalCollision = true;
                            break;
                        }
                    }
                    
                    // Check for internal collision (multiple selected keyframes to same position)
                    if (newPositionMap.has(newFrame)) {
                        hasInternalCollision = true;
                        break;
                    }
                    newPositionMap.set(newFrame, originalFrame);
                }
                
                // If collision detected, find maximum safe scale
                if (hasInternalCollision) {
                    // Binary search for maximum safe scale factor
                    let minScale = minRequiredSpan / this.skewStartSpan;
                    let maxScale = this.scaleFactor;
                    let safeScale = minScale;
                    
                    for (let i = 0; i < 10; i++) { // 10 iterations should be enough
                        const testScale = (minScale + maxScale) / 2;
                        const testSpan = this.skewStartSpan * testScale;
                        const testPositionMap = new Map();
                        let testHasCollision = false;
                        
                        for (const [originalFrameStr, originalPos] of Object.entries(this.skewStartPositions)) {
                            const originalFrame = parseInt(originalFrameStr);
                            const relativePos = (originalPos - this.skewStartMinFrame) / this.skewStartSpan;
                            const testFrame = Math.round((this.skewStartMaxFrame - testSpan) + relativePos * testSpan);
                            
                            if (testFrame >= 0 && testFrame < this.timeline_frames) {
                                const existingKf = this.keyframes[testFrame];
                                if (existingKf && !this.selectedKeyframes.has(testFrame) && originalFrame !== testFrame) {
                                    testHasCollision = true;
                                    break;
                                }
                            }
                            
                            if (testPositionMap.has(testFrame)) {
                                testHasCollision = true;
                                break;
                            }
                            testPositionMap.set(testFrame, originalFrame);
                        }
                        
                        if (testHasCollision) {
                            maxScale = testScale;
                        } else {
                            safeScale = testScale;
                            minScale = testScale;
                        }
                    }
                    
                    this.scaleFactor = safeScale;
                }
            } else if (this.isDraggingRightHandle) {
                // Right handle: scale from left edge (min stays fixed)
                const newMax = Math.min(this.timeline_frames - 1, this.skewStartMaxFrame + deltaFrames);
                let newSpan = newMax - this.skewStartMinFrame;
                
                // Ensure minimum span where all keyframes are at least 1 frame apart
                const minRequiredSpan = this.selectedKeyframes.size - 1;
                if (newSpan < minRequiredSpan) {
                    newSpan = minRequiredSpan;
                    this.scaleFactor = newSpan / this.skewStartSpan;
                } else {
                    this.scaleFactor = newSpan / this.skewStartSpan;
                }
                
                // Check for internal collisions and limit scale if needed
                const newPositionMap = new Map();
                let hasInternalCollision = false;
                
                for (const [originalFrameStr, originalPos] of Object.entries(this.skewStartPositions)) {
                    const originalFrame = parseInt(originalFrameStr);
                    const relativePos = (originalPos - this.skewStartMinFrame) / this.skewStartSpan;
                    const newFrame = Math.round(this.skewStartMinFrame + relativePos * newSpan);
                    
                    // Check for collision with non-selected keyframe
                    if (newFrame >= 0 && newFrame < this.timeline_frames) {
                        const existingKf = this.keyframes[newFrame];
                        if (existingKf && !this.selectedKeyframes.has(newFrame) && originalFrame !== newFrame) {
                            hasInternalCollision = true;
                            break;
                        }
                    }
                    
                    // Check for internal collision (multiple selected keyframes to same position)
                    if (newPositionMap.has(newFrame)) {
                        hasInternalCollision = true;
                        break;
                    }
                    newPositionMap.set(newFrame, originalFrame);
                }
                
                // If collision detected, find maximum safe scale
                if (hasInternalCollision) {
                    // Binary search for maximum safe scale factor
                    let minScale = minRequiredSpan / this.skewStartSpan;
                    let maxScale = this.scaleFactor;
                    let safeScale = minScale;
                    
                    for (let i = 0; i < 10; i++) { // 10 iterations should be enough
                        const testScale = (minScale + maxScale) / 2;
                        const testSpan = this.skewStartSpan * testScale;
                        const testPositionMap = new Map();
                        let testHasCollision = false;
                        
                        for (const [originalFrameStr, originalPos] of Object.entries(this.skewStartPositions)) {
                            const originalFrame = parseInt(originalFrameStr);
                            const relativePos = (originalPos - this.skewStartMinFrame) / this.skewStartSpan;
                            const testFrame = Math.round(this.skewStartMinFrame + relativePos * testSpan);
                            
                            if (testFrame >= 0 && testFrame < this.timeline_frames) {
                                const existingKf = this.keyframes[testFrame];
                                if (existingKf && !this.selectedKeyframes.has(testFrame) && originalFrame !== testFrame) {
                                    testHasCollision = true;
                                    break;
                                }
                            }
                            
                            if (testPositionMap.has(testFrame)) {
                                testHasCollision = true;
                                break;
                            }
                            testPositionMap.set(testFrame, originalFrame);
                        }
                        
                        if (testHasCollision) {
                            maxScale = testScale;
                        } else {
                            safeScale = testScale;
                            minScale = testScale;
                        }
                    }
                    
                    this.scaleFactor = safeScale;
                }
            }
            
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        if ((this.isDragging && this.selectedKeyframe !== null) || this.isDraggingBlock) {
            // Calculate new frame position with zoom
            const frame = screenXToFrame(x);
            const clampedFrame = Math.max(0, Math.min(this.timeline_frames - 1, frame));
            
            // Calculate offset from the dragged keyframe's original position
            const offset = clampedFrame - this.dragStartFrame;
            
            // Check if we can move all selected keyframes by this offset
            const canMoveGroup = this.canMoveKeyframeGroup(offset);
            
            if (canMoveGroup) {
                // Store the target position for the dragged keyframe
                this.dragTargetFrame = clampedFrame;
            }
            
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        if (this.isScrubbing) {
            // Update current frame based on mouse position with zoom
            const frame = screenXToFrame(x);
            this.currentFrame = Math.max(0, Math.min(this.timeline_frames - 1, frame));
            
            // Edge scrolling removed - no automatic scrolling when scrubbing
            
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        // Change cursor based on what we're hovering over
        if (this.hoverKeyframe !== null) {
            document.body.style.cursor = "grab";
        } else {
            document.body.style.cursor = "default";
        }
        
        return false;
    }
    
    handleMouseUp(x, y, width) {
        console.log(`[handleMouseUp] Called with x=${x}, y=${y}`);
        console.log(`[handleMouseUp] State: isDragging=${this.isDragging}, isBoxSelecting=${this.isBoxSelecting}, isScrubbing=${this.isScrubbing}`);
        console.log(`[handleMouseUp] Selection state: selectedKeyframes.size=${this.selectedKeyframes.size}, selectedKeyframe=${this.selectedKeyframe}`);
        
        // RECOVERY MECHANISM: Detect if we got a mouse up without a corresponding mouse down
        // This indicates the widget is in a frozen state
        // BUT: Don't trigger recovery if we just selected keyframes (shift+click selection)
        const hasJustSelectedKeyframe = (this.selectedKeyframes.size > 0 || this.selectedKeyframe !== null) && 
                                       this.lastMouseDownTime && (Date.now() - this.lastMouseDownTime < 500);
        
        console.log(`[handleMouseUp] Recovery check:`);
        console.log(`  - hasJustSelectedKeyframe = ${hasJustSelectedKeyframe}`);
        console.log(`  - lastMouseDownTime = ${this.lastMouseDownTime}`);
        console.log(`  - time since mouse down = ${this.lastMouseDownTime ? Date.now() - this.lastMouseDownTime : 'N/A'}ms`);
        
        if (!this.isDragging && !this.isBoxSelecting && !this.isScrubbing && 
            !this.isDraggingZoom && !this.isDraggingScroll && 
            !this.isDraggingLeftHandle && !this.isDraggingRightHandle && !this.isDraggingBlock &&
            !hasJustSelectedKeyframe) {
            console.log("[handleMouseUp] WARNING: Mouse up without any active drag state - widget may be frozen!");
            console.log("[handleMouseUp] Attempting recovery by forcing widget focus...");
            
            // Force widget to regain focus
            if (this.node && this.node.graph && this.node.graph.canvas) {
                this.node.graph.canvas.node_widget = [this.node, this];
                console.log("[handleMouseUp] Forced widget focus reset");
            }
            
            // Clear any stuck selection states
            this.selectedKeyframes.clear();
            this.selectedKeyframe = null;
            this.hoverKeyframe = null;
            // Clear scaling handles
            this.handleDrawInfo = null;
            this.leftHandleBounds = null;
            this.rightHandleBounds = null;
            
            // Reset all interaction flags just in case
            this.isDragging = false;
            this.isBoxSelecting = false;
            this.isScrubbing = false;
            this.isDraggingZoom = false;
            this.isDraggingScroll = false;
            this.isDraggingLeftHandle = false;
            this.isDraggingRightHandle = false;
            this.isDraggingBlock = false;
            
            // Force a redraw
            this.node.setDirtyCanvas(true);
            
            console.log("[handleMouseUp] Recovery complete - all states reset");
            return true;
        }
        
        // Stop all dragging states
        if (this.isDraggingZoom) {
            console.log("[handleMouseUp] Ending zoom drag");
            this.isDraggingZoom = false;
            return true;
        }
        
        if (this.isDraggingScroll) {
            console.log("[handleMouseUp] Ending scroll drag");
            this.isDraggingScroll = false;
            return true;
        }
        
        // Handle skewing/scaling release
        if (this.isDraggingLeftHandle || this.isDraggingRightHandle) {
            const selectedFrames = Array.from(this.selectedKeyframes);
            if (selectedFrames.length >= 2 && this.scaleFactor !== 1.0) {
                // Calculate final positions for all keyframes
                const originalPositions = Object.values(this.skewStartPositions).sort((a, b) => a - b);
                const minOriginal = Math.min(...originalPositions);
                const maxOriginal = Math.max(...originalPositions);
                const originalSpan = maxOriginal - minOriginal;
                
                let newPositions = {};
                
                if (this.isDraggingLeftHandle) {
                    // Calculate final positions with right edge as pivot
                    const newSpan = originalSpan * this.scaleFactor;
                    const newMin = maxOriginal - newSpan;
                    
                    for (const [frameStr, originalPos] of Object.entries(this.skewStartPositions)) {
                        const frame = parseInt(frameStr);
                        const relativePos = (originalPos - minOriginal) / originalSpan;
                        const newFrame = Math.round(newMin + relativePos * newSpan);
                        newPositions[frame] = Math.max(0, Math.min(this.timeline_frames - 1, newFrame));
                    }
                } else if (this.isDraggingRightHandle) {
                    // Calculate final positions with left edge as pivot
                    const newSpan = originalSpan * this.scaleFactor;
                    
                    for (const [frameStr, originalPos] of Object.entries(this.skewStartPositions)) {
                        const frame = parseInt(frameStr);
                        const relativePos = (originalPos - minOriginal) / originalSpan;
                        const newFrame = Math.round(minOriginal + relativePos * newSpan);
                        newPositions[frame] = Math.max(0, Math.min(this.timeline_frames - 1, newFrame));
                    }
                }
                
                // Check for collisions
                let hasCollision = false;
                const targetSet = new Set(Object.values(newPositions));
                
                for (const newFrame of targetSet) {
                    if (this.keyframes[newFrame] && !this.selectedKeyframes.has(newFrame)) {
                        hasCollision = true;
                        break;
                    }
                }
                
                // Only proceed if no collisions
                if (!hasCollision) {
                    // Store all data temporarily
                    const tempStorage = {};
                    for (const oldFrame of selectedFrames) {
                        tempStorage[oldFrame] = {
                            keyframe: this.keyframes[oldFrame],
                            image: this.keyframeImages[oldFrame],
                            cache: this.imageCache[oldFrame]
                        };
                        delete this.keyframes[oldFrame];
                        delete this.keyframeImages[oldFrame];
                        delete this.imageCache[oldFrame];
                    }
                    
                    // Place at new positions
                    for (const [oldFrameStr, newFrame] of Object.entries(newPositions)) {
                        const oldFrame = parseInt(oldFrameStr);
                        const data = tempStorage[oldFrame];
                        if (data) {
                            this.keyframes[newFrame] = data.keyframe;
                            if (data.image) this.keyframeImages[newFrame] = data.image;
                            if (data.cache) this.imageCache[newFrame] = data.cache;
                        }
                    }
                    
                    // Update selected keyframes set
                    this.selectedKeyframes.clear();
                    for (const newFrame of Object.values(newPositions)) {
                        this.selectedKeyframes.add(newFrame);
                    }
                    
                    // Update primary selected keyframe
                    if (this.selectedKeyframe !== null && newPositions[this.selectedKeyframe] !== undefined) {
                        this.selectedKeyframe = newPositions[this.selectedKeyframe];
                    }
                    
                    // Update global storage
                    if (window._comfyui_wan_keyframes && window._comfyui_wan_keyframes[this.node.id]) {
                        const globalStorage = window._comfyui_wan_keyframes[this.node.id];
                        
                        // Move images in global storage
                        const tempImages = {};
                        for (const oldFrame of selectedFrames) {
                            if (globalStorage[oldFrame]) {
                                tempImages[oldFrame] = globalStorage[oldFrame];
                                delete globalStorage[oldFrame];
                            }
                        }
                        
                        for (const [oldFrameStr, newFrame] of Object.entries(newPositions)) {
                            const oldFrame = parseInt(oldFrameStr);
                            if (tempImages[oldFrame]) {
                                globalStorage[newFrame] = tempImages[oldFrame];
                            }
                        }
                    }
                    
                    this.updateNodeProperty();
                }
            }
            
            // Reset scaling state
            this.isDraggingLeftHandle = false;
            this.isDraggingRightHandle = false;
            this.scaleFactor = 1.0;
            this.skewStartPositions = {};
            this.skewStartMinFrame = 0;
            this.skewStartMaxFrame = 0;
            this.skewStartSpan = 0;
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        if ((this.isDragging || this.isDraggingBlock) && this.dragStartFrame !== undefined && this.dragTargetFrame !== undefined) {
            // Commit the move for all selected keyframes
            const offset = this.dragTargetFrame - this.dragStartFrame;
            
            if (offset !== 0) {
                // Get all selected keyframes and move them
                const selectedFrames = Array.from(this.selectedKeyframes);
                const moveOperations = [];
                
                // First, check if ALL moves are valid (no collisions)
                const targetPositions = new Set();
                let hasCollision = false;
                
                for (const oldFrame of selectedFrames) {
                    const newFrame = oldFrame + offset;
                    
                    // Check bounds
                    if (newFrame < 0 || newFrame >= this.timeline_frames) {
                        hasCollision = true;
                        break;
                    }
                    
                    // Check for collision with non-selected keyframes
                    if (this.keyframes[newFrame] && !this.selectedKeyframes.has(newFrame)) {
                        hasCollision = true;
                        break;
                    }
                    
                    // Check for internal collisions (two selected keyframes moving to same spot)
                    if (targetPositions.has(newFrame)) {
                        hasCollision = true;
                        break;
                    }
                    
                    targetPositions.add(newFrame);
                    moveOperations.push({oldFrame, newFrame});
                }
                
                // Only proceed if there are no collisions
                if (!hasCollision && moveOperations.length > 0) {
                    // Create a temporary storage for all keyframe data
                    const tempStorage = {};
                    
                    // First pass: Store all data in temp storage and delete from original positions
                    for (const {oldFrame} of moveOperations) {
                        tempStorage[oldFrame] = {
                            keyframe: this.keyframes[oldFrame],
                            image: this.keyframeImages[oldFrame],
                            cache: this.imageCache[oldFrame]
                        };
                        delete this.keyframes[oldFrame];
                        delete this.keyframeImages[oldFrame];
                        delete this.imageCache[oldFrame];
                    }
                    
                    // Second pass: Place all keyframes at their new positions
                    for (const {oldFrame, newFrame} of moveOperations) {
                        const data = tempStorage[oldFrame];
                        this.keyframes[newFrame] = data.keyframe;
                        if (data.image) this.keyframeImages[newFrame] = data.image;
                        if (data.cache) this.imageCache[newFrame] = data.cache;
                    }
                    
                    // Update global storage to match new positions
                    if (window._comfyui_wan_keyframes && window._comfyui_wan_keyframes[this.node.id]) {
                        const globalStorage = window._comfyui_wan_keyframes[this.node.id];
                        
                        // First pass: Store images in temp and delete from old positions
                        const tempImages = {};
                        for (const {oldFrame} of moveOperations) {
                            if (globalStorage[oldFrame]) {
                                tempImages[oldFrame] = globalStorage[oldFrame];
                                delete globalStorage[oldFrame];
                            }
                        }
                        
                        // Second pass: Place images at new positions
                        for (const {oldFrame, newFrame} of moveOperations) {
                            if (tempImages[oldFrame]) {
                                globalStorage[newFrame] = tempImages[oldFrame];
                            }
                        }
                    }
                    
                    // Update selection to new positions
                    this.selectedKeyframes.clear();
                    for (const {newFrame} of moveOperations) {
                        this.selectedKeyframes.add(newFrame);
                    }
                    
                    // Update primary selected keyframe to new position
                    if (this.selectedKeyframe !== null) {
                        this.selectedKeyframe = this.selectedKeyframe + offset;
                    }
                    
                    this.updateNodeProperty();
                }
            }
            
            this.isDragging = false;
            this.isDraggingBlock = false;
            this.dragStartX = 0;
            this.dragStartFrame = 0;
            this.dragTargetFrame = 0;
            
            // Force boundary indicator recalculation after drag operations
            // This ensures the boundary indicators reflect the new keyframe positions
            setTimeout(() => {
                this.node.setDirtyCanvas(true);
            }, 1);
            
            return true;
        }
        
        // Handle box selection completion
        if (this.isBoxSelecting) {
            console.log("[handleMouseUp] Completing box selection");
            this.completeBoxSelection(width);
            this.isBoxSelecting = false;
            
            // Restore original scrubber position after selection
            if (this.preSelectionCurrentFrame !== undefined) {
                this.currentFrame = this.preSelectionCurrentFrame;
                this.preSelectionCurrentFrame = undefined;
                console.log(`[handleMouseUp] Restored scrubber to frame ${this.currentFrame}`);
            }
            
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        if (this.isScrubbing) {
            console.log("[handleMouseUp] Ending scrubbing");
            this.isScrubbing = false;
            
            // Edge scrolling removed - no cleanup needed
            
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        console.log("[handleMouseUp] No active state to handle, returning false");
        return false;
    }
    
    setZoomLevel(newZoom) {
        const oldZoom = this.zoomLevel;
        this.zoomLevel = Math.max(this.minZoom, Math.min(this.maxZoom, newZoom));
        
        // When zooming, try to keep the current frame in view
        if (this.zoomLevel !== oldZoom) {
            const visibleFramesOld = this.timeline_frames / oldZoom;
            const visibleFramesNew = this.timeline_frames / this.zoomLevel;
            const centerFrame = this.scrollOffset + visibleFramesOld / 2;
            
            // Adjust scroll to keep center frame centered
            this.scrollOffset = centerFrame - visibleFramesNew / 2;
            
            // Clamp scroll offset
            const maxScroll = Math.max(0, this.timeline_frames - visibleFramesNew);
            this.scrollOffset = Math.max(0, Math.min(maxScroll, this.scrollOffset));
            
            this.node.setDirtyCanvas(true);
        }
    }
    
    handleMouseWheel(event, x, y, width) {
        // Zoom functionality could be added here
        return false;
    }
    
    isPointInRect(x, y, rx, ry, rw, rh) {
        return x >= rx && x <= rx + rw && y >= ry && y <= ry + rh;
    }
    
    getKeyframeAt(x, y, trackX, trackY, trackWidth) {
        const trackHeight = this.layout.timelineHeight;
        
        // Calculate visible frame range when zoomed
        const timeline_frames_int = Math.round(this.timeline_frames); // Ensure integer
        const visibleFrames = timeline_frames_int / this.zoomLevel;
        const startFrame = Math.floor(this.scrollOffset);
        const endFrame = Math.ceil(this.scrollOffset + visibleFrames);
        
        console.log(`[getKeyframeAt] Looking for keyframe at x=${x}, y=${y}`);
        console.log(`[getKeyframeAt] Track bounds: x=${trackX}, y=${trackY}, width=${trackWidth}, height=${trackHeight}`);
        console.log(`[getKeyframeAt] Visible frames: ${startFrame} to ${endFrame}`);
        
        // COORDINATE VALIDATION: Early bounds check to prevent invalid hit detection
        if (y < trackY - 20 || y > trackY + trackHeight + 20) {
            console.log(`[getKeyframeAt] Mouse Y coordinate ${y} outside valid track bounds ${trackY - 20} to ${trackY + trackHeight + 20} - returning null`);
            return null;
        }
        
        // Helper function to convert frame to screen X position with zoom
        const frameToScreenX = (frame) => {
            const normalizedFrame = (frame - this.scrollOffset) / visibleFrames;
            return trackX + normalizedFrame * trackWidth;
        };
        
        let closestFrame = null;
        let closestDistance = Infinity;
        
        for (const frame in this.keyframes) {
            const frameNum = parseInt(frame);
            
            // Skip if not visible
            if (frameNum < startFrame - 5 || frameNum > endFrame + 5) continue;
            
            const kf = this.keyframes[frameNum];
            const kfX = frameToScreenX(frameNum);
            
            // Calculate diamond dimensions (same as drawing code)
            const isSelected = this.selectedKeyframe === frameNum || this.selectedKeyframes.has(frameNum);
            const isHovered = this.hoverKeyframe === frameNum;
            const baseSize = isSelected ? 24 : (isHovered ? 20 : 17);
            const fixedWidth = 4; // Fixed horizontal width for visual display
            const centerY = trackY + trackHeight/2;
            
            // Precise diamond hit detection using mathematical boundaries
            // Diamond vertices: top(x, y-size), right(x+width, y), bottom(x, y+size), left(x-width, y)
            const dx = Math.abs(x - kfX);
            const dy = Math.abs(y - centerY);
            
            // Diamond boundary equation: |dx/width| + |dy/height| <= 1
            // Use a wider hit area than the visual width for easier clicking
            const hitWidth = 12; // 3x wider than visual for better hit detection
            const tolerance = 1.2; // 20% more generous
            const diamondDistance = dx / hitWidth + dy / baseSize;
            
            // Only log if we're close to hitting
            if (diamondDistance <= tolerance * 2) {
                console.log(`[getKeyframeAt] Frame ${frameNum}: kfX=${kfX}, centerY=${centerY}, dx=${dx}, dy=${dy}, distance=${diamondDistance.toFixed(2)}, tolerance=${tolerance}`);
            }
            
            if (diamondDistance <= tolerance) {
                if (diamondDistance < closestDistance) {
                    closestDistance = diamondDistance;
                    closestFrame = frameNum;
                }
            }
        }
        
        console.log(`[getKeyframeAt] Found keyframe: ${closestFrame}`);
        return closestFrame;
    }
    
    completeBoxSelection(width) {
        const trackX = this.margin + 10;
        const trackWidth = width - 20;
        const trackY = this.layout.timeline;
        const trackHeight = this.layout.timelineHeight;
        
        // Calculate horizontal range bounds (only X coordinates matter)
        const minX = Math.min(this.boxSelectStart.x, this.boxSelectEnd.x);
        const maxX = Math.max(this.boxSelectStart.x, this.boxSelectEnd.x);
        
        console.log(`[completeBoxSelection] Box bounds: minX=${minX}, maxX=${maxX}`);
        console.log(`[completeBoxSelection] Start: x=${this.boxSelectStart.x}, y=${this.boxSelectStart.y}`);
        console.log(`[completeBoxSelection] End: x=${this.boxSelectEnd.x}, y=${this.boxSelectEnd.y}`);
        
        // Helper function to convert frame to screen X position with zoom
        const frameToScreenX = (frame) => {
            const visibleFrames = this.timeline_frames / this.zoomLevel;
            const normalizedFrame = (frame - this.scrollOffset) / visibleFrames;
            return trackX + normalizedFrame * trackWidth;
        };
        
        // Find keyframes within the horizontal range selection
        const selectedFrames = [];
        
        for (const frame in this.keyframes) {
            const frameNum = parseInt(frame);
            const kfX = frameToScreenX(frameNum);
            
            // Check if keyframe is within the horizontal range (ignore Y coordinates)
            if (kfX >= minX && kfX <= maxX) {
                console.log(`[completeBoxSelection] Frame ${frameNum} at x=${kfX} is within box`);
                selectedFrames.push(frameNum);
            }
        }
        
        console.log(`[completeBoxSelection] Found ${selectedFrames.length} keyframes in box: ${selectedFrames}`);
        
        // Update selection - replace any existing selection with box contents
        if (selectedFrames.length > 0) {
            // Clear existing selection and set to box contents
            this.selectedKeyframes.clear();
            selectedFrames.forEach(frame => {
                this.selectedKeyframes.add(frame);
            });
            
            // Set primary selection to first selected frame
            this.selectedKeyframe = selectedFrames[0];
            console.log(`[completeBoxSelection] Set primary selection to ${this.selectedKeyframe}`);
            
        } else {
            // No keyframes in box - clear selection
            console.log(`[completeBoxSelection] No keyframes in box - clearing selection`);
            this.selectedKeyframes.clear();
            this.selectedKeyframe = null;
        }
        
        console.log(`[completeBoxSelection] Final selection: selectedKeyframes=${Array.from(this.selectedKeyframes)}, selectedKeyframe=${this.selectedKeyframe}`);
    }
    
    canMoveKeyframeTo(fromFrame, toFrame) {
        if (fromFrame === toFrame) {
            return true;
        }
        
        // Check if target position is occupied
        if (this.keyframes[toFrame]) {
            return false;
        }
        
        // No gap enforcement - allow moving anywhere
        return true;
    }
    
    canMoveKeyframeGroup(offset) {
        // Get all selected keyframes (including the primary selection)
        const selectedFrames = Array.from(this.selectedKeyframes);
        
        // Check if all keyframes can move by the given offset
        for (const frame of selectedFrames) {
            const newFrame = frame + offset;
            
            // Check bounds
            if (newFrame < 0 || newFrame >= this.timeline_frames) {
                return false;
            }
            
            // Check if target position would be occupied by a keyframe that's not in our selection
            if (this.keyframes[newFrame] && !selectedFrames.includes(newFrame)) {
                return false;
            }
        }
        
        return true;
    }
    
    handleKeyDown(e) {
        // Handle keyboard shortcuts for timeline operations
        if (e.ctrlKey || e.metaKey) {
            switch (e.key.toLowerCase()) {
                case 'a':
                    // Ctrl+A: Select all keyframes
                    e.preventDefault();
                    this.selectAllKeyframes();
                    return true;
                    
                case 'c':
                    // Ctrl+C: Copy selected keyframes (future feature)
                    e.preventDefault();
                    return true;
                    
                case 'v':
                    // Ctrl+V: Paste keyframes (future feature)
                    e.preventDefault();
                    return true;
            }
        } else {
            switch (e.key) {
                case 'Escape':
                    // Clear selection
                    e.preventDefault();
                    this.clearSelection();
                    return true;
                    
                case 'Delete':
                case 'Backspace':
                    // Delete selected keyframes
                    e.preventDefault();
                    if (this.selectedKeyframes.size > 0 || this.selectedKeyframe !== null) {
                        this.deleteSelectedKeyframes();
                    }
                    return true;
                    
                case ' ':
                    // Space: Toggle playback
                    e.preventDefault();
                    this.togglePlayback();
                    return true;
                    
                case 'ArrowLeft':
                    // Previous frame
                    e.preventDefault();
                    this.currentFrame = Math.max(0, this.currentFrame - 1);
                    this.node.setDirtyCanvas(true);
                    return true;
                    
                case 'ArrowRight':
                    // Next frame
                    e.preventDefault();
                    this.currentFrame = Math.min(this.timeline_frames - 1, this.currentFrame + 1);
                    this.node.setDirtyCanvas(true);
                    return true;
            }
        }
        
        return false;
    }
    
    selectAllKeyframes() {
        // Select all keyframes
        this.selectedKeyframes.clear();
        for (const frame in this.keyframes) {
            this.selectedKeyframes.add(parseInt(frame));
        }
        
        // Set primary selection to first keyframe
        if (this.selectedKeyframes.size > 0) {
            this.selectedKeyframe = Array.from(this.selectedKeyframes)[0];
        }
        
        this.node.setDirtyCanvas(true);
    }
    
    clearSelection() {
        // Clear all selection
        this.selectedKeyframe = null;
        this.selectedKeyframes.clear();
        // Clear scaling handles
        this.handleDrawInfo = null;
        this.leftHandleBounds = null;
        this.rightHandleBounds = null;
        this.node.setDirtyCanvas(true);
    }
    
    // Button handlers
    handleAddKeyframes() {
        console.log("Add Keyframes button clicked");
        // Open file dialog
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.multiple = true;
        input.onchange = (e) => {
            console.log("Files selected:", e.target.files.length);
            const files = Array.from(e.target.files);
            this.loadKeyframeImages(files);
        };
        input.click();
    }
    
    handleReplaceSelected() {
        if (!this.selectedKeyframe) return;
        
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                this.replaceKeyframe(this.selectedKeyframe, file);
            }
        };
        input.click();
    }
    
    handleBatchReplace() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.multiple = true;
        input.onchange = (e) => {
            const files = Array.from(e.target.files);
            
            // Extract and store the folder path from the first file
            if (files.length > 0) {
                // Get the path from the file (browser limitations only give us the name)
                // But we can at least display that files were selected
                this.lastUsedFolder = `${files.length} files selected`;
            }
            
            this.batchReplaceKeyframes(files);
        };
        input.click();
    }
    
    handleClearAll() {
        console.log("Clear All clicked, current keyframes:", Object.keys(this.keyframes).length);
        if (confirm("Clear all keyframes?")) {
            // Clean up blob URLs to prevent memory leaks
            for (const frame in this.keyframes) {
                const kf = this.keyframes[frame];
                if (this.keyframeImages[frame] && this.keyframeImages[frame].startsWith('blob:')) {
                    URL.revokeObjectURL(this.keyframeImages[frame]);
                }
            }
            
            this.keyframes = {};
            this.keyframeImages = {};
            this.imageCache = {};
            this.selectedKeyframe = null;
            this.selectedKeyframes.clear();
            this.updateNodeProperty();
            this.node.setDirtyCanvas(true);
            console.log("Keyframes cleared");
        }
    }
    
    persistImagesToStorage() {
        // Try to save images to localStorage for persistence across browser sessions
        if (typeof(Storage) !== "undefined") {
            try {
                const storageKey = `wan_timeline_${this.node.id}_images`;
                const imageData = {};
                
                // Only store essential images to avoid quota issues
                const maxImages = 20; // Limit to prevent quota exceeded
                let count = 0;
                
                for (const frame in this.keyframeImages) {
                    if (count >= maxImages) break;
                    imageData[frame] = this.keyframeImages[frame];
                    count++;
                }
                
                // STORAGE QUOTA FIX: Size estimation and retry limits to prevent infinite loops
                const dataStr = JSON.stringify(imageData);
                const dataSize = dataStr.length;
                console.log(`[persistImagesToStorage] Attempting to save ${count} images, size: ${(dataSize/1024).toFixed(1)}KB`);
                
                // If data is over 1MB, reduce image count first
                if (dataSize > 1024 * 1024) {
                    console.warn("[persistImagesToStorage] Data too large, reducing to 10 images max");
                    const reducedData = {};
                    let reducedCount = 0;
                    for (const frame in imageData) {
                        if (reducedCount >= 10) break;
                        reducedData[frame] = imageData[frame];
                        reducedCount++;
                    }
                    imageData = reducedData;
                    count = reducedCount;
                }
                
                // Try to store with retry limit
                let retryCount = 0;
                const maxRetries = 2;
                
                while (retryCount < maxRetries) {
                    try {
                        localStorage.setItem(storageKey, JSON.stringify(imageData));
                        console.log(`[persistImagesToStorage] Saved ${count} images to localStorage`);
                        break; // Success - exit retry loop
                    } catch (quotaError) {
                        retryCount++;
                        if (retryCount === 1) {
                            console.warn("[persistImagesToStorage] localStorage quota exceeded, clearing ALL old timeline data");
                            // More aggressive cleanup on first failure
                            for (let i = localStorage.length - 1; i >= 0; i--) {
                                const key = localStorage.key(i);
                                if (key && key.startsWith('wan_timeline_')) {
                                    localStorage.removeItem(key);
                                }
                            }
                        } else {
                            console.warn("[persistImagesToStorage] Still can't save after cleanup - giving up on localStorage");
                            break; // Exit retry loop
                        }
                    }
                }
            } catch (e) {
                console.warn("[persistImagesToStorage] Failed to persist images:", e);
            }
        }
    }
    
    async attemptAutoReload() {
        console.log("[attemptAutoReload] Attempting to reload images from file paths");
        
        // Check if we have any stored file paths or metadata
        let missingImages = [];
        
        for (const frame in this.keyframes) {
            const kf = this.keyframes[frame];
            // Check if we already have an image for this frame
            if (!this.keyframeImages[frame] && !this.imageCache[frame]) {
                missingImages.push({
                    frame, 
                    filename: kf.filename || kf.path || `frame_${frame}`,
                    relativePath: kf.relativePath,
                    metadata: kf
                });
            }
        }
        
        if (missingImages.length === 0) {
            console.log("[attemptAutoReload] All images already loaded");
            this.showReloadPrompt = false;
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        console.log(`[attemptAutoReload] Missing ${missingImages.length} images`);
        
        // Try to load from various caches
        let loadedFromCache = 0;
        
        // 1. Try global storage by node ID
        const globalImages = window._comfyui_wan_keyframes && window._comfyui_wan_keyframes[this.node.id] || {};
        
        for (const {frame} of missingImages) {
            if (globalImages[frame]) {
                this.keyframeImages[frame] = globalImages[frame];
                this.loadImageToCache(parseInt(frame), globalImages[frame]);
                loadedFromCache++;
            }
        }
        
        // 2. Try by keyframe ID
        if (loadedFromCache < missingImages.length) {
            for (const {frame, metadata} of missingImages) {
                if (!this.keyframeImages[frame] && metadata.id && globalImages[metadata.id]) {
                    this.keyframeImages[frame] = globalImages[metadata.id];
                    this.loadImageToCache(parseInt(frame), globalImages[metadata.id]);
                    loadedFromCache++;
                }
            }
        }
        
        // 3. Try localStorage as last resort
        if (loadedFromCache < missingImages.length && typeof(Storage) !== "undefined") {
            const storageKey = `wan_timeline_${this.node.id}_images`;
            try {
                const storedData = localStorage.getItem(storageKey);
                if (storedData) {
                    const parsedData = JSON.parse(storedData);
                    for (const {frame} of missingImages) {
                        if (!this.keyframeImages[frame] && parsedData[frame]) {
                            this.keyframeImages[frame] = parsedData[frame];
                            this.loadImageToCache(parseInt(frame), parsedData[frame]);
                            loadedFromCache++;
                        }
                    }
                }
            } catch (e) {
                console.warn("[attemptAutoReload] localStorage read failed:", e);
            }
        }
        
        if (loadedFromCache > 0) {
            console.log(`[attemptAutoReload] Loaded ${loadedFromCache} images from various caches`);
            missingImages = missingImages.filter(({frame}) => !this.keyframeImages[frame]);
        }
        
        // If we still have missing images, return false to show the file browser
        if (missingImages.length > 0) {
            console.log(`[attemptAutoReload] Still missing ${missingImages.length} images:`);
            missingImages.forEach(({frame, filename, relativePath}) => {
                console.log(`  - Frame ${frame}: ${relativePath || filename}`);
            });
            return false;
        }
        
        // All images loaded successfully
        this.showReloadPrompt = false;
        this.node.setDirtyCanvas(true);
        return true;
    }
    
    handleReloadImages() {
        console.log("[handleReloadImages] Starting manual image reload process");
        
        // First, show dialog explaining the process
        const missingFiles = [];
        let folderHint = "";
        for (const frame in this.keyframes) {
            const kf = this.keyframes[frame];
            if (!this.keyframeImages[frame]) {
                missingFiles.push({
                    filename: kf.filename || `frame_${frame}.png`,
                    folderName: kf.folderName || "",
                    lastModified: kf.lastModified
                });
                if (!folderHint && kf.folderName) {
                    folderHint = kf.folderName;
                }
            }
        }
        
        // Try webkitdirectory first for folder selection (Chrome/Edge support)
        const input = document.createElement('input');
        input.type = 'file';
        
        // Check if webkitdirectory is supported
        if ('webkitdirectory' in input) {
            input.webkitdirectory = true;
            input.directory = true;
            input.multiple = true;
            
            console.log("[handleReloadImages] Using folder selection mode");
            console.log(`[handleReloadImages] Looking for ${missingFiles.length} files from folder: ${folderHint || 'unknown'}`);
            // No alert - go directly to file browser
        } else {
            // Fallback to multi-file selection
            input.accept = 'image/*';
            input.multiple = true;
            
            console.log("[handleReloadImages] Using multi-file selection mode (folder selection not supported)");
            console.log(`[handleReloadImages] Looking for ${missingFiles.length} files from folder: ${folderHint || 'unknown'}`);
            // No alert - go directly to file browser
        }
        
        input.onchange = (e) => {
            const files = Array.from(e.target.files);
            console.log(`[handleReloadImages] Selected ${files.length} files from ${input.webkitdirectory ? 'folder' : 'file picker'}`);
            
            // Try to match files to existing keyframes by filename
            let matchedCount = 0;
            let processedCount = 0;
            const totalKeyframes = Object.keys(this.keyframes).length;
            
            // Create a map for faster lookup
            const fileMap = new Map();
            for (const file of files) {
                fileMap.set(file.name, file);
            }
            
            for (const frame in this.keyframes) {
                const kf = this.keyframes[frame];
                const filename = kf.filename || kf.path;
                
                if (filename) {
                    // Try exact match first
                    let matchingFile = fileMap.get(filename);
                    
                    // If no exact match, try without path
                    if (!matchingFile && filename.includes('/')) {
                        const basename = filename.split('/').pop();
                        matchingFile = fileMap.get(basename);
                    }
                    
                    if (matchingFile) {
                        console.log(`[handleReloadImages] Found matching file for frame ${frame}: ${filename}`);
                        
                        // Read and store the image
                        const reader = new FileReader();
                        reader.onload = (ev) => {
                            this.keyframeImages[frame] = ev.target.result;
                            this.loadImageToCache(parseInt(frame), ev.target.result);
                            
                            // Also store in global storage
                            if (!window._comfyui_wan_keyframes) {
                                window._comfyui_wan_keyframes = {};
                            }
                            if (!window._comfyui_wan_keyframes[this.node.id]) {
                                window._comfyui_wan_keyframes[this.node.id] = {};
                            }
                            window._comfyui_wan_keyframes[this.node.id][frame] = ev.target.result;
                            
                            matchedCount++;
                            processedCount++;
                            console.log(`[handleReloadImages] Loaded image for frame ${frame} (${matchedCount}/${totalKeyframes})`);
                            
                            // Update UI to show progress
                            if (processedCount === totalKeyframes || matchedCount === totalKeyframes) {
                                this.showReloadPrompt = false;
                                this.node.setDirtyCanvas(true);
                                
                                // Persist to localStorage after all images loaded
                                this.persistImagesToStorage();
                                
                                if (matchedCount === totalKeyframes) {
                                    console.log("[handleReloadImages] All images reloaded successfully!");
                                } else {
                                    console.warn(`[handleReloadImages] Loaded ${matchedCount} out of ${totalKeyframes} keyframes`);
                                }
                            }
                        };
                        reader.readAsDataURL(matchingFile);
                    } else {
                        processedCount++;
                        console.warn(`[handleReloadImages] No matching file found for frame ${frame} (${filename})`);
                    }
                } else {
                    processedCount++;
                }
            }
            
            // If we didn't match any files, just log it
            if (matchedCount === 0) {
                console.error(`[handleReloadImages] No matching files found in the selected ${input.webkitdirectory ? 'folder' : 'files'}`);
                console.error(`[handleReloadImages] Expected files:`, missingFiles.slice(0, 10).map(f => f.filename));
            } else if (matchedCount < totalKeyframes) {
                // Partial match - still hide prompt but warn user
                setTimeout(() => {
                    this.showReloadPrompt = false;
                    this.node.setDirtyCanvas(true);
                    console.warn(`[handleReloadImages] Only matched ${matchedCount} out of ${totalKeyframes} keyframes`);
                }, 1000);
            }
        };
        
        input.click();
    }
    
    handleSaveConfig() {
        // Include both keyframe metadata and images
        const keyframesWithImages = {};
        for (const frame in this.keyframes) {
            keyframesWithImages[frame] = {
                ...this.keyframes[frame],
                image: this.keyframeImages[frame] || null
            };
        }
        
        const config = {
            frames: this.timeline_frames,
            fps: this.fps,
            batch_size: this.batchSize,
            ignore_held_frames_mask: !!this.ignore_held_frames_mask,
            nextKeyframeID: this.nextKeyframeID,
            keyframes: keyframesWithImages
        };
        // Include frame_darkness if widget exists
        const fdWidget = this.node.widgets.find(w => w.name === "frame_darkness");
        if (fdWidget && typeof fdWidget.value !== 'undefined') {
            config.frame_darkness = fdWidget.value;
        }
        
        const blob = new Blob([JSON.stringify(config, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'keyframe_timeline.json';
        a.click();
        URL.revokeObjectURL(url);
    }
    
    handleLoadConfig() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const config = JSON.parse(e.target.result);
                        this.loadConfiguration(config);
                    } catch (err) {
                        console.error("Error loading config:", err);
                        alert("Error loading configuration file");
                    }
                };
                reader.readAsText(file);
            }
        };
        input.click();
    }
    
    // Playback controls
    togglePlayback() {
        if (this.isPlaying) {
            this.isPlaying = false;
        } else {
            this.isPlaying = true;
            this.lastPlayTime = performance.now();
            this.animatePlayback();
        }
        this.node.setDirtyCanvas(true);
    }
    
    stopPlayback() {
        this.isPlaying = false;
        // Reset to start of visible range when zoomed, or 0 when not zoomed
        if (this.zoomLevel > 1.0) {
            this.currentFrame = this.scrollOffset;
        } else {
            this.currentFrame = 0;
        }
        this.node.setDirtyCanvas(true);
    }
    
    animatePlayback() {
        if (!this.isPlaying) return;
        
        const now = performance.now();
        const deltaTime = now - this.lastPlayTime;
        const frameDelta = (deltaTime / 1000) * this.fps;
        
        // Calculate visible frame range when zoomed
        const visibleFrames = this.timeline_frames / this.zoomLevel;
        const startFrame = this.scrollOffset;
        const endFrame = Math.min(this.timeline_frames, startFrame + visibleFrames);
        
        this.currentFrame += frameDelta;
        
        // Loop within visible range when zoomed, or full timeline when not zoomed
        if (this.zoomLevel > 1.0) {
            // Zoomed in - loop within visible range
            if (this.currentFrame >= endFrame) {
                this.currentFrame = startFrame;
            } else if (this.currentFrame < startFrame) {
                this.currentFrame = startFrame;
            }
        } else {
            // Not zoomed - loop entire timeline
            if (this.currentFrame >= this.timeline_frames) {
                this.currentFrame = 0;
            }
        }
        
        this.lastPlayTime = now;
        this.node.setDirtyCanvas(true);
        
        requestAnimationFrame(() => this.animatePlayback());
    }
    
    // Keyframe management
    addKeyframeAtPosition(frame) {
        if (this.keyframes[frame]) return;
        
        
        // Add placeholder keyframe
        const keyframeID = `kf_${this.nextKeyframeID++}`;
        this.keyframes[frame] = {
            image: null,
            path: `Frame ${frame}`,
            enabled: true,
            hold: 1,
            id: keyframeID
        };
        
        this.updateNodeProperty();
        this.node.setDirtyCanvas(true);
    }
    
    loadKeyframeImages(files) {
        console.log("loadKeyframeImages called with", files.length, "files");
        
        // Convert FileList to Array and sort naturally by filename
        const filesArray = Array.from(files);
        
        // Natural sort function that handles numbers in filenames correctly
        filesArray.sort((a, b) => {
            const regex = /(\d+)/g;
            
            // Split filenames into parts (text and numbers)
            const aParts = a.name.split(regex);
            const bParts = b.name.split(regex);
            
            for (let i = 0; i < Math.min(aParts.length, bParts.length); i++) {
                const aPart = aParts[i];
                const bPart = bParts[i];
                
                // If both parts are numbers, compare numerically
                if (/^\d+$/.test(aPart) && /^\d+$/.test(bPart)) {
                    const diff = parseInt(aPart) - parseInt(bPart);
                    if (diff !== 0) return diff;
                } else {
                    // Otherwise compare as strings
                    if (aPart < bPart) return -1;
                    if (aPart > bPart) return 1;
                }
            }
            
            // If all parts are equal, compare by length
            return aParts.length - bParts.length;
        });
        
        console.log("Files after sorting:", filesArray.map(f => f.name));
        
        // Calculate positions to evenly distribute frames
        const numFiles = filesArray.length;
        if (numFiles === 0) return;
        
        // Always distribute evenly across the entire timeline, regardless of existing keyframes
        console.log("Distributing", numFiles, "files evenly across timeline");
        
        // Calculate positions for even distribution
        let positions = [];
        
        if (numFiles === 1) {
            // Single frame goes in the middle
            positions = [Math.floor(this.timeline_frames / 2)];
        } else {
            // Multiple frames evenly distributed from start to end
            const step = (this.timeline_frames - 1) / (numFiles - 1);
            for (let i = 0; i < numFiles; i++) {
                positions.push(Math.floor(i * step));
            }
        }
        
        
        // Check for conflicts with existing keyframes and find alternative positions if needed
        const existingFrames = new Set(Object.keys(this.keyframes).map(f => parseInt(f)));
        const finalPositions = [];
        
        for (let pos of positions) {
            // If position is occupied, find nearest available position
            if (existingFrames.has(pos)) {
                let found = false;
                let offset = 1;
                
                while (!found && offset < this.timeline_frames) {
                    // Try positions alternating before and after
                    const before = pos - offset;
                    const after = pos + offset;
                    
                    if (before >= 0 && !existingFrames.has(before) && this.canPlaceKeyframe(before)) {
                        finalPositions.push(before);
                        existingFrames.add(before);
                        found = true;
                    } else if (after < this.timeline_frames && !existingFrames.has(after) && this.canPlaceKeyframe(after)) {
                        finalPositions.push(after);
                        existingFrames.add(after);
                        found = true;
                    }
                    
                    offset++;
                }
                
                if (!found) {
                    console.warn(`Could not find position for keyframe near ${pos}`);
                }
            } else if (this.canPlaceKeyframe(pos)) {
                finalPositions.push(pos);
                existingFrames.add(pos);
            }
        }
        
        // Place keyframes at calculated positions
        let processedCount = 0;
        filesArray.forEach((file, index) => {
            if (index < finalPositions.length) {
                const frame = finalPositions[index];
                
                // Read file as base64 for Python backend
                const reader = new FileReader();
                reader.onload = (e) => {
                    // Store metadata separately from image data
                    const keyframeID = `kf_${this.nextKeyframeID++}`;
                    
                    // Try to get relative path if available (from webkitdirectory drops)
                    let relativePath = file.webkitRelativePath || file.name;
                    
                    // Extract folder name if available
                    let folderName = "";
                    if (file.webkitRelativePath) {
                        const parts = file.webkitRelativePath.split('/');
                        if (parts.length > 1) {
                            folderName = parts[0]; // First part is usually the folder name
                        }
                    }
                    
                    this.keyframes[frame] = {
                        path: file.name,
                        enabled: true,
                        hold: 1,
                        id: keyframeID,
                        // Add file metadata to help with recovery
                        filename: file.name,
                        relativePath: relativePath,
                        folderName: folderName,
                        filesize: file.size,
                        lastModified: file.lastModified,
                        type: file.type
                    };
                    
                    // Store image data separately
                    this.keyframeImages[frame] = e.target.result;
                    
                    // Also store by ID for undo robustness
                    this.keyframeImagesByID[keyframeID] = e.target.result;
                    
                    // Store in global storage by ID as well
                    if (!window._comfyui_wan_keyframes) {
                        window._comfyui_wan_keyframes = {};
                    }
                    if (!window._comfyui_wan_keyframes[this.node.id]) {
                        window._comfyui_wan_keyframes[this.node.id] = {};
                    }
                    window._comfyui_wan_keyframes[this.node.id][frame] = e.target.result;
                    window._comfyui_wan_keyframes[this.node.id][keyframeID] = e.target.result;
                    
                    // Also try to persist to localStorage
                    this.persistImagesToStorage();
                    
                    // Load image into cache
                    this.loadImageToCache(frame, e.target.result);
                    
                    processedCount++;
                    if (processedCount === filesArray.length) {
                        // All files processed, update node
                        console.log(`Distributed ${filesArray.length} keyframes at positions:`, finalPositions);
                        this.updateNodeProperty();
                        this.node.setDirtyCanvas(true);
                    }
                };
                reader.readAsDataURL(file);
            }
        });
    }
    
    replaceKeyframe(frame, file) {
        if (this.keyframes[frame]) {
            // Read file as base64 for Python backend
            const reader = new FileReader();
            reader.onload = (e) => {
                // Update metadata
                this.keyframes[frame] = {
                    path: file.name,
                    enabled: this.keyframes[frame].enabled,
                    hold: this.keyframes[frame].hold
                };
                
                // Store image data separately
                this.keyframeImages[frame] = e.target.result;
                
                // Load image into cache
                this.loadImageToCache(frame, e.target.result);
                
                this.updateNodeProperty();
                this.node.setDirtyCanvas(true);
            };
            reader.readAsDataURL(file);
        }
    }
    
    batchReplaceKeyframes(files) {
        const sortedFrames = Object.keys(this.keyframes).map(f => parseInt(f)).sort((a, b) => a - b);
        
        let processed = 0;
        const total = Math.min(files.length, sortedFrames.length);
        files.forEach((file, index) => {
            if (index < sortedFrames.length) {
                const frame = sortedFrames[index];
                const reader = new FileReader();
                reader.onload = (e) => {
                    // Update metadata but preserve id/enabled/hold
                    const current = this.keyframes[frame] || {};
                    this.keyframes[frame] = {
                        path: file.name,
                        enabled: current.enabled !== undefined ? current.enabled : true,
                        hold: current.hold || 1,
                        id: current.id || `kf_${this.nextKeyframeID++}`
                    };
                    this.keyframeImages[frame] = e.target.result;
                    this.loadImageToCache(frame, e.target.result);
                    processed++;
                    if (processed === total) {
                        // Completed batch replace: clear overlays and interaction flags to restore scrubbing
                        this.showReloadPrompt = false;
                        this.isScrubbing = false;
                        this.isDragging = false;
                        this.isBoxSelecting = false;
                        this.isDraggingBlock = false;
                        this.isDraggingZoom = false;
                        this.isDraggingScroll = false;
                        this.isDraggingFile = false;
                        this.reloadButtonBounds = null;
                        this.updateNodeProperty();
                        this.node.setDirtyCanvas(true);
                    }
                };
                reader.readAsDataURL(file);
            }
        });
    }
    
    loadConfiguration(config) {
        if (config.frames) {
            // LEGACY FLOAT SUPPORT: Convert float frames to integer
            const framesValue = Math.round(config.frames);
            if (framesValue !== config.frames) {
                console.log(`[loadConfiguration] Converting legacy float frames ${config.frames} to integer ${framesValue}`);
            }
            this.timeline_frames = framesValue;
            // Update node widget
            const framesWidget = this.node.widgets.find(w => w.name === "timeline_frames");
            if (framesWidget) {
                framesWidget.value = framesValue;
                if (typeof framesWidget.callback === 'function') framesWidget.callback(framesValue);
            }
        }
        
        if (config.fps) {
            // LEGACY FLOAT SUPPORT: Convert float fps to integer
            const fpsValue = Math.round(config.fps);
            if (fpsValue !== config.fps) {
                console.log(`[loadConfiguration] Converting legacy float fps ${config.fps} to integer ${fpsValue}`);
            }
            this.fps = fpsValue;
            const fpsWidget = this.node.widgets.find(w => w.name === "fps");
            if (fpsWidget) {
                fpsWidget.value = fpsValue;
                if (typeof fpsWidget.callback === 'function') fpsWidget.callback(fpsValue);
            }
        }

        if (typeof config.batch_size !== 'undefined') {
            this.batchSize = Math.round(config.batch_size);
            const batchWidget = this.node.widgets.find(w => w.name === "batch_size");
            if (batchWidget) {
                batchWidget.value = this.batchSize;
                if (typeof batchWidget.callback === 'function') batchWidget.callback(this.batchSize);
            }
        }

        if (typeof config.ignore_held_frames_mask !== 'undefined') {
            this.ignore_held_frames_mask = !!config.ignore_held_frames_mask;
        }

        if (typeof config.nextKeyframeID !== 'undefined') {
            this.nextKeyframeID = config.nextKeyframeID;
        }

        if (typeof config.frame_darkness !== 'undefined') {
            const fdWidget = this.node.widgets.find(w => w.name === "frame_darkness");
            if (fdWidget) {
                fdWidget.value = config.frame_darkness;
                if (typeof fdWidget.callback === 'function') fdWidget.callback(config.frame_darkness);
            }
        }
        
        if (config.keyframes) {
            // Clear existing data
            this.keyframes = {};
            this.keyframeImages = {};
            this.imageCache = {};
            
            // Initialize global storage if needed
            if (!window._comfyui_wan_keyframes) {
                window._comfyui_wan_keyframes = {};
            }
            if (!window._comfyui_wan_keyframes[this.node.id]) {
                window._comfyui_wan_keyframes[this.node.id] = {};
            }
            
            // Load each keyframe with its image
            for (const frame in config.keyframes) {
                const kfData = config.keyframes[frame];
                
                // LEGACY FLOAT SUPPORT: Convert float frame keys to integer
                const frameNum = Math.round(parseFloat(frame));
                if (frameNum !== parseFloat(frame)) {
                    console.log(`[loadConfiguration] Converting legacy float keyframe position ${frame} to integer ${frameNum}`);
                }
                
                // Extract metadata
                this.keyframes[frameNum] = {
                    path: kfData.path,
                    enabled: kfData.enabled !== undefined ? kfData.enabled : true,
                    hold: kfData.hold || 1,
                    id: kfData.id || this.keyframes[frameNum]?.id || `kf_${this.nextKeyframeID++}`
                };
                
                // Extract and store image
                if (kfData.image) {
                    this.keyframeImages[frameNum] = kfData.image;
                    window._comfyui_wan_keyframes[this.node.id][frameNum] = kfData.image;
                    
                    // Load into cache
                    this.loadImageToCache(frameNum, kfData.image);
                }
            }
        }
        
        this.updateNodeProperty();
        this.node.setDirtyCanvas(true);
    }
    
    findKeyframeAtPosition(position) {
        // Find keyframe at exact position or holding at position
        if (this.keyframes[position]) {
            return { frame: position, isExact: true };
        }
        
        // Check if any keyframe is holding at this position
        for (const frame in this.keyframes) {
            const frameNum = parseInt(frame);
            const kf = this.keyframes[frame];
            if (!kf) continue; // Defensive check
            if (kf.hold > 0 && position > frameNum && position <= frameNum + kf.hold) {
                return { frame: frameNum, isExact: false };
            }
        }
        
        return null;
    }
    
    
    loadImageToCache(frame, imageUrl) {
        // Ensure frame is a number for consistency
        const frameNum = typeof frame === 'string' ? parseInt(frame) : frame;
        
        const img = new Image();
        img.onload = () => {
            this.imageCache[frameNum] = img;
            this.node.setDirtyCanvas(true);
        };
        img.onerror = () => {
            console.error(`Failed to load image for frame ${frameNum}`);
            delete this.imageCache[frameNum];
        };
        img.src = imageUrl;
    }
    
    deleteSelectedKeyframe() {
        if (this.selectedKeyframe === null) return;
        
        const frame = this.selectedKeyframe;
        const kf = this.keyframes[frame];
        
        // Clean up blob URL
        if (kf && this.keyframeImages[frame] && this.keyframeImages[frame].startsWith('blob:')) {
            URL.revokeObjectURL(this.keyframeImages[frame]);
        }
        
        // Remove from data structures
        delete this.keyframes[frame];
        delete this.keyframeImages[frame];
        delete this.imageCache[frame];
        
        // Clear selection
        this.selectedKeyframe = null;
        this.selectedKeyframes.clear();
        
        this.updateNodeProperty();
        this.node.setDirtyCanvas(true);
    }
    
    // Enable drag and drop
    onDragOver(e) {
        // Accept file drops
        if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
            const firstItem = e.dataTransfer.items[0];
            if (firstItem.kind === 'file' && firstItem.type.startsWith('image/')) {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
                this.isDraggingFile = true;
                this.node.setDirtyCanvas(true);
                return true;
            }
        }
        
        // Also accept other types of drags (for ComfyUI node drags)
        if (e.dataTransfer.types.includes('text/plain') || 
            e.dataTransfer.types.includes('text/uri-list') ||
            e.dataTransfer.types.includes('application/litegraph')) {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
            this.isDraggingFile = true;
            this.node.setDirtyCanvas(true);
            return true;
        }
        
        return false;
    }
    
    onDragLeave(e) {
        this.isDraggingFile = false;
        this.node.setDirtyCanvas(true);
    }
    
    onDrop(e) {
        e.preventDefault();
        e.stopPropagation(); // Stop ComfyUI from handling this drop
        e.stopImmediatePropagation(); // Really stop it
        this.isDraggingFile = false;
        
        // Calculate drop position relative to the widget
        const canvas = app.canvas.canvas;
        const rect = canvas.getBoundingClientRect();
        const transform = app.canvas.ds;
        
        // Convert screen coordinates to node-relative coordinates
        const canvasX = (e.clientX - rect.left) / transform.scale - transform.offset[0];
        const canvasY = (e.clientY - rect.top) / transform.scale - transform.offset[1];
        
        const x = canvasX - this.node.pos[0];
        const y = canvasY - this.node.pos[1] - this.last_y;
        
        // Check if dropped on timeline
        const trackX = this.margin + 10;
        const trackY = this.layout.timeline;
        const trackWidth = this.node.size[0] - this.margin * 2 - 20;
        const trackHeight = this.layout.timelineHeight;
        
        let dropFrame = null;
        if (x >= trackX && x <= trackX + trackWidth && y >= trackY - 20 && y <= trackY + trackHeight + 20) {
            // Calculate frame position from drop location using zoom-aware calculation
            // This uses the same pattern as screenXToFrame used everywhere else in the timeline
            const visibleFrames = this.timeline_frames / this.zoomLevel;
            const normalizedX = (x - trackX) / trackWidth;
            dropFrame = Math.round(this.scrollOffset + normalizedX * visibleFrames);
            dropFrame = Math.max(0, Math.min(this.timeline_frames - 1, dropFrame));
            
        }
        
        // Try to handle ComfyUI node drag (when dragging from Load Image or other image nodes)
        const dt = e.dataTransfer;
        
        // Check various ways ComfyUI might track dragged content
        let draggedImage = null;
        let sourceName = "Unknown";
        
        // Method 1: Check canvas drag state
        if (app.canvas) {
            console.log("Canvas state:", {
                dragging_canvas: app.canvas.dragging_canvas,
                node_dragged: app.canvas.node_dragged,
                last_mouse_dragging: app.canvas.last_mouse_dragging,
                selected_nodes: app.canvas.selected_nodes
            });
            
            // Check if there's a selected node with images
            if (app.canvas.selected_nodes && Object.keys(app.canvas.selected_nodes).length > 0) {
                for (const nodeId in app.canvas.selected_nodes) {
                    const node = app.canvas.selected_nodes[nodeId];
                    if (node.imgs && node.imgs.length > 0) {
                        draggedImage = node.imgs[0];
                        sourceName = node.type || node.constructor.name;
                        console.log("Found image in selected node:", node);
                        break;
                    }
                }
            }
        }
        
        // Method 2: Check if this is a ComfyUI image drop by looking at the URL
        const textData = dt.getData('text/plain') || dt.getData('text/uri-list') || "";
        if (textData && (textData.includes('/view?') || textData.includes('/api/view'))) {
            console.log("Detected ComfyUI image URL:", textData);
            draggedImage = { src: textData };
            sourceName = "ComfyUI";
        }
        
        // If we found a dragged image, add it to the timeline
        if (draggedImage) {
            const frame = dropFrame !== null ? dropFrame : this.findNextAvailableFrame();
            
            if (frame !== null && frame < this.timeline_frames) {
                const imageUrl = draggedImage.src || (draggedImage instanceof HTMLImageElement ? draggedImage.src : null);
                
                if (imageUrl) {
                    this.keyframes[frame] = {
                        image: imageUrl,
                        path: `From ${sourceName}`,
                        enabled: true,
                        hold: 1,
                        thumbnail: true
                    };
                    
                    // Load image into cache
                    this.loadImageToCache(frame, imageUrl);
                    
                    // Select the new keyframe
                    this.selectedKeyframe = frame;
                    this.selectedKeyframes.clear();
                    this.selectedKeyframes.add(frame);
                    
                    this.updateNodeProperty();
                    this.node.setDirtyCanvas(true);
                    
                    console.log(`Added keyframe at frame ${frame} from ${sourceName}`);
                    return true;
                }
            }
        }
        
        // Also check dataTransfer for other types of drags
        if (dt.types.length > 0) {
            console.log("DataTransfer types:", dt.types);
            
            // Try to get any text data that might contain image info
            for (const type of dt.types) {
                const data = dt.getData(type);
                if (data) {
                    console.log(`Data for type ${type}:`, data);
                }
            }
        }
        
        // Handle regular file drops
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length > 0) {
            if (files.length === 1 && dropFrame !== null) {
                // Single file dropped on specific position - place it exactly there
                const file = files[0];
                
                // Read file as base64 for Python backend
                const reader = new FileReader();
                reader.onload = (ev) => {
                    // Store metadata
                    this.keyframes[dropFrame] = {
                        path: file.name,
                        enabled: true,
                        hold: 1
                    };
                    
                    // Store image data separately
                    this.keyframeImages[dropFrame] = ev.target.result;
                    
                    // Load image into cache
                    this.loadImageToCache(dropFrame, ev.target.result);
                    
                    // Select the new keyframe
                    this.selectedKeyframe = dropFrame;
                    this.selectedKeyframes.clear();
                    this.selectedKeyframes.add(dropFrame);
                    
                    this.updateNodeProperty();
                    this.node.setDirtyCanvas(true);
                    
                    console.log(`Placed single keyframe at frame ${dropFrame}`);
                };
                reader.readAsDataURL(file);
            } else {
                // Multiple files - distribute them evenly across the timeline
                console.log(`Dropping ${files.length} files - distributing evenly`);
                this.loadKeyframeImages(files);
            }
        }
        
        this.node.setDirtyCanvas(true);
        return true;
    }
    
    findNextAvailableFrame() {
        let frame = 0;
        while (this.keyframes[frame] && frame < this.timeline_frames) {
            frame++;
        }
        
        
        return frame < this.timeline_frames ? frame : null;
    }
    
    handleKeyDown(event) {
        // Existing keydown switch code from mouse method
        switch (event.key) {
            case "Delete": // Delete selected keyframes
                this.deleteSelectedKeyframes();
                event.preventDefault();
                event.stopPropagation();
                return true;
            case "ArrowLeft":
                // Move selected keyframe left
                if (this.selectedKeyframe !== null) {
                    const newFrame = Math.max(1, this.selectedKeyframe - 1);
                    if (newFrame !== this.selectedKeyframe) {
                        this.moveKeyframe(this.selectedKeyframe, newFrame);
                    }
                }
                return true;
            case "ArrowRight":
                // Move selected keyframe right
                if (this.selectedKeyframe !== null) {
                    const newFrame = Math.min(this.timeline_frames, this.selectedKeyframe + 1);
                    if (newFrame !== this.selectedKeyframe) {
                        this.moveKeyframe(this.selectedKeyframe, newFrame);
                    }
                }
                return true;
            default:
                return false;
        }
    }
}

// Register the extension
app.registerExtension({
    name: "WAN.KeyframeTimeline",
    
    async setup() {
        // Add global recovery handler for stuck timeline widgets
        let lastRecoveryTime = 0;
        const addGlobalRecoveryHandler = () => {
            const canvas = app.canvas.canvas;
            
            // Add a recovery handler that can force widget focus when needed
            canvas.addEventListener('click', (e) => {
                // Check if we should attempt recovery (rate limit to once per second)
                const now = Date.now();
                if (now - lastRecoveryTime < 1000) return;
                
                // Find any timeline widgets that might be stuck
                for (const node of app.graph._nodes) {
                    if (node.type === "WANVaceKeyframeTimeline" && node.timelineWidget) {
                        const widget = node.timelineWidget;
                        
                        // Check if widget seems stuck (has keyframes but no recent mouse down)
                        if (Object.keys(widget.keyframes).length > 0 && 
                            (!widget.lastMouseDownTime || now - widget.lastMouseDownTime > 5000)) {
                            
                            // Check if click is on this node
                            const rect = canvas.getBoundingClientRect();
                            const transform = app.canvas.ds;
                            const x = (e.clientX - rect.left) / transform.scale - transform.offset[0];
                            const y = (e.clientY - rect.top) / transform.scale - transform.offset[1];
                            
                            if (x >= node.pos[0] && x <= node.pos[0] + node.size[0] &&
                                y >= node.pos[1] && y <= node.pos[1] + node.size[1]) {
                                console.log("[Recovery] Attempting to recover stuck timeline widget");
                                // Force clear all states
                                if (widget.clearAllStates) {
                                    widget.clearAllStates();
                                }
                                // Force widget focus
                                if (app.canvas) {
                                    app.canvas.node_widget = [node, widget];
                                }
                                lastRecoveryTime = now;
                                node.setDirtyCanvas(true);
                            }
                        }
                    }
                }
            }, true); // Use capture phase
        };
        
        // Set up recovery handler after a delay to ensure canvas is ready
        setTimeout(addGlobalRecoveryHandler, 1000);
        
        // Hook into the graph to prompt conversion after app is ready
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function() {
            // Store a flag to indicate we're in prompt generation
            window._wan_in_prompt_generation = true;
            
            // Yield control to browser to allow UI updates (green outline) and ensure global storage is updated
            await new Promise(resolve => setTimeout(resolve, 50));
            
            // Before converting graph to prompt, update all timeline widgets
            for (const node of app.graph._nodes) {
                if (node.type === "WANVaceKeyframeTimeline" && node.timelineWidget) {
                    // ALWAYS reconstruct image data from global storage - don't skip based on metadata
                    // The metadata hash check was causing the second run to skip image reconstruction
                    // Even if metadata is unchanged, we need to ensure images are always included
                    
                    const keyframeDataWidget = node.widgets.find(w => w.name === "keyframe_data");
                    if (keyframeDataWidget) {
                        // Yield to browser every few nodes to keep UI responsive
                        await new Promise(resolve => requestAnimationFrame(resolve));
                        
                        // Get images from both global storage AND the widget's own storage
                        const globalImages = window._comfyui_wan_keyframes && window._comfyui_wan_keyframes[node.id] || {};
                        const widgetImages = node.timelineWidget.keyframeImages || {};
                        
                        // Merge metadata with images
                        const fullKeyframes = {};
                        let imageCount = 0;
                        for (const frame in node.timelineWidget.keyframes) {
                            const kf = node.timelineWidget.keyframes[frame];
                            let image = null;
                            
                            // Try to find image in multiple places
                            if (widgetImages[frame]) {
                                image = widgetImages[frame];
                            } else if (globalImages[frame]) {
                                image = globalImages[frame];
                            } else if (kf.id && globalImages[kf.id]) {
                                image = globalImages[kf.id];
                            } else if (kf.id && node.timelineWidget.keyframeImagesByID && node.timelineWidget.keyframeImagesByID[kf.id]) {
                                image = node.timelineWidget.keyframeImagesByID[kf.id];
                            }
                            
                            fullKeyframes[frame] = {
                                ...kf,
                                image: image
                            };
                            if (image) {
                                imageCount++;
                            }
                        }
                        
                        const keyframeData = {
                            frames: node.timelineWidget.timeline_frames,
                            keyframes: fullKeyframes,
                            ignore_held_frames_mask: node.timelineWidget.ignore_held_frames_mask
                        };
                        
                        // IMPORTANT: Only set the value temporarily for prompt execution
                        // Store original value to restore after
                        const originalValue = keyframeDataWidget.value;
                        
                        const jsonData = JSON.stringify(keyframeData);
                        keyframeDataWidget.value = jsonData;
                        
                        // Also set internal properties to ensure the value is sent
                        if (keyframeDataWidget._value !== undefined) {
                            keyframeDataWidget._value = jsonData;
                        }
                        if (keyframeDataWidget._internalValue !== undefined) {
                            keyframeDataWidget._internalValue = jsonData;
                        }
                        
                        console.log(`graphToPrompt: Updated timeline node ${node.id} with ${Object.keys(fullKeyframes).length} keyframes (${imageCount} with images)`);
                        
                        // Schedule restoration of original value after prompt is sent
                        setTimeout(() => {
                            keyframeDataWidget.value = originalValue;
                            if (keyframeDataWidget._value !== undefined) {
                                keyframeDataWidget._value = originalValue;
                            }
                            if (keyframeDataWidget._internalValue !== undefined) {
                                keyframeDataWidget._internalValue = originalValue;
                            }
                            console.log(`Restored widget to metadata-only value`);
                        }, 100);
                    }
                }
            }
            
            // Call original function
            const result = await originalGraphToPrompt.apply(this, arguments);
            
            // Clear the flag after prompt generation
            window._wan_in_prompt_generation = false;
            
            return result;
        };
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WANVaceKeyframeTimeline") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Create the timeline widget
                const timelineWidget = new WANKeyframeTimelineWidget(this);
                this.addCustomWidget(timelineWidget);
                
                // Store reference
                this.timelineWidget = timelineWidget;
                
                // Force initial size (user requested width)
                this.size = [740, 600];
                this.setDirtyCanvas(true);
                
                // Override getExtraMenuOptions to update data before queueing
                const originalGetExtraMenuOptions = this.getExtraMenuOptions;
                this.getExtraMenuOptions = function(canvas, options) {
                    // Skip expensive image processing for context menu display
                    // Only process full data when actually executing workflow
                    if (!window._wan_in_prompt_generation && this.timelineWidget) {
                        const hiddenWidget = this.widgets.find(w => w.name === "keyframe_data");
                        if (hiddenWidget && !hiddenWidget.value) {
                            // Only set minimal metadata for context menu
                            const keyframeData = {
                                frames: this.timelineWidget.timeline_frames,
                                keyframes: this.timelineWidget.keyframes
                            };
                            hiddenWidget.value = JSON.stringify(keyframeData);
                        }
                    }
                    
                    if (originalGetExtraMenuOptions) {
                        return originalGetExtraMenuOptions.call(this, canvas, options);
                    }
                    return options;
                };
                
                // Update timeline when parameters change
                const framesWidget = this.widgets.find(w => w.name === "timeline_frames");
                if (framesWidget) {
                    const originalCallback = framesWidget.callback;
                    framesWidget.callback = function(value) {
                        if (originalCallback) originalCallback.call(this, value);
                        timelineWidget.timeline_frames = Math.round(value);
                        timelineWidget.updateNodeProperty();
                    };
                }
                
                const fpsWidget = this.widgets.find(w => w.name === "fps");
                if (fpsWidget) {
                    const originalCallback = fpsWidget.callback;
                    fpsWidget.callback = function(value) {
                        if (originalCallback) originalCallback.call(this, value);
                        timelineWidget.fps = Math.round(value);
                    };
                }
                
                const batchWidget = this.widgets.find(w => w.name === "batch_size");
                if (batchWidget) {
                    const originalCallback = batchWidget.callback;
                    batchWidget.callback = function(value) {
                        if (originalCallback) originalCallback.call(this, value);
                        timelineWidget.batchSize = Math.round(value);
                    };
                }
                
                // Sync initial widget values into the timeline widget so visuals match saved settings
                (function syncInitialWidgetValues(node, timelineWidget) {
                    try {
                        const framesW = node.widgets.find(w => w.name === "timeline_frames");
                        if (framesW && typeof framesW.value !== 'undefined') {
                            timelineWidget.timeline_frames = Math.round(framesW.value);
                        }
                        const fpsW = node.widgets.find(w => w.name === "fps");
                        if (fpsW && typeof fpsW.value !== 'undefined') {
                            timelineWidget.fps = Math.round(fpsW.value);
                        }
                        const batchW = node.widgets.find(w => w.name === "batch_size");
                        if (batchW && typeof batchW.value !== 'undefined') {
                            timelineWidget.batchSize = Math.round(batchW.value);
                        }
                        // Apply and redraw
                        timelineWidget.updateNodeProperty();
                        node.setDirtyCanvas(true);
                    } catch (e) {
                        console.warn("[Timeline] Failed to sync initial widget values:", e);
                    }
                })(this, timelineWidget);

                
                // Setup drag and drop handling
                // Add event listeners to the canvas for this node
                const setupDragDrop = () => {
                    const canvas = app.canvas.canvas;
                    
                    // Create handlers that check if over this node
                    const handleDragOver = (e) => {
                        const rect = canvas.getBoundingClientRect();
                        const transform = app.canvas.ds;
                        
                        // Convert to canvas coordinates
                        const x = (e.clientX - rect.left) / transform.scale - transform.offset[0];
                        const y = (e.clientY - rect.top) / transform.scale - transform.offset[1];
                        
                        // Check if over this node's timeline widget
                        if (x >= this.pos[0] && x <= this.pos[0] + this.size[0] &&
                            y >= this.pos[1] + timelineWidget.last_y && 
                            y <= this.pos[1] + timelineWidget.last_y + timelineWidget.height) {
                            e.preventDefault();
                            e.stopPropagation();
                            timelineWidget.isDraggingFile = true;
                            this.setDirtyCanvas(true);
                        }
                    };
                    
                    const handleDrop = (e) => {
                        const rect = canvas.getBoundingClientRect();
                        const transform = app.canvas.ds;
                        
                        // Convert to canvas coordinates
                        const x = (e.clientX - rect.left) / transform.scale - transform.offset[0];
                        const y = (e.clientY - rect.top) / transform.scale - transform.offset[1];
                        
                        // Check if over this node's timeline widget
                        if (x >= this.pos[0] && x <= this.pos[0] + this.size[0] &&
                            y >= this.pos[1] + timelineWidget.last_y && 
                            y <= this.pos[1] + timelineWidget.last_y + timelineWidget.height) {
                            e.preventDefault();
                            e.stopPropagation();
                            
                            console.log("Drop detected on timeline!");
                            
                            // Handle the drop
                            timelineWidget.onDrop(e);
                            return false;
                        }
                    };
                    
                    // Add listeners with capture to intercept before ComfyUI
                    canvas.addEventListener('dragover', handleDragOver, true);
                    canvas.addEventListener('drop', handleDrop, true);
                    
                    // Store handlers for cleanup
                    this._dragHandlers = { handleDragOver, handleDrop };
                };
                
                // Setup immediately
                setupDragDrop();
                
                // Handle keyboard events
                this.onKeyDown = (e) => {
                    if (this.timelineWidget && this.timelineWidget.handleKeyDown) {
                        return this.timelineWidget.handleKeyDown(e);
                    }
                    return false;
                };
                
                // Clean up on node removal
                const onRemoved = this.onRemoved;
                this.onRemoved = function() {
                    if (this._dragHandlers) {
                        const canvas = app.canvas.canvas;
                        canvas.removeEventListener('dragover', this._dragHandlers.handleDragOver, true);
                        canvas.removeEventListener('drop', this._dragHandlers.handleDrop, true);
                    }
                    if (onRemoved) onRemoved.call(this);
                };
                
                // Setup the keyframe_data widget after a longer delay to ensure ComfyUI has created it
                const setupWidget = () => {
                    const keyframeDataWidget = this.widgets.find(w => w.name === "keyframe_data");
                    if (keyframeDataWidget) {
                        console.log("Found keyframe_data widget, configuring...");
                        // Hide the widget visually but keep it functional
                        keyframeDataWidget.computeSize = () => [0, -4];
                        // Don't change the type to "hidden" as it might prevent data transmission
                        
                        // Initialize with empty metadata only
                        keyframeDataWidget.value = JSON.stringify({
                            frames: this.timelineWidget.timeline_frames,
                            keyframes: {}  // No images, just metadata
                        });
                        
                        console.log("Widget configured and initialized");
                        
                        // Also ensure the widget is accessible to the timeline widget
                        this.timelineWidget.keyframeDataWidget = keyframeDataWidget;
                    } else {
                        console.warn("keyframe_data widget not found! Widgets available:", this.widgets.map(w => w.name));
                        // Try again after a longer delay
                        setTimeout(setupWidget, 500);
                    }
                };
                
                setTimeout(setupWidget, 200);  // Initial delay
                
                // In onNodeCreated, after this.timelineWidget = timelineWidget;
                return r;
            };
            
            
            // Handle serialization
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (onSerialize) onSerialize.call(this, o);
                if (this.timelineWidget) {
                    // Save metadata in the workflow
                    o.keyframeData = {
                        frames: this.timelineWidget.timeline_frames,
                        keyframes: this.timelineWidget.keyframes,  // Metadata only with filenames
                        nextKeyframeID: this.timelineWidget.nextKeyframeID,  // Preserve ID counter
                        ignore_held_frames_mask: this.timelineWidget.ignore_held_frames_mask
                    };
                    
                    // Note: Images are not saved - they will be reloaded from files when needed
                }
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                if (onConfigure) onConfigure.call(this, o);
                if (this.timelineWidget && o.keyframeData) {
                    // Load timeline frames
                    this.timelineWidget.timeline_frames = Math.round(o.keyframeData.frames || 81);
                    
                    // Load keyframes metadata only (images must be re-dropped after load)
                    const keyframes = o.keyframeData.keyframes || {};
                    this.timelineWidget.keyframes = keyframes;
                    
                    // Restore ID counter
                    this.timelineWidget.nextKeyframeID = o.keyframeData.nextKeyframeID || 1;
                    
                    // Restore mask settings
                    this.timelineWidget.ignore_held_frames_mask = o.keyframeData.ignore_held_frames_mask || false;
                    
                    // Initialize global storage if needed
                    if (!window._comfyui_wan_keyframes) {
                        window._comfyui_wan_keyframes = {};
                    }
                    if (!window._comfyui_wan_keyframes[this.id]) {
                        window._comfyui_wan_keyframes[this.id] = {};
                    }
                    
                    // IMPORTANT: Don't clear image storage on configure - preserve existing data
                    // Only initialize if they don't exist
                    if (!this.timelineWidget.keyframeImages) {
                        this.timelineWidget.keyframeImages = {};
                    }
                    if (!this.timelineWidget.keyframeImagesByID) {
                        this.timelineWidget.keyframeImagesByID = {};
                    }
                    
                    // Check if we have keyframes that need images loaded
                    const needsImageReload = Object.keys(keyframes).length > 0 && 
                                           Object.keys(this.timelineWidget.keyframeImages).length === 0;
                    
                    if (needsImageReload) {
                        console.log(`[onConfigure] Timeline has ${Object.keys(keyframes).length} keyframes but no images loaded`);
                        
                        // First attempt to auto-reload from cache
                        this.timelineWidget.attemptAutoReload().then(success => {
                            if (!success) {
                                // Only show reload prompt if auto-reload failed
                                console.log("[onConfigure] Auto-reload failed, showing reload prompt");
                                this.timelineWidget.showReloadPrompt = true;
                                this.setDirtyCanvas(true);
                            } else {
                                console.log("[onConfigure] Auto-reload successful!");
                            }
                        });
                    }
                    
                    // Get ALL images from global storage (not just the ones we know about)
                    const globalImages = window._comfyui_wan_keyframes[this.id] || {};
                    
                    console.log(`[onConfigure] Restoring images for node ${this.id}`);
                    console.log(`  - Global storage has ${Object.keys(globalImages).length} entries`);
                    console.log(`  - Keyframes to restore: ${Object.keys(keyframes).length}`);
                    
                    // First pass: collect all available images by any key
                    const allAvailableImages = {};
                    let imageCount = 0;
                    for (const key in globalImages) {
                        if (globalImages[key] && typeof globalImages[key] === 'string' && globalImages[key].startsWith('data:')) {
                            allAvailableImages[key] = globalImages[key];
                            imageCount++;
                        }
                    }
                    console.log(`  - Found ${imageCount} valid images in global storage`);
                    
                    // Second pass: remap images to match restored keyframes (fixes preview after undo)
                    let restoredCount = 0;
                    const oldImages = this.timelineWidget.keyframeImages || {};
                    const oldByID = this.timelineWidget.keyframeImagesByID || {};
                    const newImages = {};
                    const newByID = {};
                    
                    // Clear preview image cache to avoid stale bitmaps
                    this.timelineWidget.imageCache = {};
                    
                    for (const frame in keyframes) {
                        const kf = keyframes[frame];
                        // Ensure keyframe has an ID
                        if (!kf.id) {
                            kf.id = `kf_${this.timelineWidget.nextKeyframeID++}`;
                        }
                        let img = null;
                        // Prefer image mapped by ID (most stable)
                        if (kf.id && oldByID[kf.id]) {
                            img = oldByID[kf.id];
                        } else if (kf.id && allAvailableImages[kf.id]) {
                            img = allAvailableImages[kf.id];
                        } else if (oldImages[frame]) {
                            img = oldImages[frame];
                        } else if (allAvailableImages[frame]) {
                            img = allAvailableImages[frame];
                        }
                        if (img) {
                            newImages[frame] = img;
                            newByID[kf.id] = img;
                            this.timelineWidget.loadImageToCache(parseInt(frame), img);
                            restoredCount++;
                        } else {
                            console.warn(`  - Frame ${frame}: No image found to restore`);
                        }
                    }
                    // Replace UI caches with remapped versions
                    this.timelineWidget.keyframeImages = newImages;
                    this.timelineWidget.keyframeImagesByID = newByID;
                    
                    console.log(`[onConfigure] Remapped ${restoredCount}/${Object.keys(keyframes).length} keyframe images for preview cache`);
                    
                    // Update widgets
                    const framesWidget = this.widgets.find(w => w.name === "timeline_frames");
                    if (framesWidget) {
                        framesWidget.value = this.timelineWidget.timeline_frames;
                        if (typeof framesWidget.callback === 'function') framesWidget.callback(framesWidget.value);
                    }
                    // Ensure timeline state reflects current widget values after configuration restore
                    const fpsWidget = this.widgets.find(w => w.name === "fps");
                    if (fpsWidget && typeof fpsWidget.value !== 'undefined') {
                        this.timelineWidget.fps = Math.round(fpsWidget.value);
                    }
                    const batchWidget = this.widgets.find(w => w.name === "batch_size");
                    if (batchWidget && typeof batchWidget.value !== 'undefined') {
                        this.timelineWidget.batchSize = Math.round(batchWidget.value);
                    }
                    const fdWidget = this.widgets.find(w => w.name === "frame_darkness");
                    if (fdWidget && typeof fdWidget.value !== 'undefined') {
                        // no internal field stored; sent via keyframe_data when updated
                    }
                    
                    this.timelineWidget.updateNodeProperty();
                    this.setDirtyCanvas(true);
                }
            };
        }
    }
});// Cache bust: 1751763700
// Force reload: 1751763700000000000
