import { app } from "../../../../scripts/app.js";

// Enhanced Save Video Extension - Parameter graying and UI improvements
app.registerExtension({
    name: "WANVace.SaveVideo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WANSaveVideo") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Find the output_mode widget
                const outputModeWidget = this.widgets.find(w => w.name === "output_mode");
                
                if (outputModeWidget) {
                    // Store references to widgets for easy access
                    this.videoWidgets = [
                        this.widgets.find(w => w.name === "fps"),
                        this.widgets.find(w => w.name === "video_format"),
                        this.widgets.find(w => w.name === "video_codec"), 
                        this.widgets.find(w => w.name === "video_quality")
                    ].filter(w => w); // Remove any undefined widgets
                    
                    this.imageWidgets = [
                        this.widgets.find(w => w.name === "image_format"),
                        this.widgets.find(w => w.name === "filename_prefix"),
                        this.widgets.find(w => w.name === "start_number"),
                        this.widgets.find(w => w.name === "number_padding")
                    ].filter(w => w); // Remove any undefined widgets
                    
                    // Incremental save widget is common to both modes
                    this.incrementalSaveWidget = this.widgets.find(w => w.name === "incremental_save");
                    
                    // Function to update widget states
                    this.updateWidgetStates = function(mode) {
                        const isVideoMode = mode === "video";
                        
                        // Update video widgets
                        this.videoWidgets.forEach(widget => {
                            if (widget) {
                                widget.disabled = !isVideoMode;
                                // Apply visual styling for disabled state
                                if (widget.element) {
                                    widget.element.style.opacity = isVideoMode ? "1.0" : "0.5";
                                    widget.element.style.pointerEvents = isVideoMode ? "auto" : "none";
                                }
                            }
                        });
                        
                        // Update image widgets  
                        this.imageWidgets.forEach(widget => {
                            if (widget) {
                                widget.disabled = isVideoMode;
                                // Apply visual styling for disabled state
                                if (widget.element) {
                                    widget.element.style.opacity = !isVideoMode ? "1.0" : "0.5";
                                    widget.element.style.pointerEvents = !isVideoMode ? "auto" : "none";
                                }
                            }
                        });
                        
                        // Update incremental save tooltip based on mode
                        if (this.incrementalSaveWidget) {
                            const modeText = isVideoMode ? "video files" : "image sequence folders";
                            this.incrementalSaveWidget.tooltip = `Enable automatic versioning - creates new numbered ${modeText} if output already exists`;
                        }
                        
                        // Force UI refresh
                        this.setDirtyCanvas(true, true);
                    };
                    
                    // Hook into widget value changes
                    const originalCallback = outputModeWidget.callback;
                    outputModeWidget.callback = (value) => {
                        // Call original callback if it exists
                        if (originalCallback) {
                            originalCallback.call(outputModeWidget, value);
                        }
                        
                        // Update widget states based on new mode
                        this.updateWidgetStates(value);
                    };
                    
                    // Initialize widget states based on current value
                    setTimeout(() => {
                        this.updateWidgetStates(outputModeWidget.value);
                    }, 100);
                }
                
                return r;
            };

            // Override widget draw method for visual feedback
            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                const r = onDrawBackground ? onDrawBackground.apply(this, arguments) : undefined;
                
                // Add subtle visual indicators for mode and incremental save
                const outputModeWidget = this.widgets.find(w => w.name === "output_mode");
                const incrementalSaveWidget = this.widgets.find(w => w.name === "incremental_save");
                
                if (outputModeWidget) {
                    const isVideoMode = outputModeWidget.value === "video";
                    const isIncrementalSave = incrementalSaveWidget ? incrementalSaveWidget.value : false;
                    
                    ctx.save();
                    
                    // Draw mode indicator in top-right corner
                    ctx.fillStyle = isVideoMode ? "#4CAF50" : "#2196F3";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "right";
                    ctx.fillText(isVideoMode ? "📹" : "🖼️", this.size[0] - 10, 25);
                    
                    // Draw incremental save indicator if enabled
                    if (isIncrementalSave) {
                        ctx.fillStyle = "#FF9800";
                        ctx.fillText("🔄", this.size[0] - 30, 25);
                    }
                    
                    ctx.restore();
                }
                
                return r;
            };
        }
    }
});