import { app } from "../../../../scripts/app.js";

// Frame Injector Parameter Graying - Shows/hides parameters based on injection mode
app.registerExtension({
    name: "WANVace.FrameInjector",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WANVaceFrameInjector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Find the injection_mode widget
                const injectionModeWidget = this.widgets.find(w => w.name === "injection_mode");
                
                if (injectionModeWidget) {
                    // Store references to widgets for easy access
                    this.manualModeWidgets = [
                        this.widgets.find(w => w.name === "injection_positions"),
                        this.widgets.find(w => w.name === "inject_indices")
                    ].filter(w => w);
                    
                    this.nthFrameModeWidgets = [
                        this.widgets.find(w => w.name === "nth_interval"),
                        this.widgets.find(w => w.name === "start_offset"),
                        this.widgets.find(w => w.name === "include_last_frame"),
                        this.widgets.find(w => w.name === "inject_skip_frames")
                    ].filter(w => w);
                    
                    // Get mode-specific widgets
                    this.replaceModeWidget = this.widgets.find(w => w.name === "replace_mode");
                    this.truncateWidget = this.widgets.find(w => w.name === "truncate_after_last");
                    
                    // Function to update widget states
                    this.updateWidgetStates = function(mode) {
                        const isNthFrameMode = mode === "nth_frame";
                        
                        // Update manual mode widgets
                        this.manualModeWidgets.forEach(widget => {
                            if (widget) {
                                widget.disabled = isNthFrameMode;
                                // Apply visual styling for disabled state
                                if (widget.element) {
                                    widget.element.style.opacity = isNthFrameMode ? "0.5" : "1.0";
                                    widget.element.style.pointerEvents = isNthFrameMode ? "none" : "auto";
                                }
                            }
                        });
                        
                        // Update nth frame mode widgets
                        this.nthFrameModeWidgets.forEach(widget => {
                            if (widget) {
                                widget.disabled = !isNthFrameMode;
                                // Apply visual styling for disabled state
                                if (widget.element) {
                                    widget.element.style.opacity = !isNthFrameMode ? "0.5" : "1.0";
                                    widget.element.style.pointerEvents = !isNthFrameMode ? "none" : "auto";
                                }
                            }
                        });
                        
                        // Update tooltips to indicate which mode uses which parameters
                        const positionsWidget = this.widgets.find(w => w.name === "injection_positions");
                        const indicesWidget = this.widgets.find(w => w.name === "inject_indices");
                        const intervalWidget = this.widgets.find(w => w.name === "nth_interval");
                        
                        if (positionsWidget) {
                            positionsWidget.tooltip = isNthFrameMode 
                                ? "⚠️ Not used in nth_frame mode - positions are calculated automatically" 
                                : "Comma-separated base frame positions (manual mode only)";
                        }
                        
                        if (indicesWidget) {
                            indicesWidget.tooltip = isNthFrameMode 
                                ? "⚠️ Not used in nth_frame mode - uses sequential indices automatically" 
                                : "Comma-separated indices of inject frames (manual mode only)";
                        }
                        
                        if (intervalWidget) {
                            intervalWidget.tooltip = !isNthFrameMode 
                                ? "⚠️ Only used in nth_frame mode" 
                                : "Inject every Nth frame (nth_frame mode only)";
                        }
                        
                        // Force UI refresh
                        this.setDirtyCanvas(true, true);
                    };
                    
                    // Function to update truncate widget based on replace mode
                    this.updateTruncateWidget = function() {
                        if (this.truncateWidget && this.replaceModeWidget) {
                            const isReplaceMode = this.replaceModeWidget.value;
                            this.truncateWidget.disabled = !isReplaceMode;
                            
                            if (this.truncateWidget.element) {
                                this.truncateWidget.element.style.opacity = isReplaceMode ? "1.0" : "0.5";
                                this.truncateWidget.element.style.pointerEvents = isReplaceMode ? "auto" : "none";
                            }
                            
                            this.truncateWidget.tooltip = isReplaceMode 
                                ? "Remove frames after the last injection position (replace mode only)"
                                : "⚠️ Only available in replace mode";
                        }
                    };
                    
                    // Hook into widget value changes
                    const originalCallback = injectionModeWidget.callback;
                    injectionModeWidget.callback = (value) => {
                        // Call original callback if it exists
                        if (originalCallback) {
                            originalCallback.call(injectionModeWidget, value);
                        }
                        
                        // Update widget states based on new mode
                        this.updateWidgetStates(value);
                    };
                    
                    // Hook into replace mode widget changes
                    if (this.replaceModeWidget) {
                        const originalReplaceModeCallback = this.replaceModeWidget.callback;
                        this.replaceModeWidget.callback = (value) => {
                            if (originalReplaceModeCallback) {
                                originalReplaceModeCallback.call(this.replaceModeWidget, value);
                            }
                            // Update truncate widget state
                            this.updateTruncateWidget();
                        };
                    }
                    
                    // Initialize widget states based on current value
                    setTimeout(() => {
                        this.updateWidgetStates(injectionModeWidget.value);
                        this.updateTruncateWidget();
                    }, 100);
                }
                
                return r;
            };

            // Override widget draw method for visual feedback
            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                const r = onDrawBackground ? onDrawBackground.apply(this, arguments) : undefined;
                
                // Add visual indicator for current mode
                const injectionModeWidget = this.widgets.find(w => w.name === "injection_mode");
                
                if (injectionModeWidget) {
                    const isNthFrameMode = injectionModeWidget.value === "nth_frame";
                    
                    ctx.save();
                    
                    // Draw mode indicator in top-right corner
                    ctx.fillStyle = isNthFrameMode ? "#4CAF50" : "#2196F3";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "right";
                    ctx.fillText(isNthFrameMode ? "📐 Nth" : "✏️ Manual", this.size[0] - 10, 25);
                    
                    ctx.restore();
                }
                
                return r;
            };
        }
    }
});