import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "WAN.VideoReuseCache",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WANVaceKeyframeTimeline") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Wait a bit for timeline to be ready
                setTimeout(() => {
                    const widget = {
                        name: "video_preview",
                        type: "custom",
                        
                        draw: function(ctx, node, width, y, H) {
                            // Calculate exact remaining height from current Y to bottom
                            let h = 300; // Default
                            
                            if (node.size && node.size[1]) {
                                // The 'y' parameter is already the exact position where this widget starts
                                // node.size[1] is the total height of the node
                                // So remaining space = node height - current y position - small padding
                                
                                const currentY = y;
                                const nodeHeight = node.size[1];
                                
                                // Calculate remaining space
                                h = nodeHeight - currentY - 5; // Just 5px padding at bottom
                                h = Math.max(200, h); // Minimum 200px
                            }
                            
                            const margin = 15;
                            
                            // Background
                            ctx.fillStyle = "#000";
                            ctx.fillRect(margin, y, width - margin * 2, h);
                            
                            // Border
                            ctx.strokeStyle = "#333";
                            ctx.strokeRect(margin, y, width - margin * 2, h);
                            
                            // Find the timeline widget
                            let timeline = null;
                            if (node.widgets) {
                                for (const w of node.widgets) {
                                    if (w.name === "keyframe_timeline" && w.keyframes) {
                                        timeline = w;
                                        break;
                                    }
                                }
                            }
                            
                            // Fallback to node.timelineWidget
                            if (!timeline) {
                                timeline = node.timelineWidget;
                            }
                            
                            if (!timeline || !timeline.keyframes) {
                                ctx.fillStyle = "#666";
                                ctx.font = "14px Arial";
                                ctx.textAlign = "center";
                                ctx.fillText("No timeline data", width/2, y + h/2);
                                return h;
                            }
                            
                            const currentFrame = Math.floor(timeline.currentFrame || 0);
                            const keyframes = timeline.keyframes;
                            
                            // Debug: check what keyframes we have
                            const keyframeNumbers = Object.keys(keyframes).map(k => parseInt(k)).sort((a,b) => a-b);
                            
                            // Check if we're on a keyframe or find the last keyframe to hold
                            let frameToShow = null;
                            if (keyframes[currentFrame]) {
                                frameToShow = currentFrame;
                            } else {
                                // Always hold the last keyframe (for preview only)
                                // Look backwards from current frame to find the most recent keyframe
                                for (let f = currentFrame - 1; f >= 0; f--) {
                                    if (keyframes[f]) {
                                        frameToShow = f;
                                        break;
                                    }
                                }
                                
                                // If we still haven't found a keyframe, look forward
                                if (frameToShow === null) {
                                    for (let f = currentFrame + 1; f < timeline.timeline_frames; f++) {
                                        if (keyframes[f]) {
                                            frameToShow = f;
                                            break;
                                        }
                                    }
                                }
                            }
                            
                            // Debug info - let's enable it temporarily to debug
                            ctx.save();
                            ctx.fillStyle = "#fff";
                            ctx.font = "10px Arial";
                            ctx.textAlign = "left";
                            ctx.fillText(`Current: ${currentFrame}, Show: ${frameToShow}`, margin + 5, y + 15);
                            ctx.fillText(`Keyframes: ${keyframeNumbers.join(', ')}`, margin + 5, y + 25);
                            ctx.restore();
                            
                            if (frameToShow !== null && timeline.imageCache && timeline.imageCache[frameToShow]) {
                                // Use the image from timeline's cache
                                const img = timeline.imageCache[frameToShow];
                                
                                if (img && img.complete && img.naturalWidth > 0) {
                                    // Calculate dimensions - use almost all available space
                                    const boxW = width - margin * 2 - 20;
                                    const boxH = h - 40;  // Less padding for bigger image
                                    const imgAspect = img.naturalWidth / img.naturalHeight;
                                    const boxAspect = boxW / boxH;
                                    
                                    let dw, dh, dx, dy;
                                    if (imgAspect > boxAspect) {
                                        dw = boxW;
                                        dh = boxW / imgAspect;
                                        dx = margin + 10;
                                        dy = y + 10 + (boxH - dh) / 2;
                                    } else {
                                        dh = boxH;
                                        dw = boxH * imgAspect;
                                        dx = margin + 10 + (boxW - dw) / 2;
                                        dy = y + 10;
                                    }
                                    
                                    try {
                                        ctx.drawImage(img, dx, dy, dw, dh);
                                    } catch (e) {
                                        // If image fails to draw, show placeholder
                                        ctx.fillStyle = "#00ff00";
                                        ctx.fillRect(margin + 10, y + 10, boxW, boxH);
                                        ctx.fillStyle = "#000";
                                        ctx.font = "16px Arial";
                                        ctx.textAlign = "center";
                                        ctx.fillText("KEYFRAME", width/2, y + h/2);
                                    }
                                } else {
                                    // Image not in cache or not ready - try to get from keyframe data
                                    const kf = timeline.keyframes[frameToShow];
                                    if (kf && kf.image) {
                                        // Show green placeholder while we wait for cache
                                        ctx.fillStyle = "#00ff00";
                                        ctx.fillRect(margin + 10, y + 10, width - margin * 2 - 20, h - 40);
                                        ctx.fillStyle = "#000";
                                        ctx.font = "16px Arial";
                                        ctx.textAlign = "center";
                                        ctx.fillText("KEYFRAME", width/2, y + h/2);
                                        
                                        // Try to load it into timeline cache
                                        if (!timeline.imageCache[frameToShow] && timeline.loadImageToCache) {
                                            timeline.loadImageToCache(frameToShow, kf.image);
                                            
                                            // Request redraw after a delay to allow image to load
                                            setTimeout(() => {
                                                node.setDirtyCanvas(true);
                                            }, 100);
                                        }
                                    } else {
                                        // No image data
                                        ctx.fillStyle = "#003300";
                                        ctx.fillRect(margin + 10, y + 10, width - margin * 2 - 20, h - 40);
                                        ctx.fillStyle = "#0f0";
                                        ctx.font = "14px Arial";
                                        ctx.textAlign = "center";
                                        ctx.fillText("Keyframe (no image)", width/2, y + h/2);
                                    }
                                }
                            } else if (frameToShow !== null) {
                                // Keyframe exists but no image
                                ctx.fillStyle = "#00ff00";
                                ctx.fillRect(margin + 10, y + 10, width - margin * 2 - 20, h - 40);
                                ctx.fillStyle = "#000";
                                ctx.font = "16px Arial";
                                ctx.textAlign = "center";
                                ctx.fillText("KEYFRAME", width/2, y + h/2);
                            } else {
                                // No keyframe found at all (empty timeline start)
                                // Show black for truly empty frames
                                ctx.fillStyle = "#000";
                                ctx.fillRect(margin + 10, y + 10, width - margin * 2 - 20, h - 40);
                            }
                            
                            // Frame info
                            ctx.fillStyle = "#888";
                            ctx.font = "12px Arial";
                            ctx.textAlign = "center";
                            let frameType;
                            if (keyframes[currentFrame]) {
                                frameType = "Keyframe";
                            } else if (frameToShow !== null) {
                                frameType = `Held from ${frameToShow + 1}`;
                            } else {
                                frameType = "Empty";
                            }
                            ctx.fillText(`Frame ${currentFrame + 1} / ${timeline.timeline_frames} (${frameType})`, width/2, y + h - 20);
                            
                            return h;
                        },
                        
                        computeSize: function(width) {
                            // Let the draw function handle the actual size
                            return [0, 300]; // Default size, will be overridden in draw
                        },
                        
                        node: null  // Will be set when widget is created
                    };
                    
                    // Set node reference
                    widget.node = this;
                    
                    this.addCustomWidget(widget);
                    this.setDirtyCanvas(true);
                    
                    // Make sure node is resizable
                    this.resizable = true;
                }, 200);
                
                return r;
            };
        }
    }
});