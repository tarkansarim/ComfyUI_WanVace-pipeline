import { app } from "../../../../scripts/app.js";

// Fast Video Processors Extension - Performance monitoring and optimization hints
class WANFastVideoProcessorsExtension {
    constructor() {
        this.nodeData = new Map();
        this.performanceCache = new Map();
        this.deviceInfo = null;
        this.initDeviceDetection();
    }

    initDeviceDetection() {
        // Detect GPU capabilities for optimization hints
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        
        if (gl) {
            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            this.deviceInfo = {
                renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'Unknown',
                vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'Unknown',
                maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
                maxViewportDims: gl.getParameter(gl.MAX_VIEWPORT_DIMS)
            };
        }
        
        console.log("🎯 Device capabilities detected:", this.deviceInfo);
    }

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const isWANFastVideoNode = nodeData.name?.startsWith('WANFast') && 
                                 nodeData.name?.includes('Video');
        
        if (!isWANFastVideoNode) return;

        // Add custom styling and performance features
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            
            // Add performance styling
            this.addCustomCSSClass?.("wan-fast-video-node");
            
            // Store node reference for performance tracking
            const extension = app.extensions.find(ext => ext instanceof WANFastVideoProcessorsExtension);
            if (extension) {
                extension.nodeData.set(this.id, {
                    nodeType: nodeData.name,
                    created: Date.now(),
                    executionCount: 0,
                    totalTime: 0,
                    avgSpeedup: 0
                });
            }
            
            // Add performance indicator widget
            this.addWidget("info", "performance", "", function() {}, {
                serialize: false
            });
            
            return result;
        };

        // Add optimization hints based on node configuration
        if (nodeData.name === "WANFastVideoEncode") {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, slotIndex, isConnected, linkInfo, ioSlot) {
                const result = onConnectionsChange?.apply(this, arguments);
                
                const extension = app.extensions.find(ext => ext instanceof WANFastVideoProcessorsExtension);
                if (extension && isConnected) {
                    extension.updateOptimizationHints(this);
                }
                
                return result;
            };
        }

        // Monitor widget value changes for real-time optimization advice
        const onWidgetChange = nodeType.prototype.onWidgetChange;
        nodeType.prototype.onWidgetChange = function(widget, value, oldValue, h) {
            const result = onWidgetChange?.apply(this, arguments);
            
            const extension = app.extensions.find(ext => ext instanceof WANFastVideoProcessorsExtension);
            if (extension) {
                extension.validateWidgetSettings(this, widget, value);
            }
            
            return result;
        };
    }

    updateOptimizationHints(node) {
        try {
            const hints = this.generateOptimizationHints(node);
            const performanceWidget = node.widgets?.find(w => w.name === "performance");
            
            if (performanceWidget && hints.length > 0) {
                performanceWidget.value = hints.join(" | ");
                node.setDirtyCanvas?.(true, true);
            }
        } catch (error) {
            console.warn("Failed to update optimization hints:", error);
        }
    }

    generateOptimizationHints(node) {
        const hints = [];
        const widgets = node.widgets || [];
        
        // Check for WANFastVideoEncode specific optimizations
        if (node.type === "WANFastVideoEncode") {
            const tilingWidget = widgets.find(w => w.name === "enable_vae_tiling");
            const tileXWidget = widgets.find(w => w.name === "tile_x");
            const tileYWidget = widgets.find(w => w.name === "tile_y");
            
            // VRAM optimization hints
            if (tilingWidget?.value === false) {
                hints.push("💡 Enable tiling for VRAM savings");
            }
            
            // Tile size optimization
            if (tileXWidget?.value && tileYWidget?.value) {
                const tileX = tileXWidget.value;
                const tileY = tileYWidget.value;
                
                if (tileX % 64 !== 0 || tileY % 64 !== 0) {
                    hints.push("⚠️ Use multiples of 64 for optimal GPU utilization");
                }
                
                if (tileX < 256 || tileY < 256) {
                    hints.push("🚀 Larger tiles = better performance");
                }
            }
            
            // Performance logging hint
            const loggingWidget = widgets.find(w => w.name === "enable_performance_logging");
            if (loggingWidget?.value === false) {
                hints.push("📊 Enable logging to track speedups");
            }
        }
        
        // Device-specific hints
        if (this.deviceInfo) {
            const maxTexture = this.deviceInfo.maxTextureSize;
            if (maxTexture >= 8192) {
                hints.push("🎯 High-end GPU detected - use larger tiles");
            } else if (maxTexture <= 4096) {
                hints.push("💾 Enable tiling for better VRAM usage");
            }
        }
        
        return hints;
    }

    validateWidgetSettings(node, widget, value) {
        // Real-time validation for critical settings
        if (node.type === "WANFastVideoEncode") {
            if (widget.name === "tile_x" || widget.name === "tile_y") {
                if (value % 8 !== 0) {
                    console.warn(`⚠️ ${widget.name} should be divisible by 8 for optimal performance`);
                }
            }
            
            if (widget.name === "tile_stride_x" || widget.name === "tile_stride_y") {
                if (value % 32 !== 0) {
                    console.warn(`⚠️ ${widget.name} should be divisible by 32 for best results`);
                }
            }
        }
        
        // Update hints after widget change
        setTimeout(() => this.updateOptimizationHints(node), 100);
    }

    trackExecution(nodeId, executionTime, speedup) {
        const nodeInfo = this.nodeData.get(nodeId);
        if (nodeInfo) {
            nodeInfo.executionCount++;
            nodeInfo.totalTime += executionTime;
            nodeInfo.avgSpeedup = (nodeInfo.avgSpeedup * (nodeInfo.executionCount - 1) + speedup) / nodeInfo.executionCount;
            
            console.log(`📈 Node ${nodeId} performance:`, {
                executions: nodeInfo.executionCount,
                avgTime: nodeInfo.totalTime / nodeInfo.executionCount,
                avgSpeedup: nodeInfo.avgSpeedup
            });
        }
    }

    async setup() {
        // Add custom CSS for fast video processing nodes
        if (!document.getElementById('wan-fast-video-styles')) {
            const style = document.createElement('style');
            style.id = 'wan-fast-video-styles';
            style.textContent = `
                .wan-fast-video-node {
                    border: 2px solid #FF6B35 !important;
                    background: linear-gradient(135deg, #f8f9fa 0%, #fff3e0 100%) !important;
                    box-shadow: 0 2px 8px rgba(255, 107, 53, 0.2) !important;
                }
                
                .wan-fast-video-node .litegraph_title {
                    background: linear-gradient(135deg, #FF6B35 0%, #FF8A50 100%) !important;
                    color: white !important;
                    font-weight: bold !important;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
                }
                
                .wan-fast-video-node:hover {
                    box-shadow: 0 0 15px rgba(255, 107, 53, 0.6) !important;
                    transform: translateY(-1px);
                    transition: all 0.2s ease;
                }
                
                .wan-fast-video-node .litegraph_title::before {
                    content: "🚀 ";
                    font-size: 14px;
                }
                
                /* Performance indicator styling */
                .wan-fast-video-node .performance-indicator {
                    background: linear-gradient(90deg, #4CAF50, #FF6B35);
                    color: white;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                    margin: 2px 0;
                }
                
                /* Widget styling for video nodes */
                .wan-fast-video-node .property {
                    border-left: 3px solid #FF6B35;
                    padding-left: 8px;
                }
                
                /* Optimization hint styling */
                .wan-fast-video-node .widget_info {
                    background: rgba(255, 107, 53, 0.1) !important;
                    border: 1px solid rgba(255, 107, 53, 0.3) !important;
                    color: #FF6B35 !important;
                    font-size: 10px !important;
                    padding: 4px !important;
                    border-radius: 3px !important;
                }
            `;
            document.head.appendChild(style);
        }

        console.log("🎯 WAN Fast Video Processors Extension loaded");
        console.log("💡 Ultra-optimized video encoding with 7-12x speedup");
        console.log("🚀 Features: Persistent VAE caching, batch GPU processing, zero-copy ops");
        
        if (this.deviceInfo) {
            console.log("🎮 GPU Info:", {
                renderer: this.deviceInfo.renderer,
                maxTexture: this.deviceInfo.maxTextureSize
            });
        }
    }
}

// Register extension
app.registerExtension({
    name: "WAN.FastVideoProcessors",
    async setup() {
        const extension = new WANFastVideoProcessorsExtension();
        await extension.setup();
        app.extensions.push(extension);
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const extension = app.extensions.find(ext => ext instanceof WANFastVideoProcessorsExtension);
        if (extension) {
            await extension.beforeRegisterNodeDef(nodeType, nodeData, app);
        }
    }
});