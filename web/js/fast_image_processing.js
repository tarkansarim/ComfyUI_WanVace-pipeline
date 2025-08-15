import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// Performance monitoring and enhanced controls for WAN Fast Image Processing nodes
class WANFastProcessingExtension {
    constructor() {
        this.performanceData = new Map(); // Store performance metrics per node
        this.lastProcessingTimes = new Map(); // Track processing times
        this.hardwareInfo = null; // Cache hardware information
    }

    async setup() {
        // Initialize hardware detection
        await this.detectHardwareCapabilities();
        
        // Hook into graph execution for performance monitoring
        const original = app.graphToPrompt;
        app.graphToPrompt = async function() {
            // Pre-execution setup for performance monitoring
            const extension = app.extensions.find(ext => ext instanceof WANFastProcessingExtension);
            if (extension) {
                extension.onPreExecution();
            }
            
            return original.apply(this, arguments);
        };
    }

    async detectHardwareCapabilities() {
        // Detect available hardware for optimization recommendations
        try {
            // Try to detect CUDA availability through a simple API call
            const response = await api.fetchApi('/system/stats', {
                method: 'GET',
                cache: 'no-cache'
            });
            
            if (response.ok) {
                const stats = await response.json();
                this.hardwareInfo = {
                    hasCUDA: stats.cuda_available || false,
                    totalMemory: stats.total_memory || 'Unknown',
                    deviceName: stats.device_name || 'CPU',
                    cores: navigator.hardwareConcurrency || 4
                };
            }
        } catch (error) {
            // Fallback hardware detection
            this.hardwareInfo = {
                hasCUDA: false,
                totalMemory: 'Unknown',
                deviceName: 'CPU',
                cores: navigator.hardwareConcurrency || 4
            };
        }
        
        console.log('🚀 WAN Fast Processing - Hardware detected:', this.hardwareInfo);
    }

    onPreExecution() {
        // Called before graph execution to prepare performance monitoring
        // Reset performance counters
        this.performanceData.clear();
        
        // Add timestamp for overall execution tracking
        this.executionStartTime = performance.now();
    }

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const isWANFastNode = nodeData.name?.startsWith('WAN') && 
                             nodeData.category?.includes('Fast Processing');
        
        if (!isWANFastNode) return;

        // Add performance monitoring widget to WAN Fast nodes
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            
            // Add performance monitoring widget
            this.addWidget("info", "performance_monitor", "", () => {}, {
                serialize: false
            });
            
            // Add optimization suggestions widget
            this.addWidget("info", "optimization_hints", "", () => {}, {
                serialize: false
            });
            
            // Initialize performance tracking for this node
            const extension = app.extensions.find(ext => ext instanceof WANFastProcessingExtension);
            if (extension) {
                extension.initializeNodePerformanceTracking(this);
            }
            
            return result;
        };

        // Hook into execution completion for performance reporting
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            const result = onExecuted?.apply(this, arguments);
            
            const extension = app.extensions.find(ext => ext instanceof WANFastProcessingExtension);
            if (extension && message.performance_info) {
                extension.updateNodePerformance(this, message.performance_info);
            }
            
            return result;
        };
    }

    initializeNodePerformanceTracking(node) {
        // Initialize performance tracking for a node
        
        // Add custom CSS for performance indicators
        if (!document.getElementById('wan-fast-processing-styles')) {
            const style = document.createElement('style');
            style.id = 'wan-fast-processing-styles';
            style.textContent = `
                .wan-performance-good { color: #4CAF50; font-weight: bold; }
                .wan-performance-medium { color: #FF9800; font-weight: bold; }
                .wan-performance-slow { color: #F44336; font-weight: bold; }
                .wan-optimization-hint { 
                    background: #E3F2FD; 
                    padding: 4px 8px; 
                    border-radius: 4px; 
                    margin: 2px 0;
                    font-size: 11px;
                    border-left: 3px solid #2196F3;
                }
                .wan-hardware-indicator {
                    display: inline-block;
                    padding: 2px 6px;
                    border-radius: 10px;
                    font-size: 10px;
                    font-weight: bold;
                    margin-right: 4px;
                }
                .wan-cuda-enabled { background: #4CAF50; color: white; }
                .wan-cuda-disabled { background: #757575; color: white; }
            `;
            document.head.appendChild(style);
        }

        // Initialize node-specific performance data
        this.performanceData.set(node.id, {
            lastExecutionTime: 0,
            averageExecutionTime: 0,
            executionCount: 0,
            lastThroughput: 0,
            lastMemoryUsage: 0,
            optimizationSuggestions: []
        });

        // Add real-time parameter validation and optimization hints
        this.addParameterOptimizationHints(node);
        
        // Display hardware capabilities
        this.updateHardwareDisplay(node);
    }

    addParameterOptimizationHints(node) {
        // Add real-time optimization hints based on parameters
        
        const updateOptimizationHints = () => {
            const hints = this.generateOptimizationHints(node);
            const hintsWidget = node.widgets?.find(w => w.name === "optimization_hints");
            
            if (hintsWidget && hints.length > 0) {
                hintsWidget.value = hints.map(hint => 
                    `💡 ${hint}`
                ).join('\n');
            }
        };

        // Hook into parameter changes
        if (node.widgets) {
            node.widgets.forEach(widget => {
                if (widget.type === "number" || widget.type === "combo") {
                    const originalCallback = widget.callback;
                    widget.callback = function(value) {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        setTimeout(updateOptimizationHints, 10);
                    };
                }
            });
        }

        // Initial hints update
        setTimeout(updateOptimizationHints, 100);
    }

    generateOptimizationHints(node) {
        // Generate optimization hints based on current parameters
        const hints = [];
        
        if (node.type === "WANFastImageBatchProcessor") {
            const skipFrames = node.widgets?.find(w => w.name === "skip_first_frames")?.value || 0;
            const nthFrame = node.widgets?.find(w => w.name === "select_every_nth")?.value || 1;
            const cap = node.widgets?.find(w => w.name === "frame_load_cap")?.value || 0;
            
            // Detect optimization pattern
            if (skipFrames === 0 && nthFrame === 1 && cap > 0) {
                hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Simple truncation (tensor view)");
            } else if (skipFrames > 0 && nthFrame === 1 && cap === 0) {
                hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Simple skip (tensor view)");
            } else if (skipFrames === 0 && nthFrame > 1 && cap === 0) {
                hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Simple sampling (tensor view)");
            } else if (skipFrames > 0 && nthFrame === 1 && cap > 0) {
                hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Skip + truncation (tensor view)");
            } else if (skipFrames === 0 && nthFrame > 1 && cap > 0) {
                hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Sampling + truncation (tensor view)");
            } else {
                hints.push("🚀 GPU-optimized indexing (no device transfers)");
            }
            
            if (nthFrame === 1 && skipFrames === 0 && cap === 0) {
                hints.push("💡 No-op detected - output will be identical to input");
            }
            
            if (nthFrame > 10) {
                hints.push("⚠️ High nth value may cause temporal discontinuities");
            }
            
        } else if (node.type === "WANFastImageCompositeMasked") {
            const resize = node.widgets?.find(w => w.name === "resize_source")?.value;
            
            hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Full GPU vectorization enabled");
            
            if (resize) {
                hints.push("Source resizing enabled - optimized with antialias");
            }
            
            if (this.hardwareInfo?.hasCUDA) {
                hints.push("⚡ CUDA detected - expect 10-15x speedup over CPU");
            } else {
                hints.push("💡 Consider CUDA for maximum performance gains");
            }
            
        } else if (node.type === "WANFastImageScaleBy") {
            const scaleFactor = node.widgets?.find(w => w.name === "scale_by")?.value || 1.0;
            const method = node.widgets?.find(w => w.name === "upscale_method")?.value;
            
            hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Zero format conversions + GPU vectorization");
            
            if (Math.abs(scaleFactor - 1.0) < 0.001) {
                hints.push("💡 No-op detected - scale factor ~1.0, output will be identical");
            } else if (scaleFactor > 2.0) {
                hints.push("⚡ Large upscale - GPU acceleration highly beneficial");
            } else if (scaleFactor < 0.5) {
                hints.push("📉 Downscaling - excellent performance expected");
            }
            
            if (method === "lanczos") {
                hints.push("🎨 Lanczos mode uses bicubic+antialias for best quality");
            }
            
        } else if (node.type === "WANFastImageScaleToMegapixels") {
            const megapixels = node.widgets?.find(w => w.name === "megapixels")?.value || 1.0;
            const method = node.widgets?.find(w => w.name === "upscale_method")?.value;
            
            hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Vectorized math + GPU interpolation");
            
            if (megapixels >= 4.0) {
                hints.push("⚡ High megapixel target - GPU acceleration essential");
            } else if (megapixels < 0.1) {
                hints.push("📉 Low megapixel target - ultra-fast downscaling");
            }
            
            hints.push(`🎯 Target: ${megapixels}MP - automatic scale factor calculation`);
            
        } else if (node.type === "WANFastImageResize") {
            const width = node.widgets?.find(w => w.name === "width")?.value || 512;
            const height = node.widgets?.find(w => w.name === "height")?.value || 512;
            const crop = node.widgets?.find(w => w.name === "crop")?.value;
            
            hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Direct dimension resize + aspect ratio handling");
            
            const megapixels = (width * height) / (1024 * 1024);
            hints.push(`🎯 Target: ${width}x${height} (${megapixels.toFixed(2)}MP)`);
            
            if (crop === "center") {
                hints.push("✂️ Center cropping enabled - maintains aspect ratio");
            } else {
                hints.push("🔧 No cropping - may distort aspect ratio");
            }
            
        } else if (node.type === "WANFastImageBlend") {
            const blendMode = node.widgets?.find(w => w.name === "blend_mode")?.value;
            const factor = node.widgets?.find(w => w.name === "blend_factor")?.value || 0.5;
            
            if (blendMode === "soft_light") {
                hints.push("Soft light mode is computationally intensive");
            }
            
            if (factor === 0.0) {
                hints.push("Blend factor is 0 - output will be image1 unchanged");
            } else if (factor === 1.0) {
                hints.push("Blend factor is 1 - output will be pure blend mode result");
            }
        }

        // Hardware-specific hints
        if (this.hardwareInfo) {
            if (!this.hardwareInfo.hasCUDA) {
                hints.push("🖥️ CUDA not available - processing on CPU");
            } else {
                hints.push("⚡ CUDA acceleration available");
            }
            
            if (this.hardwareInfo.cores <= 2) {
                hints.push("⚠️ Limited CPU cores - threading may not provide benefits");
            }
        }

        return hints;
    }

    updateHardwareDisplay(node) {
        // Update hardware capability display
        if (!this.hardwareInfo) return;
        
        // Add hardware indicator to node title or create a widget
        const performanceWidget = node.widgets?.find(w => w.name === "performance_monitor");
        if (performanceWidget) {
            const hardwareStatus = this.hardwareInfo.hasCUDA ? 
                '<span class="wan-hardware-indicator wan-cuda-enabled">⚡ CUDA</span>' :
                '<span class="wan-hardware-indicator wan-cuda-disabled">🖥️ CPU</span>';
            
            performanceWidget.value = `${hardwareStatus} ${this.hardwareInfo.cores} cores`;
        }
    }

    updateNodePerformance(node, performanceInfo) {
        // Update node performance display with execution results
        
        const nodeData = this.performanceData.get(node.id);
        if (!nodeData) return;
        
        // Parse performance info
        const executionTime = this.extractExecutionTime(performanceInfo);
        const throughput = this.extractThroughput(performanceInfo);
        const memoryUsage = this.extractMemoryUsage(performanceInfo);
        
        // Update performance data
        nodeData.lastExecutionTime = executionTime;
        nodeData.executionCount++;
        nodeData.averageExecutionTime = 
            (nodeData.averageExecutionTime * (nodeData.executionCount - 1) + executionTime) / 
            nodeData.executionCount;
        nodeData.lastThroughput = throughput;
        nodeData.lastMemoryUsage = memoryUsage;
        
        // Update performance display
        const performanceWidget = node.widgets?.find(w => w.name === "performance_monitor");
        if (performanceWidget) {
            const performanceClass = this.getPerformanceClass(throughput, executionTime);
            const display = `
                <div class="${performanceClass}">
                    ⚡ ${executionTime.toFixed(3)}s (${throughput.toFixed(1)} fps)
                </div>
                <div>🧠 ${memoryUsage.toFixed(1)}MB | Avg: ${nodeData.averageExecutionTime.toFixed(3)}s</div>
                <div>📊 Runs: ${nodeData.executionCount}</div>
            `;
            performanceWidget.value = display;
        }
        
        // Log performance data for debugging
        console.log(`🚀 ${node.type} Performance:`, {
            executionTime,
            throughput,
            memoryUsage,
            nodeId: node.id
        });
    }

    extractExecutionTime(performanceInfo) {
        const match = performanceInfo.match(/Processing time: ([\d.]+)s/);
        return match ? parseFloat(match[1]) : 0;
    }

    extractThroughput(performanceInfo) {
        const match = performanceInfo.match(/Throughput: ([\d.]+) fps/);
        return match ? parseFloat(match[1]) : 0;
    }

    extractMemoryUsage(performanceInfo) {
        const match = performanceInfo.match(/Memory used: ~([\d.]+)MB/);
        return match ? parseFloat(match[1]) : 0;
    }

    getPerformanceClass(throughput, executionTime) {
        // Determine performance indicator class based on metrics
        // Updated thresholds for ultra-optimized nodes
        if (throughput > 30 && executionTime < 5.0) {
            return "wan-performance-good";
        } else if (throughput > 10 && executionTime < 15.0) {
            return "wan-performance-medium";
        } else {
            return "wan-performance-slow";
        }
    }

    // Batch size optimization suggestions
    generateBatchSizeRecommendations(nodeType, currentBatchSize) {
        // Generate batch size recommendations based on hardware
        const recommendations = [];
        
        if (!this.hardwareInfo) return recommendations;
        
        if (nodeType === "WANFastImageBatchProcessor") {
            if (currentBatchSize > 1000 && !this.hardwareInfo.hasCUDA) {
                recommendations.push("Large batch on CPU - consider reducing batch size or using CUDA");
            } else if (currentBatchSize < 100 && this.hardwareInfo.hasCUDA) {
                recommendations.push("Small batch with CUDA - consider larger batches for better GPU utilization");
            }
        }
        
        return recommendations;
    }
}

// Enhanced parameter validation widget
class WANParameterValidator {
    static validateBatchProcessorParams(skip, nth, cap, totalFrames) {
        const warnings = [];
        
        if (skip >= totalFrames) {
            warnings.push("⚠️ Skip exceeds total frames - no output will be generated");
        }
        
        if (nth > totalFrames / 2) {
            warnings.push("⚠️ Large nth value - very few frames will be selected");
        }
        
        if (cap > 0 && cap > totalFrames) {
            warnings.push("💡 Cap exceeds available frames - cap will have no effect");
        }
        
        const estimatedOutput = Math.max(0, Math.floor((totalFrames - skip) / nth));
        if (cap > 0) {
            estimatedOutput = Math.min(estimatedOutput, cap);
        }
        
        return {
            warnings,
            estimatedOutput,
            reductionPercentage: totalFrames > 0 ? (1 - estimatedOutput / totalFrames) * 100 : 0
        };
    }
}

// Real-time performance dashboard
class WANPerformanceDashboard {
    constructor() {
        this.isVisible = false;
        this.dashboardElement = null;
    }

    createDashboard() {
        if (this.dashboardElement) return;
        
        this.dashboardElement = document.createElement('div');
        this.dashboardElement.id = 'wan-performance-dashboard';
        this.dashboardElement.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            width: 300px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            z-index: 10000;
            border: 2px solid #2196F3;
            display: none;
            max-height: 400px;
            overflow-y: auto;
        `;
        
        document.body.appendChild(this.dashboardElement);
    }

    show() {
        this.createDashboard();
        this.dashboardElement.style.display = 'block';
        this.isVisible = true;
        this.updateDashboard();
    }

    hide() {
        if (this.dashboardElement) {
            this.dashboardElement.style.display = 'none';
        }
        this.isVisible = false;
    }

    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    updateDashboard() {
        if (!this.isVisible || !this.dashboardElement) return;
        
        const extension = app.extensions.find(ext => ext instanceof WANFastProcessingExtension);
        if (!extension) return;
        
        let content = '<div style="text-align: center; margin-bottom: 10px;"><strong>🚀 WAN Fast Processing Dashboard</strong></div>';
        
        // Hardware info
        if (extension.hardwareInfo) {
            content += `
                <div style="margin-bottom: 10px;">
                    <strong>💻 Hardware:</strong><br>
                    Device: ${extension.hardwareInfo.deviceName}<br>
                    CUDA: ${extension.hardwareInfo.hasCUDA ? '✅' : '❌'}<br>
                    Cores: ${extension.hardwareInfo.cores}<br>
                    Memory: ${extension.hardwareInfo.totalMemory}
                </div>
            `;
        }
        
        // Performance data
        content += '<div style="margin-bottom: 10px;"><strong>📊 Active Nodes:</strong></div>';
        
        extension.performanceData.forEach((data, nodeId) => {
            if (data.executionCount > 0) {
                content += `
                    <div style="margin-bottom: 8px; padding: 6px; background: rgba(255,255,255,0.1); border-radius: 4px;">
                        <strong>Node ${nodeId}:</strong><br>
                        Last: ${data.lastExecutionTime.toFixed(3)}s<br>
                        Avg: ${data.averageExecutionTime.toFixed(3)}s<br>
                        Throughput: ${data.lastThroughput.toFixed(1)} fps<br>
                        Memory: ${data.lastMemoryUsage.toFixed(1)}MB<br>
                        Runs: ${data.executionCount}
                    </div>
                `;
            }
        });
        
        this.dashboardElement.innerHTML = content;
    }
}

// Initialize extension
const dashboard = new WANPerformanceDashboard();

// Add keyboard shortcut for dashboard
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === 'P') {
        e.preventDefault();
        dashboard.toggle();
    }
});

// Register extension
app.registerExtension({
    name: "WAN.FastImageProcessing",
    async setup() {
        const extension = new WANFastProcessingExtension();
        await extension.setup();
        app.extensions.push(extension);
        
        console.log("🚀 WAN Fast Image Processing Extension loaded");
        console.log("💡 Press Ctrl+Shift+P to toggle performance dashboard");
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const extension = app.extensions.find(ext => ext instanceof WANFastProcessingExtension);
        if (extension) {
            await extension.beforeRegisterNodeDef(nodeType, nodeData, app);
        }
    }
});