import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// Performance monitoring for WAN Fast ControlNet Processor nodes
class WANFastControlNetExtension {
    constructor() {
        this.performanceData = new Map();
        this.hardwareInfo = null;
    }

    async setup() {
        // Detect hardware capabilities
        await this.detectHardwareCapabilities();
        
        // Hook into graph execution for performance monitoring
        const original = app.graphToPrompt;
        app.graphToPrompt = async function() {
            const extension = app.extensions.find(ext => ext instanceof WANFastControlNetExtension);
            if (extension) {
                extension.onPreExecution();
            }
            return original.apply(this, arguments);
        };
    }

    async detectHardwareCapabilities() {
        try {
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
            this.hardwareInfo = {
                hasCUDA: false,
                totalMemory: 'Unknown',
                deviceName: 'CPU',
                cores: navigator.hardwareConcurrency || 4
            };
        }
        
        console.log('🚀 WAN Fast ControlNet - Hardware detected:', this.hardwareInfo);
    }

    onPreExecution() {
        this.performanceData.clear();
        this.executionStartTime = performance.now();
    }

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const isWANFastControlNetNode = nodeData.name?.startsWith('WANFast') && 
                                       (nodeData.name?.includes('DepthAnything') || 
                                        nodeData.name?.includes('DWPose'));
        
        if (!isWANFastControlNetNode) return;

        // Add performance monitoring widget
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
            
            // Initialize performance tracking
            const extension = app.extensions.find(ext => ext instanceof WANFastControlNetExtension);
            if (extension) {
                extension.initializeNodePerformanceTracking(this);
            }
            
            return result;
        };

        // Hook into execution completion for performance reporting
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            const result = onExecuted?.apply(this, arguments);
            
            const extension = app.extensions.find(ext => ext instanceof WANFastControlNetExtension);
            if (extension && message.performance_info) {
                extension.updateNodePerformance(this, message.performance_info);
            }
            
            return result;
        };
    }

    initializeNodePerformanceTracking(node) {
        // Add custom CSS for ControlNet performance indicators
        if (!document.getElementById('wan-fast-controlnet-styles')) {
            const style = document.createElement('style');
            style.id = 'wan-fast-controlnet-styles';
            style.textContent = `
                .wan-controlnet-good { color: #4CAF50; font-weight: bold; }
                .wan-controlnet-medium { color: #FF9800; font-weight: bold; }
                .wan-controlnet-slow { color: #F44336; font-weight: bold; }
                .wan-controlnet-hint { 
                    background: #E8F5E8; 
                    padding: 4px 8px; 
                    border-radius: 4px; 
                    margin: 2px 0;
                    font-size: 11px;
                    border-left: 3px solid #4CAF50;
                }
                .wan-model-indicator {
                    display: inline-block;
                    padding: 2px 6px;
                    border-radius: 10px;
                    font-size: 10px;
                    font-weight: bold;
                    margin-right: 4px;
                    background: #2196F3;
                    color: white;
                }
            `;
            document.head.appendChild(style);
        }

        // Initialize performance data
        this.performanceData.set(node.id, {
            lastExecutionTime: 0,
            averageExecutionTime: 0,
            executionCount: 0,
            lastThroughput: 0,
            lastMemoryUsage: 0,
            modelType: node.type || 'Unknown'
        });

        // Add parameter optimization hints
        this.addControlNetOptimizationHints(node);
        this.updateHardwareDisplay(node);
    }

    addControlNetOptimizationHints(node) {
        const updateOptimizationHints = () => {
            const hints = this.generateControlNetOptimizationHints(node);
            const hintsWidget = node.widgets?.find(w => w.name === "optimization_hints");
            
            if (hintsWidget && hints.length > 0) {
                hintsWidget.value = hints.map(hint => `💡 ${hint}`).join('\\n');
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

        setTimeout(updateOptimizationHints, 100);
    }

    generateControlNetOptimizationHints(node) {
        const hints = [];
        
        if (node.type === "WANFastDepthAnythingV2") {
            const ckptName = node.widgets?.find(w => w.name === "ckpt_name")?.value;
            const resolution = node.widgets?.find(w => w.name === "resolution")?.value || 512;
            const maxDepth = node.widgets?.find(w => w.name === "max_depth")?.value || 1.0;
            
            hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Batch GPU processing + persistent model caching");
            
            if (ckptName) {
                const modelSize = {
                    "depth_anything_v2_vits.pth": "Small (fastest)",
                    "depth_anything_v2_vitb.pth": "Base (balanced)",
                    "depth_anything_v2_vitl.pth": "Large (best quality)",
                    "depth_anything_v2_vitg.pth": "Giant (highest quality, slower)"
                };
                hints.push(`📊 Model: ${modelSize[ckptName] || ckptName}`);
            }
            
            if (resolution >= 1024) {
                hints.push("⚡ High resolution - GPU acceleration highly beneficial");
            } else if (resolution <= 256) {
                hints.push("📉 Low resolution - ultra-fast processing expected");
            }
            
            if (maxDepth !== 1.0) {
                hints.push(`🎯 Custom max depth: ${maxDepth} (metric mode optimization)`);
            }
            
            if (this.hardwareInfo?.hasCUDA) {
                hints.push("⚡ CUDA detected - expect 8-15x speedup over CPU");
            } else {
                hints.push("💡 Consider CUDA for maximum depth estimation performance");
            }
            
        } else if (node.type === "WANFastDWPose") {
            const detectBody = node.widgets?.find(w => w.name === "detect_body")?.value;
            const detectHand = node.widgets?.find(w => w.name === "detect_hand")?.value;
            const detectFace = node.widgets?.find(w => w.name === "detect_face")?.value;
            const bboxDetector = node.widgets?.find(w => w.name === "bbox_detector")?.value;
            const poseEstimator = node.widgets?.find(w => w.name === "pose_estimator")?.value;
            
            hints.push("🚀🚀🚀 ULTRA-OPTIMIZED: Vectorized pose detection + GPU canvas operations");
            
            const activeDetections = [];
            if (detectBody === "enable") activeDetections.push("Body");
            if (detectHand === "enable") activeDetections.push("Hand");
            if (detectFace === "enable") activeDetections.push("Face");
            
            if (activeDetections.length > 0) {
                hints.push(`🎯 Detecting: ${activeDetections.join(", ")}`);
            } else {
                hints.push("⚠️ No detection enabled - output will be blank");
            }
            
            if (bboxDetector === "None") {
                hints.push("📦 No bbox detector - using pose-only detection");
            } else if (bboxDetector.includes("torchscript")) {
                hints.push("⚡ TorchScript bbox detector - optimized for batch processing");
            } else if (bboxDetector.includes("onnx")) {
                hints.push("🔧 ONNX bbox detector - good cross-platform performance");
            }
            
            if (poseEstimator.includes("torchscript")) {
                hints.push("⚡ TorchScript pose estimator - batch-optimized inference");
            } else if (poseEstimator.includes("onnx")) {
                hints.push("🔧 ONNX pose estimator - broad compatibility");
            }
            
            if (this.hardwareInfo?.hasCUDA) {
                hints.push("⚡ CUDA detected - expect 10-20x speedup over CPU");
            } else {
                hints.push("💡 Consider CUDA for maximum pose detection performance");
            }
        }

        // Hardware-specific hints
        if (this.hardwareInfo) {
            if (!this.hardwareInfo.hasCUDA) {
                hints.push("🖥️ CPU-only processing - consider GPU acceleration");
            }
            
            if (this.hardwareInfo.cores <= 2) {
                hints.push("⚠️ Limited CPU cores - GPU acceleration recommended");
            }
        }

        return hints;
    }

    updateHardwareDisplay(node) {
        if (!this.hardwareInfo) return;
        
        const performanceWidget = node.widgets?.find(w => w.name === "performance_monitor");
        if (performanceWidget) {
            const modelIndicator = `<span class="wan-model-indicator">${node.type?.replace('WANFast', '') || 'ControlNet'}</span>`;
            const hardwareStatus = this.hardwareInfo.hasCUDA ? 
                '<span class="wan-hardware-indicator wan-cuda-enabled">⚡ CUDA</span>' :
                '<span class="wan-hardware-indicator wan-cuda-disabled">🖥️ CPU</span>';
            
            performanceWidget.value = `${modelIndicator}${hardwareStatus} ${this.hardwareInfo.cores} cores`;
        }
    }

    updateNodePerformance(node, performanceInfo) {
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
            const performanceClass = this.getControlNetPerformanceClass(throughput, executionTime);
            const speedupInfo = performanceInfo.includes('speedup') ? 
                performanceInfo.match(/speedup: ~([\\d.]+)x/)?.[1] || '?' : '?';
                
            const display = `
                <div class="${performanceClass}">
                    ⚡ ${executionTime.toFixed(3)}s (${throughput.toFixed(1)} fps)
                </div>
                <div>🧠 ${memoryUsage.toFixed(1)}MB | Speedup: ${speedupInfo}x</div>
                <div>📊 Runs: ${nodeData.executionCount} | Avg: ${nodeData.averageExecutionTime.toFixed(3)}s</div>
            `;
            performanceWidget.value = display;
        }
        
        // Log performance data
        console.log(`🚀 ${node.type} Performance:`, {
            executionTime,
            throughput,
            memoryUsage,
            nodeId: node.id,
            modelType: nodeData.modelType
        });
    }

    extractExecutionTime(performanceInfo) {
        const match = performanceInfo.match(/Processing time: ([\\d.]+)s/);
        return match ? parseFloat(match[1]) : 0;
    }

    extractThroughput(performanceInfo) {
        const match = performanceInfo.match(/Throughput: ([\\d.]+) fps/);
        return match ? parseFloat(match[1]) : 0;
    }

    extractMemoryUsage(performanceInfo) {
        const match = performanceInfo.match(/Memory used: ~([\\d.]+)MB/);
        return match ? parseFloat(match[1]) : 0;
    }

    getControlNetPerformanceClass(throughput, executionTime) {
        // ControlNet-specific performance thresholds
        if (throughput > 20 && executionTime < 10.0) {
            return "wan-controlnet-good";
        } else if (throughput > 5 && executionTime < 30.0) {
            return "wan-controlnet-medium";
        } else {
            return "wan-controlnet-slow";
        }
    }
}

// Register extension
app.registerExtension({
    name: "WAN.FastControlNetProcessors",
    async setup() {
        const extension = new WANFastControlNetExtension();
        await extension.setup();
        app.extensions.push(extension);
        
        console.log("🚀 WAN Fast ControlNet Processors Extension loaded");
        console.log("💡 Optimized for Depth Anything V2 and DWPose processing");
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const extension = app.extensions.find(ext => ext instanceof WANFastControlNetExtension);
        if (extension) {
            await extension.beforeRegisterNodeDef(nodeType, nodeData, app);
        }
    }
});