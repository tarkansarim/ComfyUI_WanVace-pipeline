import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r
        };
    } else {
        object[property] = callback;
    }
}

function addVideoPreview(nodeType, isInput=true) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        var element = document.createElement("div");
        const previewNode = this;
        var previewWidget = this.addDOMWidget("videopreview", "preview", element, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return element.value;
            },
            setValue(v) {
                element.value = v;
            },
        });
        
        previewWidget.computeSize = function(width) {
            if (this.aspectRatio && !this.parentEl.hidden) {
                let height = (previewNode.size[0]-20)/ this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4];//no loaded src, widget should not display
        }
        
        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "vhs_preview";
        previewWidget.parentEl.style['width'] = "100%"
        element.appendChild(previewWidget.parentEl);
        
        previewWidget.videoEl = document.createElement("video");
        previewWidget.videoEl.controls = false;
        previewWidget.videoEl.loop = true;
        previewWidget.videoEl.muted = true;
        previewWidget.videoEl.style['width'] = "100%"
        previewWidget.videoEl.addEventListener("loadedmetadata", () => {
            previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
            fitHeight(this);
        });
        previewWidget.videoEl.addEventListener("error", () => {
            // If loading fails, hide the widget
            previewWidget.parentEl.hidden = true;
            fitHeight(this);
        });
        
        previewWidget.videoEl.onmouseenter = () => {
            previewWidget.videoEl.muted = previewWidget.value?.muted || false;
        };
        previewWidget.videoEl.onmouseleave = () => {
            previewWidget.videoEl.muted = true;
        };
        
        previewWidget.parentEl.appendChild(previewWidget.videoEl);
        
        // Hide by default until we have content
        previewWidget.parentEl.hidden = true;
        
        // Let ComfyUI/LiteGraph handle widget sizing naturally
        // The widget's computeSize method will be automatically used by LiteGraph
        
        // Handle node configuration/restoration to ensure proper sizing
        chainCallback(this, "onConfigure", function() {
            // Clear any stale video state on node restore
            previewWidget.aspectRatio = undefined;
            previewWidget.parentEl.hidden = true;
            setTimeout(() => {
                fitHeight(this);
            }, 0);
        });
        
        // Initialize node size to ensure compact state on creation/reload
        setTimeout(() => {
            fitHeight(this);
        }, 0);
        
        // Add update method to handle video data
        this.updateVideoPreview = function(params) {
            if (!params) {
                // Clear video content and reset state
                previewWidget.videoEl.src = '';
                previewWidget.aspectRatio = undefined;
                previewWidget.parentEl.hidden = true;
                fitHeight(this);
                return;
            }
            
            // Set the video source
            const url = api.apiURL('/view?' + new URLSearchParams(params));
            previewWidget.videoEl.src = url;
            previewWidget.value = { params: params, hidden: false, paused: false, muted: true };
            
            // Show the preview
            previewWidget.parentEl.hidden = false;
            
            // Auto-play the video
            setTimeout(() => {
                previewWidget.videoEl.play();
            }, 100);
            
            // Trigger resize after showing content (fitHeight will be called again on loadedmetadata)
            fitHeight(this);
        };
    });
}

// Register extension for WANFastVideoCombine
app.registerExtension({
    name: "WANVace.FastVideoCombine",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WANFastVideoCombine") {
            // Add video preview widget
            addVideoPreview(nodeType, false);
            
            // Handle execution messages to update video preview
            chainCallback(nodeType.prototype, "onExecuted", function(message) {
                if (message?.gifs && message.gifs.length > 0) {
                    this.updateVideoPreview(message.gifs[0]);
                }
            });
        }
    }
});