import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

console.log("[WAN Mask Editor] JavaScript loading...");

// Helper functions for modal spinner
function showMaskEditorSpinner(message) {
    if (document.getElementById('wan-mask-editor-modal')) return;
    const modal = document.createElement('div');
    modal.id = 'wan-mask-editor-modal';
    modal.style.position = 'fixed';
    modal.style.top = '0';
    modal.style.left = '0';
    modal.style.width = '100vw';
    modal.style.height = '100vh';
    modal.style.background = 'rgba(0,0,0,0.4)';
    modal.style.display = 'flex';
    modal.style.alignItems = 'center';
    modal.style.justifyContent = 'center';
    modal.style.zIndex = '9999';
    modal.innerHTML = `
      <div style="background: #222; color: #fff; padding: 32px 48px; border-radius: 12px; box-shadow: 0 2px 16px #0008; display: flex; flex-direction: column; align-items: center;">
        <div class="wan-mask-spinner" style="margin-bottom: 18px; width: 48px; height: 48px; border: 6px solid #888; border-top: 6px solid #fff; border-radius: 50%; animation: wan-spin 1s linear infinite;"></div>
        <div style="font-size: 1.2em; margin-bottom: 6px;">${message}</div>
        <div style="font-size: 0.95em; color: #ccc;">This may take a while for large videos or image sequences.</div>
      </div>
      <style>
        @keyframes wan-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      </style>
    `;
    document.body.appendChild(modal);
}

function hideMaskEditorSpinner() {
    const modal = document.getElementById('wan-mask-editor-modal');
    if (modal) modal.remove();
}

app.registerExtension({
    name: "WAN.MaskEditor",
    
    async setup() {
        // Hook into the graph to prompt conversion to pass mask data
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function() {
            // Update all mask editor nodes with their data
            for (const node of app.graph._nodes) {
                // Check both type and comfyClass to be safe
                if ((node.type === "WANVaceMaskEditor" || node.comfyClass === "WANVaceMaskEditor") && node.maskData) {
                    console.log(`[WAN Mask Editor] Found mask editor node ${node.id} with maskData`);
                    const maskDataWidget = node.widgets.find(w => w.name === "mask_data");
                    if (maskDataWidget) {
                        console.log(`[WAN Mask Editor] Found mask_data widget`);
                        // Set the mask data for prompt execution
                        const jsonData = JSON.stringify(node.maskData);
                        maskDataWidget.value = jsonData;
                        
                        // Also set internal properties to ensure the value is sent
                        if (maskDataWidget._value !== undefined) {
                            maskDataWidget._value = jsonData;
                        }
                        if (maskDataWidget._internalValue !== undefined) {
                            maskDataWidget._internalValue = jsonData;
                        }
                        
                        console.log(`[WAN Mask Editor] Updated mask editor node ${node.id} with mask data`);
                        console.log(`[WAN Mask Editor] Data keys:`, Object.keys(node.maskData));
                        console.log(`[WAN Mask Editor] Widget value set to:`, jsonData.substring(0, 200) + "...");
                    }
                }
            }
            
            // Call original function
            return originalGraphToPrompt.apply(this, arguments);
        };
    },
    
    async nodeCreated(node) {
        console.log("[WAN Mask Editor] Node created:", node.comfyClass);
        if (node.comfyClass === "WANVaceMaskEditor") {
            console.log("[WAN Mask Editor] Adding widgets to mask editor node");
            // Add button widget
            const buttonWidget = node.addWidget(
                "button",
                "Open Mask Editor",
                "open",
                () => {
                    console.log("[WAN Mask Editor] Button clicked, launching editor...");
                    // Determine descriptive message
                    let message = "Launching Mask Editor…";
                    if (node.maskData) {
                        let info = node.maskData.video_info || node.maskData.source_video;
                        if (info && info.type) {
                            if (info.type === "image_sequence") {
                                message = "Launching Mask Editor for image sequence…";
                            } else if (info.type === "video") {
                                message = "Launching Mask Editor for video file…";
                            }
                        }
                    }
                    // Show spinner modal with message
                    showMaskEditorSpinner(message);
                    // Update status
                    const statusWidget = node.widgets.find(w => w.name === "status");
                    if (statusWidget) {
                        statusWidget.value = "Launching mask editor...";
                    }
                    
                    let requestData = {
                        node_id: node.id
                    };
                    
                    // If we have existing mask data, send the output directory
                    if (node.maskData && node.maskData.output_dir) {
                        requestData.output_dir = node.maskData.output_dir;
                        console.log("[WAN Mask Editor] Sending existing output_dir:", requestData.output_dir);
                    }
                    
                    // Call Python to launch the mask editor
                    api.fetchApi("/wan_vace/launch_mask_editor", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(requestData)
                    })
                    .then(response => response.json())
                    .then(data => {
                        hideMaskEditorSpinner();
                        if (data.success) {
                            console.log("[WAN Mask Editor] Mask editor closed, data:", data);
                            console.log("[WAN Mask Editor] mask_data structure:", data.mask_data);
                            // Store the mask data when editor closes
                            if (data.mask_data) {
                                node.maskData = data.mask_data;
                                // Update the widget
                                const maskDataWidget = node.widgets.find(w => w.name === "mask_data");
                                if (maskDataWidget) {
                                    maskDataWidget.value = JSON.stringify(data.mask_data);
                                    console.log("[WAN Mask Editor] Set widget value to:", maskDataWidget.value.substring(0, 200) + "...");
                                } else {
                                    console.log("[WAN Mask Editor] ERROR: Could not find mask_data widget!");
                                }
                                if (statusWidget) {
                                    // Use backend status if available
                                    if (data.status) {
                                        statusWidget.value = data.status;
                                    } else if (Array.isArray(data.result) && data.result.length >= 3) {
                                        statusWidget.value = data.result[2];
                                    } else {
                                        statusWidget.value = "Mask editor finished.";
                                    }
                                }
                            } else {
                                if (statusWidget) {
                                    statusWidget.value = "Mask editor cancelled";
                                }
                            }
                        } else {
                            console.error("Failed to launch mask editor:", data.error);
                            if (statusWidget) {
                                statusWidget.value = "Error: " + data.error;
                            }
                            alert("Failed to launch mask editor: " + data.error);
                        }
                    })
                    .catch(error => {
                        hideMaskEditorSpinner();
                        console.error("Error launching mask editor:", error);
                        if (statusWidget) {
                            statusWidget.value = "Error: " + error;
                        }
                        alert("Error launching mask editor: " + error);
                    });
                }
            );
            
            // Add status widget
            const statusWidget = node.addWidget(
                "text",
                "status",
                "No mask data",
                () => {},
                { multiline: false }
            );
            statusWidget.disabled = true;
            
            // Store mask data on the node
            node.maskData = null;
            
            // Override serialize to save mask data
            const origSerialize = node.serialize;
            node.serialize = function() {
                const data = origSerialize.apply(this, arguments);
                if (this.maskData) {
                    data.maskData = this.maskData;
                }
                return data;
            };
            
            // Override configure to load mask data
            const origConfigure = node.configure;
            node.configure = function(data) {
                origConfigure.apply(this, arguments);
                if (data.maskData) {
                    this.maskData = data.maskData;
                    const statusWidget = this.widgets.find(w => w.name === "status");
                    if (statusWidget) {
                        statusWidget.value = "Mask data loaded";
                    }
                }
            };
            
            // Override getExtraMenuOptions to add mask data to execution
            const origGetExtraMenuOptions = node.getExtraMenuOptions;
            node.getExtraMenuOptions = function(canvas, options) {
                if (origGetExtraMenuOptions) {
                    origGetExtraMenuOptions.apply(this, arguments);
                }
                
                // Add clear masks option
                options.push({
                    content: "Clear Mask Data",
                    callback: () => {
                        this.maskData = null;
                        const statusWidget = this.widgets.find(w => w.name === "status");
                        if (statusWidget) {
                            statusWidget.value = "No mask data";
                        }
                    }
                });
            };
            
        }
    }
});