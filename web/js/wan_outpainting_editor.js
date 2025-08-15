import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

console.log("[WAN Outpainting Editor] JavaScript loading...");

// Helper functions for modal spinner
function showOutpaintingSpinner(message) {
    if (document.getElementById('wan-outpainting-modal')) return;
    const modal = document.createElement('div');
    modal.id = 'wan-outpainting-modal';
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
        <div class="wan-outpainting-spinner" style="margin-bottom: 18px; width: 48px; height: 48px; border: 6px solid #888; border-top: 6px solid #fff; border-radius: 50%; animation: wan-spin 1s linear infinite;"></div>
        <div style="font-size: 1.2em; margin-bottom: 6px;">${message}</div>
        <div style="font-size: 0.95em; color: #ccc;">This may take a while for large videos or image sequences.</div>
      </div>
      <style>
        @keyframes wan-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      </style>
    `;
    document.body.appendChild(modal);
}

function hideOutpaintingSpinner() {
    const modal = document.getElementById('wan-outpainting-modal');
    if (modal) modal.remove();
}

app.registerExtension({
    name: "WAN.OutpaintingEditor",
    
    async setup() {
        // Hook into the graph to prompt conversion to pass canvas data
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function() {
            // Update all outpainting editor nodes with their data
            for (const node of app.graph._nodes) {
                // Check both type and comfyClass to be safe
                if ((node.type === "WANVaceOutpaintingEditor" || node.comfyClass === "WANVaceOutpaintingEditor") && node.canvasData) {
                    console.log(`[WAN Outpainting Editor] Found outpainting editor node ${node.id} with canvasData`);
                    const canvasDataWidget = node.widgets.find(w => w.name === "canvas_data");
                    if (canvasDataWidget) {
                        console.log(`[WAN Outpainting Editor] Found canvas_data widget`);
                        // Set the canvas data for prompt execution
                        const jsonData = JSON.stringify(node.canvasData);
                        canvasDataWidget.value = jsonData;
                        
                        // Also set internal properties to ensure the value is sent
                        if (canvasDataWidget._value !== undefined) {
                            canvasDataWidget._value = jsonData;
                        }
                        if (canvasDataWidget._internalValue !== undefined) {
                            canvasDataWidget._internalValue = jsonData;
                        }
                        
                        console.log(`[WAN Outpainting Editor] Updated outpainting editor node ${node.id} with canvas data`);
                        console.log(`[WAN Outpainting Editor] Data keys:`, Object.keys(node.canvasData));
                        console.log(`[WAN Outpainting Editor] Widget value set to:`, jsonData.substring(0, 200) + "...");
                    }
                }
            }
            
            // Call original function
            return originalGraphToPrompt.apply(this, arguments);
        };
    },
    
    async nodeCreated(node) {
        console.log("[WAN Outpainting Editor] Node created:", node.comfyClass);
        if (node.comfyClass === "WANVaceOutpaintingEditor") {
            console.log("[WAN Outpainting Editor] Adding widgets to outpainting editor node");
            // Add button widget
            const buttonWidget = node.addWidget(
                "button",
                "Open Outpainting Editor",
                "open",
                () => {
                    console.log("[WAN Outpainting Editor] Button clicked, launching editor...");
                    // Determine descriptive message
                    let message = "Launching Outpainting Editor…";
                    if (node.canvasData) {
                        let info = node.canvasData.video_info || node.canvasData.source_video;
                        if (info && info.type) {
                            if (info.type === "image_sequence") {
                                message = "Launching Outpainting Editor for image sequence…";
                            } else if (info.type === "video") {
                                message = "Launching Outpainting Editor for video file…";
                            }
                        }
                    }
                    
                    showOutpaintingSpinner(message);
                    
                    // Send canvas data to server
                    const body = {
                        node_id: node.id
                    };
                    
                    // Add existing output directory if we have previous canvas data
                    if (node.canvasData && node.canvasData.output_dir) {
                        body.output_dir = node.canvasData.output_dir;
                        console.log("[WAN Outpainting Editor] Including existing output_dir:", body.output_dir);
                    }
                    
                    fetch("/wan_vace/launch_outpainting_editor", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify(body),
                    })
                    .then(response => response.json())
                    .then(data => {
                        hideOutpaintingSpinner();
                        console.log("[WAN Outpainting Editor] Server response:", data);
                        if (data.success && data.canvas_data) {
                            // Parse the canvas data
                            const canvasData = JSON.parse(data.canvas_data);
                            console.log("[WAN Outpainting Editor] Parsed canvas data:", canvasData);
                            
                            // Store the canvas data on the node
                            node.canvasData = canvasData;
                            
                            // Update the widget value
                            const canvasDataWidget = node.widgets.find(w => w.name === "canvas_data");
                            if (canvasDataWidget) {
                                canvasDataWidget.value = data.canvas_data;
                                console.log("[WAN Outpainting Editor] Updated canvas_data widget");
                            }
                            
                            // Update button text to indicate data is loaded
                            buttonWidget.name = "Edit Canvas Settings";
                            
                            // Mark node as needing update
                            node.setDirtyCanvas(true);
                            
                            console.log("[WAN Outpainting Editor] Canvas data loaded successfully");
                        } else {
                            console.error("[WAN Outpainting Editor] Failed to load canvas data:", data.error);
                        }
                    })
                    .catch(error => {
                        hideOutpaintingSpinner();
                        console.error("[WAN Outpainting Editor] Error launching outpainting editor:", error);
                        alert("Failed to launch outpainting editor: " + error);
                    });
                }
            );
            
            // Hide the canvas_data widget as it's for internal use
            const canvasDataWidget = node.widgets.find(w => w.name === "canvas_data");
            if (canvasDataWidget) {
                canvasDataWidget.widget.hidden = true;
                console.log("[WAN Outpainting Editor] Hidden canvas_data widget");
            }
            
            // Override serialize to include canvas data
            const origSerialize = node.serialize;
            node.serialize = function() {
                const data = origSerialize ? origSerialize.apply(this, arguments) : {};
                if (this.canvasData) {
                    if (!data.widgets_values) data.widgets_values = [];
                    // Find the canvas_data widget index
                    const canvasDataIndex = this.widgets.findIndex(w => w.name === "canvas_data");
                    if (canvasDataIndex >= 0) {
                        data.widgets_values[canvasDataIndex] = JSON.stringify(this.canvasData);
                    }
                }
                return data;
            };
            
            // Override configure to restore canvas data
            const origConfigure = node.configure;
            node.configure = function(data) {
                if (origConfigure) {
                    origConfigure.apply(this, arguments);
                }
                // Restore canvas data if available
                if (data.widgets_values) {
                    const canvasDataIndex = this.widgets.findIndex(w => w.name === "canvas_data");
                    if (canvasDataIndex >= 0 && data.widgets_values[canvasDataIndex]) {
                        try {
                            this.canvasData = JSON.parse(data.widgets_values[canvasDataIndex]);
                            buttonWidget.name = "Edit Canvas Settings";
                            console.log("[WAN Outpainting Editor] Restored canvas data from save");
                        } catch (e) {
                            console.error("[WAN Outpainting Editor] Failed to parse canvas data:", e);
                        }
                    }
                }
            };
        }
    }
});