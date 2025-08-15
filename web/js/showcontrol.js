import { app } from "../../../../scripts/app.js";

// Some fragments of this code are from https://github.com/LucianoCirino/efficiency-nodes-comfyui

function wanCropAndStitchHandler(node) {
    console.log(`[ShowControl] Handler called for: ${node.comfyClass}`);
    
    if (node.comfyClass == "WanCropImproved") {
        console.log(`[ShowControl] Handling WanCropImproved node`);
        
        // ComfyUI automatically handles dynamic inputs for optional inputs
        // No additional JavaScript needed for input visibility
        
        toggleWidget(node, findWidgetByName(node, "preresize_mode"));
        toggleWidget(node, findWidgetByName(node, "preresize_min_width"));
        toggleWidget(node, findWidgetByName(node, "preresize_min_height"));
        toggleWidget(node, findWidgetByName(node, "preresize_max_width"));
        toggleWidget(node, findWidgetByName(node, "preresize_max_height"));
        if (findWidgetByName(node, "preresize").value == true) {
            toggleWidget(node, findWidgetByName(node, "preresize_mode"), true);
            if (findWidgetByName(node, "preresize_mode").value == "ensure minimum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_min_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_min_height"), true);
            }
            else if (findWidgetByName(node, "preresize_mode").value == "ensure minimum and maximum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_min_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_min_height"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_height"), true);
            }
            else if (findWidgetByName(node, "preresize_mode").value == "ensure maximum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_max_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_height"), true);
            }
        }
        toggleWidget(node, findWidgetByName(node, "extend_up_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_down_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_left_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_right_factor"));
        if (findWidgetByName(node, "extend_for_outpainting").value == true) {
            toggleWidget(node, findWidgetByName(node, "extend_up_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_down_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_left_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_right_factor"), true);
        }
        toggleWidget(node, findWidgetByName(node, "output_target_width"));
        toggleWidget(node, findWidgetByName(node, "output_target_height"));
        if (findWidgetByName(node, "output_resize_to_target_size").value == true) {
            toggleWidget(node, findWidgetByName(node, "output_target_width"), true);
            toggleWidget(node, findWidgetByName(node, "output_target_height"), true);
        }
    }

    // Handle WanCropVideo node
    if (node.comfyClass == "WanCropVideo") {
        toggleWidget(node, findWidgetByName(node, "preresize_mode"));
        toggleWidget(node, findWidgetByName(node, "preresize_min_width"));
        toggleWidget(node, findWidgetByName(node, "preresize_min_height"));
        toggleWidget(node, findWidgetByName(node, "preresize_max_width"));
        toggleWidget(node, findWidgetByName(node, "preresize_max_height"));
        if (findWidgetByName(node, "preresize").value == true) {
            toggleWidget(node, findWidgetByName(node, "preresize_mode"), true);
            if (findWidgetByName(node, "preresize_mode").value == "ensure minimum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_min_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_min_height"), true);
            }
            else if (findWidgetByName(node, "preresize_mode").value == "ensure minimum and maximum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_min_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_min_height"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_height"), true);
            }
            else if (findWidgetByName(node, "preresize_mode").value == "ensure maximum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_max_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_height"), true);
            }
        }
        toggleWidget(node, findWidgetByName(node, "extend_up_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_down_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_left_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_right_factor"));
        if (findWidgetByName(node, "extend_for_outpainting").value == true) {
            toggleWidget(node, findWidgetByName(node, "extend_up_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_down_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_left_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_right_factor"), true);
        }
        toggleWidget(node, findWidgetByName(node, "output_width"));
        toggleWidget(node, findWidgetByName(node, "output_height"));
        if (findWidgetByName(node, "output_resize").value == true) {
            toggleWidget(node, findWidgetByName(node, "output_width"), true);
            toggleWidget(node, findWidgetByName(node, "output_height"), true);
        }
    }

    // Handle legacy InpaintCropImproved for backward compatibility
    if (node.comfyClass == "InpaintCropImproved") {
        toggleWidget(node, findWidgetByName(node, "preresize_mode"));
        toggleWidget(node, findWidgetByName(node, "preresize_min_width"));
        toggleWidget(node, findWidgetByName(node, "preresize_min_height"));
        toggleWidget(node, findWidgetByName(node, "preresize_max_width"));
        toggleWidget(node, findWidgetByName(node, "preresize_max_height"));
        if (findWidgetByName(node, "preresize").value == true) {
            toggleWidget(node, findWidgetByName(node, "preresize_mode"), true);
            if (findWidgetByName(node, "preresize_mode").value == "ensure minimum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_min_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_min_height"), true);
            }
            else if (findWidgetByName(node, "preresize_mode").value == "ensure minimum and maximum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_min_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_min_height"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_height"), true);
            }
            else if (findWidgetByName(node, "preresize_mode").value == "ensure maximum resolution") {
                toggleWidget(node, findWidgetByName(node, "preresize_max_width"), true);
                toggleWidget(node, findWidgetByName(node, "preresize_max_height"), true);
            }
        }
        toggleWidget(node, findWidgetByName(node, "extend_up_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_down_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_left_factor"));
        toggleWidget(node, findWidgetByName(node, "extend_right_factor"));
        if (findWidgetByName(node, "extend_for_outpainting").value == true) {
            toggleWidget(node, findWidgetByName(node, "extend_up_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_down_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_left_factor"), true);
            toggleWidget(node, findWidgetByName(node, "extend_right_factor"), true);
        }
        toggleWidget(node, findWidgetByName(node, "output_target_width"));
        toggleWidget(node, findWidgetByName(node, "output_target_height"));
        if (findWidgetByName(node, "output_resize_to_target_size").value == true) {
            toggleWidget(node, findWidgetByName(node, "output_target_width"), true);
            toggleWidget(node, findWidgetByName(node, "output_target_height"), true);
        }
    }

    // OLD - Keep for backward compatibility
    if (node.comfyClass == "InpaintCrop") {
        toggleWidget(node, findWidgetByName(node, "force_width"));
        toggleWidget(node, findWidgetByName(node, "force_height"));
        toggleWidget(node, findWidgetByName(node, "rescale_factor"));
        toggleWidget(node, findWidgetByName(node, "min_width"));
        toggleWidget(node, findWidgetByName(node, "min_height"));
        toggleWidget(node, findWidgetByName(node, "max_width"));
        toggleWidget(node, findWidgetByName(node, "max_height"));
        toggleWidget(node, findWidgetByName(node, "padding"));
        if (findWidgetByName(node, "mode").value == "free size") {
            toggleWidget(node, findWidgetByName(node, "rescale_factor"), true);
            toggleWidget(node, findWidgetByName(node, "padding"), true);
        }
        else if (findWidgetByName(node, "mode").value == "ranged size") {
            toggleWidget(node, findWidgetByName(node, "min_width"), true);
            toggleWidget(node, findWidgetByName(node, "min_height"), true);
            toggleWidget(node, findWidgetByName(node, "max_width"), true);
            toggleWidget(node, findWidgetByName(node, "max_height"), true);
            toggleWidget(node, findWidgetByName(node, "padding"), true);
        }
        else if (findWidgetByName(node, "mode").value == "forced size") {
            toggleWidget(node, findWidgetByName(node, "force_width"), true);
            toggleWidget(node, findWidgetByName(node, "force_height"), true);
        }
    } else if (node.comfyClass == "InpaintExtendOutpaint") {
        toggleWidget(node, findWidgetByName(node, "expand_up_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_up_factor"));
        toggleWidget(node, findWidgetByName(node, "expand_down_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_down_factor"));
        toggleWidget(node, findWidgetByName(node, "expand_left_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_left_factor"));
        toggleWidget(node, findWidgetByName(node, "expand_right_pixels"));
        toggleWidget(node, findWidgetByName(node, "expand_right_factor"));
        if (findWidgetByName(node, "mode").value == "factors") {
            toggleWidget(node, findWidgetByName(node, "expand_up_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_down_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_left_factor"), true);
            toggleWidget(node, findWidgetByName(node, "expand_right_factor"), true);
        }
        if (findWidgetByName(node, "mode").value == "pixels") {
            toggleWidget(node, findWidgetByName(node, "expand_up_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_down_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_left_pixels"), true);
            toggleWidget(node, findWidgetByName(node, "expand_right_pixels"), true);
        }
    } else if (node.comfyClass == "InpaintResize") {
        toggleWidget(node, findWidgetByName(node, "min_width"));
        toggleWidget(node, findWidgetByName(node, "min_height"));
        toggleWidget(node, findWidgetByName(node, "rescale_factor"));
        if (findWidgetByName(node, "mode").value == "ensure minimum size") {
            toggleWidget(node, findWidgetByName(node, "min_width"), true);
            toggleWidget(node, findWidgetByName(node, "min_height"), true);
        }
        else if (findWidgetByName(node, "mode").value == "factor") {
            toggleWidget(node, findWidgetByName(node, "rescale_factor"), true);
        }
    }
    return;
}

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

// Toggle Widget + change size
function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget) return;
    widget.disabled = !show;
    
    // Add visual styling for grayed out widgets
    if (widget.element) {
        widget.element.style.opacity = show ? "1.0" : "0.5";
        widget.element.style.pointerEvents = show ? "auto" : "none";
    }
    
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));
}   

function dumpLiteGraphConstants(LG) {
    try {
        return {
            ON_TRIGGER: LG?.ON_TRIGGER,
            NEVER: LG?.NEVER,
            ALWAYS: LG?.ALWAYS,
            PASS_THROUGH: LG?.PASS_THROUGH,
        };
    } catch (_) { return {}; }
}

function dumpNodeState(node, label = "STATE") {
    try {
        const LG = globalThis.LiteGraph || window.LiteGraph;
        const constants = dumpLiteGraphConstants(LG);
        const flags = node.flags || {};
        const inputs = (node.inputs || []).map((inp, idx) => ({ idx, name: inp?.name, type: inp?.type, link: inp?.link }));
        const outputs = (node.outputs || []).map((out, idx) => ({ idx, name: out?.name, type: out?.type, links: out?.links }));
        console.log(`[Dynamic][${label}] ${node.comfyClass} mode=${node.mode} constants=${JSON.stringify(constants)} flags=${JSON.stringify(flags)}`);
        console.log(`[Dynamic][${label}] inputs(${inputs.length}):`, inputs);
        console.log(`[Dynamic][${label}] outputs(${outputs.length}):`, outputs);
    } catch (e) {
        console.warn(`[Dynamic][${label}] dump error:`, e);
    }
}

// Register dynamic inputs extension using proper LiteGraph API
app.registerExtension({
    name: "wanvace-cropandstitch.dynamic",
    nodeCreated(node) {
        if (node.comfyClass === "WanCropImproved") {
            console.log(`[Dynamic] Setting up dynamic inputs for WanCropImproved`);
            
            const numInputsWidget = node.widgets?.find(w => w.name === "num_inputs");
            if (!numInputsWidget) {
                console.log(`[Dynamic] No num_inputs widget found`);
                return;
            }
            
            const updateInputsOutputs = () => {
                const numInputs = numInputsWidget.value;
                console.log(`[Dynamic] Synchronizing node for num_inputs: ${numInputs}`);
                const LG = globalThis.LiteGraph || window.LiteGraph;
                const detectBypass = () => {
                    const flags = node.flags || {};
                    const modes = { always: LG?.ALWAYS ?? 0, never: LG?.NEVER ?? 2, onTrigger: LG?.ON_TRIGGER ?? 3, passThrough: LG?.PASS_THROUGH };
                    const knownModes = new Set([modes.always, modes.never, modes.onTrigger]);
                    const isWeirdMode = (typeof node.mode !== 'undefined') && !knownModes.has(node.mode);
                    const passThrough = (typeof modes.passThrough !== 'undefined') && node.mode === modes.passThrough;
                    const anyFlag = !!(flags.bypass || flags.bypassed || flags.skip_processing || flags.skip_process || flags.muted || flags.pass_through);
                    const result = !!(passThrough || anyFlag || isWeirdMode);
                    console.log(`[Dynamic] detectBypass: mode=${node.mode}, knownModes=${JSON.stringify([...knownModes])}, passThroughConst=${modes.passThrough}, isWeirdMode=${isWeirdMode}, flags=${JSON.stringify(flags)}, result=${result}`);
                    return result;
                };
                const isBypassed = detectBypass();
                console.log(`[Dynamic] Node bypass state: ${isBypassed} (mode=${node.mode}, flags=${JSON.stringify(node.flags || {})})`);
                dumpNodeState(node, isBypassed ? "BEFORE-BYPASS" : "BEFORE-NORMAL");
                
                // Calculate expected state
                let expectedInputs = [];
                if (isBypassed) {
                    // Align inputs to outputs count by inserting dummy passthrough stitcher inputs per triplet
                    for (let i = 1; i <= numInputs; i++) {
                        const imageName = i === 1 ? "image" : `image_${i}`;
                        const maskName = i === 1 ? "mask" : `mask_${i}`;
                        expectedInputs.push({name: imageName, type: "IMAGE"});
                        expectedInputs.push({name: maskName, type: "MASK"});
                        expectedInputs.push({name: `pt_stitcher_${i}`, type: "STITCHER"});
                    }
                } else {
                    // Normal: only image/mask inputs
                    expectedInputs.push({name: "image", type: "IMAGE"});
                    expectedInputs.push({name: "mask", type: "MASK"});
                    for (let i = 2; i <= numInputs; i++) {
                        expectedInputs.push({name: `image_${i}`, type: "IMAGE"});
                        expectedInputs.push({name: `mask_${i}`, type: "MASK"});
                    }
                }
                
                const expectedOutputs = [];
                // Always interleaved triplets; do not remove on bypass to preserve links
                for (let i = 1; i <= numInputs; i++) {
                    expectedOutputs.push({name: `cropped_image_${i}`, type: "IMAGE"});
                    expectedOutputs.push({name: `cropped_mask_${i}`, type: "MASK"});
                    expectedOutputs.push({name: `stitcher_${i}`, type: "STITCHER"});
                }
                
                console.log(`[Dynamic] Target state: ${expectedInputs.length} inputs, ${expectedOutputs.length} outputs`);
                console.log(`[Dynamic] Current state: ${node.inputs?.length || 0} inputs, ${node.outputs?.length || 0} outputs`);
                
                // Ensure inputs/outputs arrays exist
                if (!node.inputs) node.inputs = [];
                if (!node.outputs) node.outputs = [];
                
                // Sync inputs - only modify what's different
                const currentInputNames = node.inputs.map(input => input.name);
                const expectedInputNames = expectedInputs.map(input => input.name);
                
                {
                    // Non-destructive input sync to preserve links across bypass toggles
                    for (let i = node.inputs.length - 1; i >= 0; i--) {
                        const inputName = node.inputs[i].name;
                        if (!expectedInputNames.includes(inputName) && (inputName.match(/^(image_|mask_)[2-5]$/) || inputName.match(/^pt_stitcher_[1-5]$/))) {
                            console.log(`[Dynamic] Removing excess input: ${inputName}`);
                            node.removeInput(i);
                        }
                    }
                    for (const expectedInput of expectedInputs) {
                        const exists = node.inputs.find(input => input.name === expectedInput.name);
                        if (!exists) {
                            console.log(`[Dynamic] Adding missing input: ${expectedInput.name}`);
                            node.addInput(expectedInput.name, expectedInput.type);
                        }
                    }
                }
                
                // Outputs: non-destructive update to preserve existing links
                const graph = app.graph;
                for (const expectedOutput of expectedOutputs) {
                    const exists = node.outputs.find(o => o.name === expectedOutput.name);
                    if (!exists) {
                        console.log(`[Dynamic] Adding output: ${expectedOutput.name}`);
                        node.addOutput(expectedOutput.name, expectedOutput.type);
                    }
                }
                // Optionally, when not bypassed, prune excess managed outputs beyond num_inputs
                if (!isBypassed) {
                    const expectedOutputNames = expectedOutputs.map(o => o.name);
                    const managedTripletPattern = /^(cropped_(image|mask)_[1-5]|stitcher_[1-5])$/;
                    for (let i = node.outputs.length - 1; i >= 0; i--) {
                        const out = node.outputs[i];
                        if (managedTripletPattern.test(out.name) && !expectedOutputNames.includes(out.name)) {
                            console.log(`[Dynamic] Removing excess output: ${out.name}`);
                            node.removeOutput(i);
                        }
                    }
                }
                dumpNodeState(node, "AFTER-SYNC");
                
                // Refresh node layout and canvas
                try {
                    node.setSize(node.computeSize());
                    if (app.graph) {
                        app.graph.setDirtyCanvas(true);
                    }
                } catch (e) {
                    console.warn(`[Dynamic] Layout refresh error (non-critical):`, e);
                }
                
                console.log(`[Dynamic] Sync complete - node has ${node.inputs.length} inputs, ${node.outputs.length} outputs for num_inputs=${numInputs}`);
            };
            
            // Set callback with proper error handling and context preservation
            const originalCallback = numInputsWidget.callback;
            numInputsWidget.callback = function(value) {
                console.log(`[Dynamic] num_inputs widget changed to: ${value}`);
                
                // Call original callback first to maintain widget state
                if (originalCallback) {
                    try {
                        originalCallback.call(this, value);
                    } catch (e) {
                        console.warn(`[Dynamic] Original callback error (continuing):`, e);
                    }
                }
                
                // Update node structure to match new value
                try {
                    updateInputsOutputs();
                } catch (e) {
                    console.error(`[Dynamic] Error updating inputs/outputs:`, e);
                }
            };
            
            // Poll for bypass mode changes and resync
            node._wan_last_mode = node.mode;
            node._wan_last_bypass = false;
            const modePoll = setInterval(() => {
                try {
                    if (!node || !node.graph) { clearInterval(modePoll); return; }
                    const LG = globalThis.LiteGraph || window.LiteGraph;
                    const flags = node.flags || {};
                    const passThrough = LG && typeof LG.PASS_THROUGH !== 'undefined' && node.mode === LG.PASS_THROUGH;
                    const anyFlag = !!(flags.bypass || flags.bypassed || flags.skip_processing || flags.skip_process || flags.muted || flags.pass_through);
                    const isBypassedNow = passThrough || anyFlag;

                    if (node._wan_last_mode !== node.mode || node._wan_last_bypass !== isBypassedNow) {
                        const wasBypassed = node._wan_last_bypass === true;
                        node._wan_last_mode = node.mode;
                        node._wan_last_bypass = isBypassedNow;
                        console.log(`[Dynamic] Detected mode change -> resync outputs`);
                        // If leaving bypass, cleanup temporary direct links
                        if (wasBypassed && !isBypassedNow && node._wan_created_links && app.graph) {
                            try {
                                for (const id of node._wan_created_links) {
                                    try { app.graph.removeLink(id); } catch (_) {}
                                }
                                node._wan_created_links = [];
                            } catch (e) { console.warn('[Dynamic] Cleanup created links failed:', e); }
                        }
                        updateInputsOutputs();
                    }
                } catch (_) { /* ignore */ }
            }, 400);

            // Initial synchronization - respect current widget value (no forced cleanup)
            // This handles both new node creation and workflow reload correctly
            setTimeout(() => {
                try {
                    if (!node.graph) {
                        console.log(`[Dynamic] Node not yet connected to graph, will sync when ready`);
                        return;
                    }
                    
                    const currentValue = numInputsWidget.value;
                    console.log(`[Dynamic] Initial sync - respecting current num_inputs value: ${currentValue}`);
                    
                    // Sync to current widget value (could be 1 for new nodes, or 3+ for workflow reload)
                    updateInputsOutputs();
                    
                    console.log(`[Dynamic] Initial sync complete - node ready for num_inputs=${currentValue}`);
                } catch (e) {
                    console.error(`[Dynamic] Error in initial sync:`, e);
                }
            }, 50); // Minimal delay just to ensure graph connection
        }
        if (node.comfyClass === "WanStitchImproved") {
            console.log(`[Dynamic] Setting up dynamic inputs for WanStitchImproved`);
            const LG = globalThis.LiteGraph || window.LiteGraph;
            const detectBypass = () => {
                const flags = node.flags || {};
                const modes = { always: LG?.ALWAYS ?? 0, never: LG?.NEVER ?? 2, onTrigger: LG?.ON_TRIGGER ?? 3, passThrough: LG?.PASS_THROUGH };
                const knownModes = new Set([modes.always, modes.never, modes.onTrigger]);
                const isWeirdMode = (typeof node.mode !== 'undefined') && !knownModes.has(node.mode);
                const passThrough = (typeof modes.passThrough !== 'undefined') && node.mode === modes.passThrough;
                const anyFlag = !!(flags.bypass || flags.bypassed || flags.skip_processing || flags.skip_process || flags.muted || flags.pass_through);
                const result = !!(passThrough || anyFlag || isWeirdMode);
                console.log(`[Dynamic][Stitcher] detectBypass: mode=${node.mode}, knownModes=${JSON.stringify([...knownModes])}, passThroughConst=${modes.passThrough}, isWeirdMode=${isWeirdMode}, flags=${JSON.stringify(flags)}, result=${result}`);
                return result;
            };

            const updateStitcher = () => {
                const isBypassed = detectBypass();
                console.log(`[Dynamic][Stitcher] Bypass=${isBypassed}, inputs=${node.inputs?.map(i=>i.name).join(',')}`);
                if (!node.inputs) node.inputs = [];

                const stitcherIdx = node.inputs.findIndex(inp => inp.name === 'stitcher');
                const imageIdx = node.inputs.findIndex(inp => inp.name === 'inpainted_image');

                // Do nothing to connections or inputs on bypass; keep structure stable to preserve links
                // Ensure required inputs exist (non-destructive)
                if (stitcherIdx === -1) {
                    try { node.addInput('stitcher', 'STITCHER'); } catch (_) {}
                }
            };

            node._wan_last_mode = node.mode;
            node._wan_last_bypass = false;
            const modePoll = setInterval(() => {
                try {
                    if (!node || !node.graph) { clearInterval(modePoll); return; }
                    const isBypassedNow = detectBypass();
                    if (node._wan_last_mode !== node.mode || node._wan_last_bypass !== isBypassedNow) {
                        node._wan_last_mode = node.mode;
                        node._wan_last_bypass = isBypassedNow;
                        console.log(`[Dynamic][Stitcher] Mode change -> update`);
                        updateStitcher();
                    }
                } catch (_) {}
            }, 400);

            setTimeout(() => {
                try { updateStitcher(); } catch (e) { console.warn('[Dynamic][Stitcher] Initial update error', e); }
            }, 50);
        }
    }
});

// Register main showcontrol extension
app.registerExtension({
    name: "wanvace-cropandstitch.showcontrol",
    nodeCreated(node) {
        console.log(`[ShowControl] Node created: ${node.comfyClass}`);
        
        if (!node.comfyClass.startsWith("Wan") && !node.comfyClass.startsWith("Inpaint")) {
            console.log(`[ShowControl] Skipping node: ${node.comfyClass} (not Wan/Inpaint)`);
            return;
        }

        console.log(`[ShowControl] Processing node: ${node.comfyClass}`);
        wanCropAndStitchHandler(node);
        for (const w of node.widgets || []) {
            let widgetValue = w.value;

            // Store the original descriptor if it exists 
            let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value') || 
                Object.getOwnPropertyDescriptor(Object.getPrototypeOf(w), 'value');
            if (!originalDescriptor) {
                originalDescriptor = Object.getOwnPropertyDescriptor(w.constructor.prototype, 'value');
            }

            Object.defineProperty(w, 'value', {
                get() {
                    // If there's an original getter, use it. Otherwise, return widgetValue.
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;

                    return valueToReturn;
                },
                set(newVal) {
                    // If there's an original setter, use it. Otherwise, set widgetValue.
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else { 
                        widgetValue = newVal;
                    }

                    wanCropAndStitchHandler(node);
                }
            });
        }
    }
}); 