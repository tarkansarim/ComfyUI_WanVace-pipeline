"""
Node registration for WAN Vace Pipeline
This file imports nodes from separate modules and registers them with ComfyUI
"""

import os
print(f"[WAN Vace Pipeline] Loading node_mappings.py from: {os.path.abspath(__file__)}")

# Configuration flag to control which nodes are enabled
# Set to True to enable experimental/development nodes
ENABLE_EXPERIMENTAL_NODES = False

# Import all nodes from the nodes package
from .nodes import (
    WANSaveVideo,
    WANLoadVideo,
    WANVaceSplitReferenceVideo,
    WANVaceJoinVideos,
    WANVaceVideoExtension,
    WANVaceFrameInterpolation,
    WANVaceKeyframeTimeline,
    WANVaceFrameSampler,
    WANVaceFrameInjector,
    WANVaceOutpainting,
    WANVaceBatchStartIndex,
    WANFastImageBatchProcessor,
    WANFastImageCompositeMasked,
    WANFastImageBlend,
    WANFastImageScaleBy,
    WANFastImageScaleToMegapixels,
    WANFastImageResize,
    WANFastDepthAnythingV2,
    WANFastDWPose,
    WANFastVideoEncode,
    WANFastVACEEncode,
    WANFastVideoCombine
)

# Import VACE Loop Encoder node
try:
    from .nodes.vace_loop_encoder import WANVACELoopEncoder
    print("Successfully imported WANVACELoopEncoder node")
except ImportError as e:
    print(f"WARNING: Failed to import WANVACELoopEncoder node: {e}")
    WANVACELoopEncoder = None

# Try to import mask viewer node separately
try:
    from .nodes.mask_viewer import WANVaceMaskViewer
    print("Successfully imported mask viewer node")
except ImportError as e:
    print(f"WARNING: Failed to import mask viewer node: {e}")
    WANVaceMaskViewer = None

# Try test node
try:
    from .nodes.test_mask_node import WANVaceTestMask
    print("Successfully imported test mask node")
except ImportError as e:
    print(f"WARNING: Failed to import test mask node: {e}")
    WANVaceTestMask = None

# Import WAN Inpaint Conditioning node
try:
    from .wan_inpaint_conditioning import WANInpaintConditioning
    print("Successfully imported WANInpaintConditioning node")
except ImportError as e:
    print(f"WARNING: Failed to import WANInpaintConditioning node: {e}")
    WANInpaintConditioning = None

# Import WAN Video Sampler Inpaint node
try:
    from .wan_video_sampler_inpaint import WANVideoSamplerInpaint
    print("Successfully imported WANVideoSamplerInpaint node")
except ImportError as e:
    print(f"WARNING: Failed to import WANVideoSamplerInpaint node: {e}")
    WANVideoSamplerInpaint = None

# Import WAN Tiled Sampler node
try:
    from .nodes.wan_tiled_sampler import WANTiledSampler
    print("Successfully imported WANTiledSampler node")
except ImportError as e:
    print(f"WARNING: Failed to import WANTiledSampler node: {e}")
    WANTiledSampler = None

# Import WAN Match Batch Size node
try:
    from .nodes.wan_match_batch_size import WANMatchBatchSize
    print("Successfully imported WANMatchBatchSize node")
except ImportError as e:
    print(f"WARNING: Failed to import WANMatchBatchSize node: {e}")
    WANMatchBatchSize = None

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # I/O Nodes
    "WANLoadVideo": WANLoadVideo,
    "WANSaveVideo": WANSaveVideo,
    
    # Processing Nodes
    # "WANVaceSplitReferenceVideo": WANVaceSplitReferenceVideo,  # Disabled for release
    "WANVaceJoinVideos": WANVaceJoinVideos,
    "WANVaceVideoExtension": WANVaceVideoExtension,
    "WANVaceFrameInterpolation": WANVaceFrameInterpolation,
    
    # Timeline Nodes
    "WANVaceKeyframeTimeline": WANVaceKeyframeTimeline,
    
    # Frame Utility Nodes
    # "WANVaceFrameSampler": WANVaceFrameSampler,  # Disabled for release
    "WANVaceFrameInjector": WANVaceFrameInjector,
    
    # Effects Nodes
    "WANVaceOutpainting": WANVaceOutpainting,
    # "WANVaceBatchStartIndex": WANVaceBatchStartIndex,  # Disabled for release
    
    # Fast Processing Nodes
    "WANFastImageBatchProcessor": WANFastImageBatchProcessor,
    # "WANFastImageCompositeMasked": WANFastImageCompositeMasked,  # Disabled for release
    # "WANFastImageBlend": WANFastImageBlend,  # Disabled for release
    # "WANFastImageScaleBy": WANFastImageScaleBy,  # Disabled for release
    # "WANFastImageScaleToMegapixels": WANFastImageScaleToMegapixels,  # Disabled for release
    # "WANFastImageResize": WANFastImageResize,  # Disabled for release
    
    # Fast ControlNet Processors
    "WANFastDepthAnythingV2": WANFastDepthAnythingV2,
    "WANFastDWPose": WANFastDWPose,
    
    # Fast Video Processors
    # "WANFastVideoEncode": WANFastVideoEncode,  # Disabled for release
    # "WANFastVACEEncode": WANFastVACEEncode,  # Disabled for release
    # "WANFastVideoCombine": WANFastVideoCombine  # Disabled for release
}

# Add disabled nodes if experimental mode is enabled
if ENABLE_EXPERIMENTAL_NODES:
    NODE_CLASS_MAPPINGS.update({
        # Processing Nodes
        "WANVaceSplitReferenceVideo": WANVaceSplitReferenceVideo,
        
        # Frame Utility Nodes
        "WANVaceFrameSampler": WANVaceFrameSampler,
        
        # Effects Nodes
        "WANVaceBatchStartIndex": WANVaceBatchStartIndex,
        
        # Fast Processing Nodes
        "WANFastImageCompositeMasked": WANFastImageCompositeMasked,
        "WANFastImageBlend": WANFastImageBlend,
        "WANFastImageScaleBy": WANFastImageScaleBy,
        "WANFastImageScaleToMegapixels": WANFastImageScaleToMegapixels,
        "WANFastImageResize": WANFastImageResize,
        
        # Fast Video Processors
        "WANFastVideoEncode": WANFastVideoEncode,
        "WANFastVACEEncode": WANFastVACEEncode,
        "WANFastVideoCombine": WANFastVideoCombine
    })

# Add VACE Loop Encoder node if successfully imported
if WANVACELoopEncoder is not None and ENABLE_EXPERIMENTAL_NODES:
    NODE_CLASS_MAPPINGS["WANVACELoopEncoder"] = WANVACELoopEncoder

# Add mask nodes if they were successfully imported
# WANVaceMaskEditor removed - using WANVaceMaskEditorDirect instead
if WANVaceMaskViewer is not None:
    NODE_CLASS_MAPPINGS["WANVaceMaskViewer"] = WANVaceMaskViewer
if WANVaceTestMask is not None:
    NODE_CLASS_MAPPINGS["WANVaceTestMask"] = WANVaceTestMask

# Add WAN Inpaint Conditioning node if successfully imported
if WANInpaintConditioning is not None:
    NODE_CLASS_MAPPINGS["WANInpaintConditioning"] = WANInpaintConditioning

# Add WAN Video Sampler Inpaint node if successfully imported
if WANVideoSamplerInpaint is not None:
    NODE_CLASS_MAPPINGS["WANVideoSamplerInpaint"] = WANVideoSamplerInpaint

# Add WAN Tiled Sampler node if successfully imported
if WANTiledSampler is not None:
    NODE_CLASS_MAPPINGS["WANTiledSampler"] = WANTiledSampler

# Add WAN Match Batch Size node if successfully imported
if WANMatchBatchSize is not None:
    NODE_CLASS_MAPPINGS["WANMatchBatchSize"] = WANMatchBatchSize

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    # I/O Nodes
    "WANLoadVideo": "WanVace-pipeline Load Video 🎬",
    "WANSaveVideo": "WanVace-pipeline Save Video 💾",
    
    # Processing Nodes
    # "WANVaceSplitReferenceVideo": "WanVace-pipeline Split Video Batch ✂️",  # Disabled for release
    "WANVaceJoinVideos": "WanVace-pipeline Join Videos 🔗",
    "WANVaceVideoExtension": "WanVace-pipeline Video Extension 🔄",
    "WANVaceFrameInterpolation": "WanVace-pipeline Frame Interpolator 🎞️",
    
    # Timeline Nodes
    "WANVaceKeyframeTimeline": "WanVace-pipeline Keyframe Timeline 📽️",
    
    # Frame Utility Nodes
    # "WANVaceFrameSampler": "WanVace-pipeline Frame Sampler 📊",  # Disabled for release
    "WANVaceFrameInjector": "WanVace-pipeline Frame Injector 💉",
    
    # Effects Nodes
    "WANVaceOutpainting": "WanVace-pipeline Outpainting Prep 🖼️",
    # "WANVaceBatchStartIndex": "WanVace-pipeline Batch Start Index 🔢",  # Disabled for release
    
    # Fast Processing Nodes
    "WANFastImageBatchProcessor": "WanVace-pipeline Fast Image Batch Processor 🚀",
    # "WANFastImageCompositeMasked": "WanVace-pipeline Fast Image Composite Masked 🚀",  # Disabled for release
    # "WANFastImageBlend": "WanVace-pipeline Fast Image Blend 🚀",  # Disabled for release
    # "WANFastImageScaleBy": "WanVace-pipeline Fast Image Scale By 🚀",  # Disabled for release
    # "WANFastImageScaleToMegapixels": "WanVace-pipeline Fast Image Scale To Megapixels 🚀",  # Disabled for release
    # "WANFastImageResize": "WanVace-pipeline Fast Image Resize 🚀",  # Disabled for release
    
    # Fast ControlNet Processors
    "WANFastDepthAnythingV2": "WanVace-pipeline Fast Depth Anything V2 🚀",
    "WANFastDWPose": "WanVace-pipeline Fast DWPose Estimator 🚀",
    
    # Fast Video Processors
    # "WANFastVideoEncode": "WanVace-pipeline Fast Video Encode 🚀",  # Disabled for release
    # "WANFastVACEEncode": "WanVace-pipeline Fast VACE Encode 🚀",  # Disabled for release
    # "WANFastVideoCombine": "WanVace-pipeline Fast Video Combine 🚀"  # Disabled for release
}

# Add disabled node display names if experimental mode is enabled
if ENABLE_EXPERIMENTAL_NODES:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        # Processing Nodes
        "WANVaceSplitReferenceVideo": "WanVace-pipeline Split Video Batch ✂️",
        
        # Frame Utility Nodes
        "WANVaceFrameSampler": "WanVace-pipeline Frame Sampler 📊",
        
        # Effects Nodes
        "WANVaceBatchStartIndex": "WanVace-pipeline Batch Start Index 🔢",
        
        # Fast Processing Nodes
        "WANFastImageCompositeMasked": "WanVace-pipeline Fast Image Composite Masked 🚀",
        "WANFastImageBlend": "WanVace-pipeline Fast Image Blend 🚀",
        "WANFastImageScaleBy": "WanVace-pipeline Fast Image Scale By 🚀",
        "WANFastImageScaleToMegapixels": "WanVace-pipeline Fast Image Scale To Megapixels 🚀",
        "WANFastImageResize": "WanVace-pipeline Fast Image Resize 🚀",
        
        # Fast Video Processors
        "WANFastVideoEncode": "WanVace-pipeline Fast Video Encode 🚀",
        "WANFastVACEEncode": "WanVace-pipeline Fast VACE Encode 🚀",
        "WANFastVideoCombine": "WanVace-pipeline Fast Video Combine 🚀"
    })

# Add VACE Loop Encoder display name if node was successfully imported
if WANVACELoopEncoder is not None and ENABLE_EXPERIMENTAL_NODES:
    NODE_DISPLAY_NAME_MAPPINGS["WANVACELoopEncoder"] = "WanVace-pipeline VACE Loop Encoder 🔁"

# Add mask node display names if they were successfully imported
# WANVaceMaskEditor removed - using WANVaceMaskEditorDirect instead
if WANVaceMaskViewer is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANVaceMaskViewer"] = "WanVace-pipeline Mask Viewer 👁️"
if WANVaceTestMask is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANVaceTestMask"] = "WanVace-pipeline Test Mask 🧪"

# Add WAN Inpaint Conditioning display name if node was successfully imported
if WANInpaintConditioning is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANInpaintConditioning"] = "WanVace-pipeline WAN Inpaint Conditioning 🎨"

# Add WAN Video Sampler Inpaint display name if node was successfully imported
if WANVideoSamplerInpaint is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANVideoSamplerInpaint"] = "WanVace-pipeline WAN Video Sampler Inpaint 🎭"

# Add WAN Tiled Sampler display name if node was successfully imported
if WANTiledSampler is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANTiledSampler"] = "WanVace-pipeline WAN Tiled Sampler 🔲"

# Add WAN Match Batch Size display name if node was successfully imported
if WANMatchBatchSize is not None:
    NODE_DISPLAY_NAME_MAPPINGS["WANMatchBatchSize"] = "WanVace-pipeline Match Batch Size 🔄"