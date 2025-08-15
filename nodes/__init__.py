"""
WAN Vace Pipeline Nodes Package
"""

# Debug: Print which nodes/__init__.py is being loaded
import os
print(f"[WAN Vace Pipeline] Loading nodes/__init__.py from: {os.path.abspath(__file__)}")

# Import all node classes
from .video_io import WANSaveVideo, WANLoadVideo
from .video_processing import (
    WANVaceSplitReferenceVideo,
    WANVaceJoinVideos,
    WANVaceVideoExtension,
    WANVaceFrameInterpolation,
    WANVaceBatchStartIndex
)
from .timeline import WANVaceKeyframeTimeline
from .frame_utils import WANVaceFrameSampler, WANVaceFrameInjector
from .effects import WANVaceOutpainting
from .fast_image_processing import (
    WANFastImageBatchProcessor,
    WANFastImageCompositeMasked,
    WANFastImageBlend,
    WANFastImageScaleBy,
    WANFastImageScaleToMegapixels,
    WANFastImageResize
)
from .fast_controlnet_processors import (
    WANFastDepthAnythingV2,
    WANFastDWPose
)
from .fast_video_processors import (
    WANFastVideoEncode,
    WANFastVACEEncode,
    WANFastVideoCombine
)
# Mask editor is imported separately in node_mappings.py

# Export all nodes
__all__ = [
    "WANSaveVideo",
    "WANLoadVideo", 
    "WANVaceSplitReferenceVideo",
    "WANVaceJoinVideos",
    "WANVaceVideoExtension",
    "WANVaceFrameInterpolation",
    "WANVaceKeyframeTimeline",
    "WANVaceFrameSampler",
    "WANVaceFrameInjector",
    "WANVaceOutpainting",
    "WANFastImageBatchProcessor",
    "WANFastImageCompositeMasked", 
    "WANFastImageBlend",
    "WANFastImageScaleBy",
    "WANFastImageScaleToMegapixels",
    "WANFastImageResize",
    "WANFastDepthAnythingV2",
    "WANFastDWPose",
    "WANFastVideoEncode",
    "WANFastVACEEncode",
    "WANFastVideoCombine",
    "WANVaceBatchStartIndex"
]