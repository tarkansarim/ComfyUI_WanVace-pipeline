# ComfyUI WAN Vace Pipeline

A comprehensive suite of video processing and timeline management nodes for ComfyUI, designed for AI-based frame interpolation workflows.

## Features

### 🎬 Video Processing Nodes
- **WANLoadVideo** - Load video files with frame extraction
- **WANSaveVideo** - Save processed frames as video or image sequences
- **WANVaceSplitReferenceVideo** - Split videos into manageable batches
- **WANVaceJoinVideos** - Join multiple videos with customizable gaps
- **WANVaceVideoExtension** - Rearrange frames (move end frames to start)
- **WANVaceFrameInterpolation** - Insert black frames for AI interpolation

### 🎯 Timeline Management
- **WANVaceKeyframeTimeline** - Advanced interactive timeline widget featuring:
  - **Drag-and-drop keyframe positioning** - Natural file dropping onto timeline
  - **Multi-keyframe selection** - Shift+click individual frames or Shift+drag on empty timeline space for box selection
  - **Real-time preview during selection** - Red scrubber follows mouse during selection
  - **Per-keyframe hold duration control** - Set how long each frame is held (1-100 frames)
  - **Visual hold duration indicators** - Green lines show hold duration below keyframes
  - **Proportional scaling system** - Scale timing between multiple selected keyframes
    - Left handle: Scale with right edge as pivot
    - Right handle: Scale with left edge as pivot
    - Real-time percentage display and golden ghost keyframes
    - Binary search collision detection prevents overlap
  - **Timeline zoom and scroll controls** - Focus on specific sections
  - **Red wooden block selection** - 3D visual selection indicator
  - **Intelligent image persistence** - Multi-layer caching system:
    - Global memory storage (survives workflow reloads)
    - localStorage backup (survives browser sessions)
    - Automatic folder reload with smart file matching
  - **Folder-based image reloading** - "Select Image Folder" button for easy reloading
  - **Integer frame display** - Clean frame numbers during playback (F0, F1, F2...)
  - **Collision detection** - Prevents keyframe overlap during all operations
  - **Robust undo/redo system** - Unique ID tracking survives multiple operations

### 🎨 Advanced Mask Editor Integration
- **WANVaceMaskEditor** - Sophisticated PyQt5 mask editor with timeline integration
- **WANVaceMaskFromProject** - Load masks from saved projects
- **Shape-based masking** with vertex editing and keyframe interpolation
- **Multiple drawing modes**: Brush, shape, and liquify tools
- **Visual enhancements**:
  - Dimmed interpolated shapes when drawing new shapes (improves video tracing)
  - Adaptive vertex sizing (scales down with high vertex counts)
  - Smooth vertex point dimming on interpolated frames
- **Performance optimizations**:
  - Lazy mask loading (only allocates masks when edited)
  - Fast multithreaded image sequence loading
  - Deferred initialization prevents UI freezes
  - Auto-save every 5 seconds with session persistence
- **User experience improvements**:
  - FPS setting persistence with QSettings
  - Proper cancel behavior (no unwanted save prompts)
  - Clean session cleanup on discard
  - Essential-only logging for production use
- **Liquify mode fixes**: Keyframes only created when actually drawing, not when switching modes

### 🛠️ Utilities
- **WANVaceFrameSampler** - Extract frames at specific intervals
- **WANVaceFrameInjector** - Insert frames at precise positions
- **WANVaceOutpainting** - Add padding for outpainting workflows
- **WANVaceFrameInfo** - Display video metadata

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI_WanVace-pipeline.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Restart ComfyUI

## Timeline Widget Usage

### Getting Started

#### 1. Adding Your First Keyframes
1. **Add the Timeline Node**: In ComfyUI, add a "WanVace-pipeline Keyframe Timeline" node
2. **Load Images**: Click the "Add Keyframes" button or simply drag & drop image files directly onto the timeline
3. **Position Keyframes**: Images appear as numbered keyframes (F0, F1, F2...). To move them: Shift+left click to select a single keyframe or Shift+drag on empty timeline space across keyframes to select multiple with a red selection box, then click inside the red selection area to drag them to new positions
4. **Preview**: Click the play button or use the red scrubber to preview your timeline

#### 2. Understanding the Interface
- **Dark Gray Timeline Track**: The main area where keyframes are placed
- **Red Scrubber**: Shows current playback position, drag to preview frames
- **Numbered Keyframes**: Your images displayed as F0, F1, F2... with small thumbnails
- **Control Buttons**: Play/Pause, Clear All, Add Keyframes, Delete Selected
- **Zoom Controls**: Slider and scroll bar for detailed timeline editing

### Basic Operations

#### Adding & Managing Keyframes
- **Add Keyframes**: Click "Add Keyframes" button or drag & drop images directly onto timeline
- **Move Single Keyframe**: Shift+left click a keyframe to select it, then click and drag to move
- **Move Multiple Keyframes**: Hold Shift+left click and drag on empty timeline space to create a red selection box around keyframes, then click anywhere inside the red selection area to move them all together
- **Delete Keyframes**: Select keyframes and click "Delete Selected" or press Delete key
- **Playback**: Use play/pause button, spacebar, or drag the red scrubber

#### Timeline Navigation
- **Scrubbing**: Click and drag on the timeline (without Shift) - the red scrubber follows your mouse movement for real-time frame preview, just like traditional timeline software
- **Zooming**: Use the zoom slider (located above the timeline) to focus on specific timeline sections
- **Scrolling**: When zoomed in, use the scroll bar (located above the timeline) or drag within the scroll area

### Advanced Features

#### Multi-Selection & Real-Time Preview
- **Single Selection**: Shift+left click to select individual keyframes (selected keyframes appear highlighted)
- **Box Selection**: Shift+left click and drag on empty timeline space to create a red selection box around multiple keyframes
- **Red Selection Visualization**: Selected keyframes show a red 3D wooden block area
- **Moving Selected Keyframes**: After selection, click anywhere inside the red selection area to drag and move all selected keyframes together
- **Selection Preview**: Red scrubber follows mouse during box selection for real-time frame preview
- **Block Operations**: Move, scale, or delete multiple keyframes simultaneously

#### Boundary Indicators (Visual Guides)
- **Red/Green Flashing Indicators**: Vertical dashed lines provide visual guidance for keyframe placement
- **Green Indicators**: Show when a keyframe is placed exactly on a batch boundary (visual confirmation only)
- **Red Indicators**: Show empty batch boundaries as visual reference points
- **Exact Placement Required**: Indicators only turn green when keyframes are placed precisely on boundary frames
- **Purpose**: Visual guides to help organize keyframes - they do not affect the output images or processing

#### Keyframe Hold Duration
- **Select and Adjust**: Select keyframes and use +/- buttons to set hold duration (1-100 frames)
- **Visual Indicators**: Green lines below keyframes show hold duration length
- **Preview Impact**: Hold duration affects how long each frame appears during playback
- **AI Video Generation Purpose**: Frame holding is useful when certain movements don't reflect as expected in the generated video - holding frames for longer can reinforce the pose to show in the generated video
- **Mask Exclusion**: "Ignore held frames for mask" checkbox excludes held frames from interpolation
- **Output Behavior**: Hold duration affects both reference and preview frame outputs

#### Proportional Timeline Scaling
- **Arrow Handles**: When multiple keyframes are selected, arrow handles appear on selection edges
- **Left Handle (←)**: Scale with right edge as pivot - drag left/right to compress/expand timing
- **Right Handle (→)**: Scale with left edge as pivot - drag left/right to adjust timing
- **Visual Feedback**: Shows scaling percentage and golden ghost keyframes during drag
- **Smart Collision Detection**: Prevents keyframes from overlapping during scaling operations
- **Proportional Spacing**: Maintains relative timing between all selected keyframes

#### Intelligent Image Persistence
- **Automatic Caching**: Images persist across ComfyUI restarts and workflow reloads
- **Multi-layer Storage**: Global memory → localStorage → file metadata backup system
- **Smart Reloading**: "Select Image Folder" button for easy batch reloading with filename matching
- **Folder Memory**: Shows original folder name hints when available
- **Browser Compatible**: Works within browser security limitations while maximizing persistence

#### Advanced Controls
- **Zoom & Scroll**: Use zoom slider to focus on specific timeline sections, scroll bar appears when zoomed
- **Collision Detection**: Prevents keyframe overlap during all operations (drag, scale, move)
- **Undo/Redo System**: Robust operation tracking with unique ID system

### Visual Elements Guide

#### Keyframe States
- **Normal Keyframes**: White numbered frames (F0, F1, F2...) with small thumbnails
- **Selected Keyframes**: Highlighted with yellow border
- **Dragged Keyframes**: Show translucent preview at target position
- **Hold Duration**: Green lines extending right from keyframes

#### Selection Indicators
- **Red Wooden Block**: 3D visualization showing multi-keyframe selection area
- **Arrow Handles**: Appear on selection edges for proportional scaling
- **Golden Ghost Keyframes**: Show preview positions during scaling operations

#### Timeline Elements
- **Batch Boundaries**: Vertical dashed orange lines marking visual reference boundaries
- **Boundary Indicators**: Red/green flashing indicators for visual keyframe placement guidance
- **Frame Markers**: Small tick marks showing frame numbers
- **Zoom Slider**: Bottom control for timeline magnification
- **Scroll Bar**: Appears when zoomed in for navigation

### Keyboard Shortcuts
- **Shift+Click**: Toggle keyframe selection
- **Shift+Drag**: Box selection (on empty timeline space)

### Tips & Tricks

#### Efficient Workflow
1. **Use Boundary Indicators**: Place keyframes on green boundary indicators for visual organization (does not affect output)
2. **Plan Hold Durations**: Set longer holds for important frames, shorter for transitions
3. **Multi-Select for Batch Operations**: Use Shift+drag on empty timeline space to select multiple keyframes for simultaneous editing
4. **Zoom for Precision**: Use the zoom slider when working with dense keyframe sequences

#### Performance Optimization
- **Folder-Based Loading**: Keep images in organized folders for easier reloading
- **Use Appropriate Hold Durations**: Longer holds reduce processing requirements
- **Batch Boundary Alignment**: Align important keyframes with batch boundaries (green indicators)

### Troubleshooting

#### Common Issues
- **After Restarting ComfyUI**: When you restart ComfyUI and reload your workflow, you'll see two buttons on the timeline node: "Add Keyframes" and "Select Image Folder". Use "Select Image Folder" to reload your previously used images from their folder location
- **Images Don't Load**: Use "Select Image Folder" button to reload from organized folder structure
- **Keyframes Won't Move**: Check if multiple keyframes are selected (use Shift+click to deselect)
- **Boundary Indicators Stay Red**: Move keyframes to exact boundary frame positions (not just nearby)
- **Timeline Appears Frozen**: Right-click on timeline area to refresh widget focus

#### Performance Issues
- **Slow Loading**: Organize images in single folder, use moderate image sizes
- **Memory Usage**: Clear unused keyframes with "Clear All" button
- **UI Responsiveness**: Reduce zoom level if timeline becomes sluggish

#### Browser Compatibility
- **Image Persistence**: Use "Select Image Folder" if images don't reload after browser restart
- **Drag & Drop**: Ensure images are dropped directly onto the dark timeline track area

## Node Structure

```
ComfyUI_WanVace-pipeline/
├── __init__.py              # Main node registration
├── nodes/                   # Node implementations
│   ├── __init__.py         
│   ├── video_io.py         # Video I/O operations
│   ├── video_processing.py  # Video manipulation
│   ├── timeline.py         # Timeline widget
│   ├── frame_utils.py      # Frame utilities
│   ├── effects.py          # Visual effects
│   └── mask_editor.py      # Mask editor integration
├── web/                    # Frontend code
│   └── js/
│       ├── wan_keyframe_timeline.js  # Timeline widget
│       └── wan_mask_editor.js        # Mask editor widget
└── reference/              # Reference implementation

```

## Workflow Examples

### Basic Frame Interpolation
1. Load video with `WANLoadVideo`
2. Use `WANVaceKeyframeTimeline` to set keyframes
3. Apply `WANVaceFrameInterpolation` to insert frames
4. Process with your AI model
5. Save with `WANSaveVideo`

### Advanced Timeline Editing
1. Drag images onto timeline to create keyframes
2. Shift+drag on empty timeline space to select multiple keyframes
3. Use arrow handles to adjust timing
4. Set per-keyframe hold durations
5. Preview with built-in playback

## Development

See [CLAUDE.md](CLAUDE.md) for development guidelines and architecture details.

## License

[Your License Here]

## Credits

Developed for ComfyUI by the WAN Vace Pipeline team.
Special thanks to the ComfyUI community for their support and feedback.