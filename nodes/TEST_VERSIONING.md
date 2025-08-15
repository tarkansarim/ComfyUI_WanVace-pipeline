# WANSaveVideo Versioning System Test

## New Features Added:

### 1. **enable_versioning** (Boolean, default: False)
- Replaces the old `incremental_save` parameter
- When enabled, creates version folders instead of filename suffixes

### 2. **version_prefix** (String, default: "v")
- Customizable prefix for version folders
- Examples: "v" → v1, v2, v3; "test" → test1, test2, test3

## How It Works:

### Example 1: Video Files with Custom Prefix
```
Input:
- output_path: "/path/to/output/my_video.mp4"
- enable_versioning: True
- version_prefix: "v"

Results:
- First run:  /path/to/output/v1/my_video.mp4
- Second run: /path/to/output/v2/my_video.mp4
- Third run:  /path/to/output/v3/my_video.mp4
```

### Example 2: Image Sequences with Custom Prefix
```
Input:
- output_path: "/path/to/output/frames"
- output_mode: "image_sequence"
- enable_versioning: True
- version_prefix: "render"

Results:
- First run:  /path/to/output/render1/frame_00001.png, frame_00002.png...
- Second run: /path/to/output/render2/frame_00001.png, frame_00002.png...
- Third run:  /path/to/output/render3/frame_00001.png, frame_00002.png...
```

### Example 3: Mixed Versions
```
If you have existing folders: v1, v3, v7
Next run will create: v8 (finds highest existing number and adds 1)
```

## Key Improvements:

1. **Clean Organization**: Each version gets its own folder
2. **Custom Naming**: Use any prefix you want (v, version, test, render, etc.)
3. **Smart Detection**: Automatically finds the next available version number
4. **Batch Support**: Works with multi-batch saves from Split node
5. **Backward Compatible**: Disabled by default, existing workflows unchanged

## Usage Tips:

- Use "v" for simple versioning: v1, v2, v3...
- Use "render" for project work: render1, render2, render3...
- Use "test" for experiments: test1, test2, test3...
- Use "draft" for iterations: draft1, draft2, draft3...

This makes organizing your outputs much cleaner and more flexible!