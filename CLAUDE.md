# CLAUDE.md - WanVace-pipeline Quick Reference

## 🚨 CRITICAL: Never Archive These Files
- **Root**: `utils.py`, `context.py`, `wan_*.py`, `*_server.py`, `node_mappings.py`, `__init__.py`
- **Nodes**: `comfyui_mask_launcher.py`, `comfyui_outpaint_launcher.py` (server components!)

### Before ANY File Removal:
```bash
grep -r "from.*filename.*import" .
grep -r "import.*filename" .
```

## Directory Structure
- `/nodes/` - Active implementations
- `/web/js/` - UI components (NEVER backup .js here!)
- `/archived_files/` - Unused with `.backup` extension

## Key Technical Info

### VACE System
- Mask = 0: Preserve reference frames
- Mask = 1: Generate new content
- Chain via `prev_vace_embeds`
- 81-frame batch limit

### Performance Rules
- Batch operations only (no loops)
- Native tensors (avoid conversions)
- `NOT_IDEMPOTENT = True` on dynamic nodes

### Recent Fixes
- **OOM**: Slice VACE per batch, not all frames
- **Negative dim**: Set `samples=None` with VACE
- **85 frames**: Force `has_ref=False` always
- **Unified inference**: Same seed, denoise=0.8 for continuations

## Commands
```bash
cd ../.. && python main.py  # Run ComfyUI
ruff check .                # Lint
```

---
**Dependencies**: torch, opencv-python, numpy, PyQt5, ComfyUI-WanVideoWrapper