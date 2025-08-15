"""
Server endpoints for mask editor integration
"""
import os
import json
import subprocess
import tempfile
import time
import numpy as np
from aiohttp import web
import server
from server import PromptServer

def load_session_data(output_dir):
    """Load session data with three-state system (current vs working vs saved)"""
    # ALWAYS load current state first (this is the accepted state)
    current_state_file = os.path.join(output_dir, "current_state.json")
    if os.path.exists(current_state_file):
        print(f"[Mask Editor] Loading current state")
        with open(current_state_file, 'r') as f:
            return json.load(f)
    
    # If no current state, check for saved project (from explicit saves)
    saved_project_file = os.path.join(output_dir, "saved_project.json")
    if os.path.exists(saved_project_file):
        print(f"[Mask Editor] Loading saved project as fallback")
        with open(saved_project_file, 'r') as f:
            return json.load(f)
    
    # Legacy support - try old file names
    legacy_files = ["saved_session_data.json", "session_data.json"]
    for legacy_file in legacy_files:
        file_path = os.path.join(output_dir, legacy_file)
        if os.path.exists(file_path):
            print(f"[Mask Editor] Loading legacy {legacy_file}")
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Migrate to new current_state.json
            current_state_file = os.path.join(output_dir, "current_state.json")
            with open(current_state_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[Mask Editor] Migrated {legacy_file} to current_state.json")
            return data
    
    # Note: We NEVER load working_autosave.json here
    # That's only for crash recovery, not normal operation
    return None

# Store active mask editor processes
active_editors = {}

@PromptServer.instance.routes.post("/wan_vace/launch_mask_editor")
async def launch_mask_editor(request):
    """Launch the PyQt mask editor and return the mask data"""
    try:
        data = await request.json()
        node_id = data.get("node_id")
        existing_output_dir = data.get("output_dir")  # Get existing output dir from client
        
        # Debug: Print the temp directory structure
        print(f"[Mask Editor] tempfile.gettempdir() = {tempfile.gettempdir()}")
        print(f"[Mask Editor] Platform: {os.name}")
        print(f"[Mask Editor] Received existing output_dir from client: {existing_output_dir}")
        
        # Create temp directory for this session
        temp_dir = tempfile.mkdtemp(prefix="wan_mask_")
        config_path = os.path.join(temp_dir, "config.json")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[Mask Editor] Temp directory: {temp_dir}")
        print(f"[Mask Editor] Debug log will be at: {os.path.join(temp_dir, 'mask_editor_debug.log')}")
        print(f"[Mask Editor] Loading debug file will be at: {os.path.join(output_dir, 'loading_debug.txt')}")
        
        # The mask editor will load its own files, so we just pass empty input_frames
        input_frames = []
        print("[Mask Editor] Mask editor will load its own files")
        
        # Debug: Print active_editors contents
        print(f"[Mask Editor] Active editors dict size: {len(active_editors)}")
        print(f"[Mask Editor] Active editors keys: {list(active_editors.keys())}")
        
        # Initialize session variables
        session_data_str = None
        session_output_dir = None
        
        # Check if we have an existing output dir from the client
        if existing_output_dir and os.path.exists(existing_output_dir):
            print(f"[Mask Editor] Found existing output dir from client: {existing_output_dir}")
            output_dir = existing_output_dir
            session_output_dir = existing_output_dir
            
            # Try to load session data from this directory
            try:
                session_data = load_session_data(output_dir)
                if session_data:
                    session_data_str = json.dumps(session_data)
                    print(f"[Mask Editor] Loaded session data from existing output dir")
                    # Update active_editors
                    active_editors[f"{node_id}_output_dir"] = output_dir
                    active_editors[f"{node_id}_session_data"] = session_data_str
                else:
                    print(f"[Mask Editor] No session data in existing output dir")
                    session_data_str = None
            except Exception as e:
                print(f"[Mask Editor] Error loading session from existing output dir: {e}")
                session_data_str = None
        else:
            # Check if we have session data for this node in memory
            session_data_str = active_editors.get(f"{node_id}_session_data", None)
            session_output_dir = active_editors.get(f"{node_id}_output_dir", None)
        
        print(f"[Mask Editor] Checking for session data for node {node_id}")
        print(f"[Mask Editor] session_data_str exists: {bool(session_data_str)}")
        print(f"[Mask Editor] session_output_dir: {session_output_dir if 'session_output_dir' in locals() else output_dir}")
        
        # If we have a previous session, use its output directory
        if session_output_dir and os.path.exists(session_output_dir):
            output_dir = session_output_dir
            print(f"[Mask Editor] Using previous session output dir: {output_dir}")
            
            # Also check for session data files if not in memory
            if not session_data_str:
                session_data = load_session_data(output_dir)
                if session_data:
                    session_data_str = json.dumps(session_data)
                    print(f"[Mask Editor] Loaded session data from file")
                    # Print the video_info to debug
                    if "video_info" in session_data:
                        print(f"[Mask Editor] video_info in session: {session_data['video_info']}")
                    else:
                        print(f"[Mask Editor] No video_info in session data")
        else:
            # No previous session in memory, check for persistent session storage
            print(f"[Mask Editor] No session in memory, checking persistent storage...")
            # Create a persistent sessions directory in temp directory, not its parent
            sessions_dir = os.path.join(tempfile.gettempdir(), "wan_mask_sessions")
            print(f"[Mask Editor] Sessions directory: {sessions_dir}")
            print(f"[Mask Editor] Sessions dir exists: {os.path.exists(sessions_dir)}")
            
            try:
                os.makedirs(sessions_dir, exist_ok=True)
                print(f"[Mask Editor] Created/verified sessions directory")
            except Exception as e:
                print(f"[Mask Editor] Error creating sessions directory: {e}")
            
            node_session_file = os.path.join(sessions_dir, f"node_{node_id}_session.json")
            
            print(f"[Mask Editor] Checking persistent session file: {node_session_file}")
            if os.path.exists(node_session_file):
                try:
                    with open(node_session_file, 'r') as f:
                        persistent_session = json.load(f)
                    
                    # Check if the output directory still exists
                    if "output_dir" in persistent_session and os.path.exists(persistent_session["output_dir"]):
                        output_dir = persistent_session["output_dir"]
                        print(f"[Mask Editor] Found persistent session, using output dir: {output_dir}")
                        
                        # Load session data
                        session_data = load_session_data(output_dir)
                        if session_data:
                            session_data_str = json.dumps(session_data)
                            print(f"[Mask Editor] Loaded session data from persistent session")
                            
                            # Update active_editors
                            active_editors[f"{node_id}_output_dir"] = output_dir
                            active_editors[f"{node_id}_session_data"] = session_data_str
                except Exception as e:
                    print(f"[Mask Editor] Error loading persistent session: {e}")
        
        # Create a config for the launcher
        config = {
            "input_frames": input_frames,
            "output_dir": output_dir,
            "project_data": session_data_str if session_data_str else ""  # Use session data if available
        }
        
        print(f"[Mask Editor] Config being sent:")
        print(f"[Mask Editor]   - output_dir: {output_dir}")
        print(f"[Mask Editor]   - has project_data: {bool(session_data_str)}")
        if session_data_str:
            print(f"[Mask Editor]   - project_data length: {len(session_data_str)}")
            # Try to parse and check for video_info
            try:
                data = json.loads(session_data_str)
                if "video_info" in data:
                    print(f"[Mask Editor]   - video_info found: {data['video_info']}")
            except:
                pass
        
        # Write config
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Path to the mask editor launcher (now in nodes folder)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        launcher_path = os.path.join(script_dir, "nodes", "comfyui_mask_launcher.py")
        
        # Debug logging to show exact paths
        print(f"[Mask Editor] Script directory: {script_dir}")
        print(f"[Mask Editor] Looking for launcher at: {launcher_path}")
        print(f"[Mask Editor] Launcher exists: {os.path.exists(launcher_path)}")
        
        # List files in the nodes directory to help debugging
        ref_dir = os.path.join(script_dir, "nodes")
        if os.path.exists(ref_dir):
            print(f"[Mask Editor] Files in {ref_dir}:")
            for f in os.listdir(ref_dir):
                if f.endswith('.py'):
                    print(f"  - {f}")
        
        if os.path.exists(launcher_path):
            print(f"[Mask Editor] Found launcher script!")
            print(f"[Mask Editor] Absolute path: {os.path.abspath(launcher_path)}")
        else:
            # Try alternative paths
            alt_paths = [
                os.path.join(script_dir, "reference_only", "PrepForWanFrameInterp_comfyui", "mask_editor_launcher.py"),
                os.path.join(script_dir, "mask_editor_launcher.py"),
                os.path.join(script_dir, "reference_only", "mask_editor_launcher.py")
            ]
            print(f"[Mask Editor] Launcher not found at expected path, trying alternatives:")
            for alt_path in alt_paths:
                print(f"  - {alt_path}: {os.path.exists(alt_path)}")
                if os.path.exists(alt_path):
                    launcher_path = alt_path
                    print(f"[Mask Editor] Using alternative launcher: {launcher_path}")
                    break
        
        # Launch the mask editor
        if os.path.exists(launcher_path):
            # Run the mask editor using the same Python interpreter as ComfyUI
            import sys
            
            # Set environment variable to avoid Qt platform issues
            env = os.environ.copy()
            env['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
            
            print(f"[Mask Editor] ========== LAUNCHING MASK EDITOR ==========")
            print(f"[Mask Editor] Python interpreter: {sys.executable}")
            print(f"[Mask Editor] Launcher script: {launcher_path}")
            print(f"[Mask Editor] Config file: {config_path}")
            print(f"[Mask Editor] Full command: {sys.executable} {launcher_path} --config {config_path}")
            print(f"[Mask Editor] ===========================================")
            
            try:
                process = subprocess.Popen(
                    [sys.executable, launcher_path, "--config", config_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env
                )
            except Exception as e:
                print(f"[Mask Editor] Failed to launch: {e}")
                return web.json_response({
                    "success": False,
                    "error": f"Failed to launch mask editor: {str(e)}"
                })
            
            # Store the process
            active_editors[node_id] = {
                "process": process,
                "temp_dir": temp_dir,
                "output_dir": output_dir
            }
            
            # Wait for the process to complete (blocking)
            stdout, stderr = process.communicate()
            
            # Print debug info
            if stdout:
                stdout_text = stdout.decode()
                for line in stdout_text.split('\n'):
                    if line.strip():
                        print(f"[Mask Editor Output] {line}")
            if stderr:
                stderr_text = stderr.decode()
                for line in stderr_text.split('\n'):
                    if line.strip():
                        print(f"[Mask Editor Error] {line}")
            print(f"[Mask Editor] Process exit code: {process.returncode}")
            
            # Also check for debug log
            debug_log = os.path.join(temp_dir, "mask_editor_debug.log")
            if os.path.exists(debug_log):
                print(f"[Mask Editor] Debug log saved at: {debug_log}")
                with open(debug_log, 'r') as f:
                    print("[Mask Editor] === Debug Log Contents ===")
                    print(f.read())
                    print("[Mask Editor] === End Debug Log ===")
            
            # Check if project data was created (we now create minimal files)
            project_file = os.path.join(output_dir, "project_data.json")
            frame_files = sorted([f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".png")]) if os.path.exists(output_dir) else []
            mask_files = sorted([f for f in os.listdir(output_dir) if f.startswith("mask_") and f.endswith(".png")]) if os.path.exists(output_dir) else []
            
            # Try project_data.json first, then use load_session_data for session files
            project_data = None
            if os.path.exists(project_file):
                # Load project data
                with open(project_file, 'r') as f:
                    project_data = json.load(f)
            else:
                # Try to load session data
                project_data = load_session_data(output_dir)
                if project_data:
                    print(f"[Mask Editor] Using session data as project data")
            
            if project_data:
                # Create mask data response - include minimal file references
                mask_data = {
                    "project_data": project_data,
                    "output_dir": output_dir,
                    "frame_files": frame_files,
                    "mask_files": mask_files
                }
                
                # Store session data for this node
                active_editors[f"{node_id}_session_data"] = json.dumps(project_data)
                active_editors[f"{node_id}_output_dir"] = output_dir
                print(f"[Mask Editor] Stored session data for node {node_id}")
                
                # Also save to persistent storage
                sessions_dir = os.path.join(tempfile.gettempdir(), "wan_mask_sessions")
                os.makedirs(sessions_dir, exist_ok=True)
                node_session_file = os.path.join(sessions_dir, f"node_{node_id}_session.json")
                
                persistent_session = {
                    "output_dir": output_dir,
                    "timestamp": time.time()
                }
                with open(node_session_file, 'w') as f:
                    json.dump(persistent_session, f)
                print(f"[Mask Editor] Saved persistent session to: {node_session_file}")
                
                # Clean up temp files (but keep masks for now)
                os.unlink(config_path)
                for frame_path in input_frames:
                    if os.path.exists(frame_path):
                        os.unlink(frame_path)
                
                if node_id in active_editors:
                    del active_editors[node_id]
                
                return web.json_response({
                    "success": True,
                    "mask_data": mask_data
                })
            else:
                # No masks created (user cancelled)
                # But still save session data if it exists (use saved state, not working state)
                session_data = load_session_data(output_dir)
                if session_data:
                    active_editors[f"{node_id}_session_data"] = json.dumps(session_data)
                    active_editors[f"{node_id}_output_dir"] = output_dir
                    print(f"[Mask Editor] Saved session data on cancel for node {node_id}")
                    
                    # Also save to persistent storage
                    sessions_dir = os.path.join(tempfile.gettempdir(), "wan_mask_sessions")
                    os.makedirs(sessions_dir, exist_ok=True)
                    node_session_file = os.path.join(sessions_dir, f"node_{node_id}_session.json")
                    
                    persistent_session = {
                        "output_dir": output_dir,
                        "timestamp": time.time()
                    }
                    with open(node_session_file, 'w') as f:
                        json.dump(persistent_session, f)
                    print(f"[Mask Editor] Saved persistent session on cancel")
                
                # Clean up config but keep output dir for session
                os.unlink(config_path)
                
                if node_id in active_editors:
                    del active_editors[node_id]
                
                return web.json_response({
                    "success": True,
                    "mask_data": None
                })
        else:
            error_msg = f"Mask editor launcher not found at {launcher_path}"
            print(f"[Mask Editor] ERROR: {error_msg}")
            print(f"[Mask Editor] Current working directory: {os.getcwd()}")
            print(f"[Mask Editor] Script __file__: {__file__}")
            print(f"[Mask Editor] Script absolute path: {os.path.abspath(__file__)}")
            
            return web.json_response({
                "success": False,
                "error": error_msg
            })
            
    except Exception as e:
        return web.json_response({
            "success": False,
            "error": str(e)
        })

# Clean up any active editors on shutdown
import atexit

def cleanup_editors():
    for node_id, editor_info in active_editors.items():
        if editor_info["process"].poll() is None:
            editor_info["process"].terminate()
        # Clean up temp files
        try:
            if os.path.exists(editor_info["temp_dir"]):
                import shutil
                shutil.rmtree(editor_info["temp_dir"])
        except:
            pass

atexit.register(cleanup_editors)