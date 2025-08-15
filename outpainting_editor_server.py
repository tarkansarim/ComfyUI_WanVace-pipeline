"""
Server endpoints for outpainting editor integration
"""
import os
import sys
import json
import subprocess
import tempfile
import time
import numpy as np
from aiohttp import web
import server
from server import PromptServer

# Store active outpainting editor processes
active_editors = {}

@PromptServer.instance.routes.post("/wan_vace/launch_outpainting_editor")
async def launch_outpainting_editor(request):
    """Launch the PyQt outpainting editor and return the canvas data"""
    try:
        data = await request.json()
        node_id = data.get("node_id")
        existing_output_dir = data.get("output_dir")  # Get existing output dir from client
        
        # Debug: Print the temp directory structure
        print(f"[Outpainting Editor] tempfile.gettempdir() = {tempfile.gettempdir()}")
        print(f"[Outpainting Editor] Platform: {os.name}")
        print(f"[Outpainting Editor] Received existing output_dir from client: {existing_output_dir}")
        
        # Create temp directory for this session
        temp_dir = tempfile.mkdtemp(prefix="wan_outpainting_")
        config_path = os.path.join(temp_dir, "config.json")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[Outpainting Editor] Temp directory: {temp_dir}")
        print(f"[Outpainting Editor] Debug log will be at: {os.path.join(temp_dir, 'outpainting_editor_debug.log')}")
        
        # The outpainting editor will load its own files, so we just pass empty input_frames
        input_frames = []
        print("[Outpainting Editor] Outpainting editor will load its own files")
        
        # Debug: Print active_editors contents
        print(f"[Outpainting Editor] Active editors dict size: {len(active_editors)}")
        print(f"[Outpainting Editor] Active editors keys: {list(active_editors.keys())}")
        
        # Initialize session variables
        session_data_str = None
        session_output_dir = None
        
        # Check if we have an existing output dir from the client
        if existing_output_dir and os.path.exists(existing_output_dir):
            print(f"[Outpainting Editor] Found existing output dir from client: {existing_output_dir}")
            output_dir = existing_output_dir
            session_output_dir = existing_output_dir
            
            # Try to load session data from this directory
            session_file = os.path.join(output_dir, "canvas_data.json")
            if os.path.exists(session_file):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    session_data_str = json.dumps(session_data)
                    print(f"[Outpainting Editor] Loaded session data from existing output dir")
                    # Update active_editors
                    active_editors[f"{node_id}_output_dir"] = output_dir
                    active_editors[f"{node_id}_session_data"] = session_data_str
                except Exception as e:
                    print(f"[Outpainting Editor] Error loading session from existing output dir: {e}")
                    session_data_str = None
            else:
                print(f"[Outpainting Editor] No canvas_data.json in existing output dir")
                session_data_str = None
        else:
            # Check for persistent session files
            sessions_dir = os.path.join(tempfile.gettempdir(), "wan_outpainting_sessions")
            print(f"[Outpainting Editor] Sessions dir: {sessions_dir}")
            print(f"[Outpainting Editor] Sessions dir exists: {os.path.exists(sessions_dir)}")
            
            try:
                os.makedirs(sessions_dir, exist_ok=True)
                print(f"[Outpainting Editor] Created/verified sessions directory")
            except Exception as e:
                print(f"[Outpainting Editor] Error creating sessions directory: {e}")
            
            node_session_file = os.path.join(sessions_dir, f"node_{node_id}_session.json")
            
            print(f"[Outpainting Editor] Checking persistent session file: {node_session_file}")
            if os.path.exists(node_session_file):
                try:
                    with open(node_session_file, 'r') as f:
                        persistent_session = json.load(f)
                    
                    # Check if the output directory still exists
                    if "output_dir" in persistent_session and os.path.exists(persistent_session["output_dir"]):
                        output_dir = persistent_session["output_dir"]
                        print(f"[Outpainting Editor] Found persistent session, using output dir: {output_dir}")
                        
                        # Load session data
                        session_file = os.path.join(output_dir, "canvas_data.json")
                        if os.path.exists(session_file):
                            with open(session_file, 'r') as f:
                                session_data = json.load(f)
                            session_data_str = json.dumps(session_data)
                            print(f"[Outpainting Editor] Loaded session data from persistent session")
                            
                            # Update active_editors
                            active_editors[f"{node_id}_output_dir"] = output_dir
                            active_editors[f"{node_id}_session_data"] = session_data_str
                except Exception as e:
                    print(f"[Outpainting Editor] Error loading persistent session: {e}")
            
            # Check if we have session data for this node in memory
            if not session_data_str:
                session_data_str = active_editors.get(f"{node_id}_session_data", None)
            session_output_dir = active_editors.get(f"{node_id}_output_dir", None)
        
        print(f"[Outpainting Editor] Checking for session data for node {node_id}")
        print(f"[Outpainting Editor] session_data_str exists: {bool(session_data_str)}")
        print(f"[Outpainting Editor] session_output_dir: {session_output_dir if 'session_output_dir' in locals() else output_dir}")
        
        # If we have a previous session, use its output directory
        if session_output_dir and os.path.exists(session_output_dir):
            output_dir = session_output_dir
            print(f"[Outpainting Editor] Using previous session output dir: {output_dir}")
            
            # Also check for canvas_data.json file if not in memory
            if not session_data_str:
                session_file = os.path.join(output_dir, "canvas_data.json")
                print(f"[Outpainting Editor] Checking for session file: {session_file}")
                print(f"[Outpainting Editor] Session file exists: {os.path.exists(session_file)}")
                if os.path.exists(session_file):
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    session_data_str = json.dumps(session_data)
                    print(f"[Outpainting Editor] Loaded session data from file: {session_file}")
                    # Print the video_info to debug
                    if "video_info" in session_data:
                        print(f"[Outpainting Editor] Video info from session: {session_data['video_info']}")
        
        # Save config for launcher
        config = {
            "output_dir": output_dir,
            "input_frames": [],
            "session_data": session_data_str  # Pass existing session data to launcher
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Launch the outpainting editor
        script_path = os.path.join(os.path.dirname(__file__), "nodes", "comfyui_outpainting_launcher.py")
        print(f"[Outpainting Editor] Launching script: {script_path}")
        
        # Add debug log redirection
        debug_log_path = os.path.join(temp_dir, "outpainting_editor_debug.log")
        
        with open(debug_log_path, 'w') as debug_log:
            # Pass the environment including PYTHONPATH if needed
            env = os.environ.copy()
            
            process = subprocess.Popen(
                [sys.executable, script_path, config_path], 
                stdout=debug_log,
                stderr=subprocess.STDOUT,
                env=env
            )
        
        print(f"[Outpainting Editor] Process launched with PID: {process.pid}")
        
        # Wait for the process to complete
        return_code = process.wait()
        print(f"[Outpainting Editor] Process completed with return code: {return_code}")
        
        # Read debug log
        if os.path.exists(debug_log_path):
            with open(debug_log_path, 'r') as f:
                debug_content = f.read()
                if debug_content:
                    print(f"[Outpainting Editor] Debug log content:")
                    print("=" * 80)
                    print(debug_content)
                    print("=" * 80)
        
        # Read the results
        result_path = os.path.join(output_dir, "canvas_data.json")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                canvas_data = json.load(f)
            
            # Store the session data and output directory for future use
            active_editors[f"{node_id}_session_data"] = json.dumps(canvas_data)
            active_editors[f"{node_id}_output_dir"] = output_dir
            
            # Save persistent session file
            try:
                sessions_dir = os.path.join(tempfile.gettempdir(), "wan_outpainting_sessions")
                os.makedirs(sessions_dir, exist_ok=True)
                node_session_file = os.path.join(sessions_dir, f"node_{node_id}_session.json")
                
                persistent_session = {
                    "output_dir": output_dir,
                    "node_id": node_id,
                    "timestamp": time.time()
                }
                
                with open(node_session_file, 'w') as f:
                    json.dump(persistent_session, f, indent=2)
                print(f"[Outpainting Editor] Saved persistent session to: {node_session_file}")
            except Exception as e:
                print(f"[Outpainting Editor] Error saving persistent session: {e}")
            
            print(f"[Outpainting Editor] Canvas data loaded from: {result_path}")
            
            # Include the output directory in the response
            canvas_data["output_dir"] = output_dir
            
            return web.json_response({
                "success": True,
                "canvas_data": json.dumps(canvas_data)
            })
        else:
            print(f"[Outpainting Editor] No canvas data found at: {result_path}")
            return web.json_response({
                "success": False,
                "error": "No canvas data generated"
            })
            
    except Exception as e:
        print(f"[Outpainting Editor] Error: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({
            "success": False,
            "error": str(e)
        })

# Register the endpoint
print("[WAN Vace Pipeline] Registering outpainting editor endpoint")