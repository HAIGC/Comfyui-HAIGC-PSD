from .haigc_psd import NODE_CLASS_MAPPINGS as PSD_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PSD_NAMES
from .haigc_layer_filter import NODE_CLASS_MAPPINGS as FILTER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FILTER_NAMES
from .advanced_features import NODE_CLASS_MAPPINGS as ADV_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ADV_NAMES

NODE_CLASS_MAPPINGS = {**PSD_MAPPINGS, **FILTER_MAPPINGS, **ADV_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**PSD_NAMES, **FILTER_NAMES, **ADV_NAMES}

WEB_DIRECTORY = "./js"

import os
import shutil
import folder_paths
from aiohttp import web
from server import PromptServer

print("[HAIGC-PSD] Initializing HAIGC PSD custom node...")

import tempfile
import uuid

# Store partial uploads: { upload_id: { "dir": temp_dir, "filename": filename, "total": N, "count": 0 } }
_UPLOAD_CACHE = {}

async def haigc_psd_upload_psd(request):
    try:
        if request.method == "OPTIONS":
            return web.Response(status=204)
        if request.method != "POST":
            return web.Response(status=405, text="Method Not Allowed")

        post = await request.post()
        
        # Check for chunked upload action
        action = post.get("action", None)
        
        # --- Handle Chunked Upload Logic ---
        if action == "init":
            filename = post.get("filename")
            total_chunks = int(post.get("total_chunks", 1))
            upload_id = str(uuid.uuid4())
            
            # Create temp directory for chunks
            temp_dir = tempfile.mkdtemp(prefix="haigc_psd_upload_")
            
            _UPLOAD_CACHE[upload_id] = {
                "dir": temp_dir,
                "filename": filename,
                "total": total_chunks,
                "count": 0
            }
            return web.json_response({"upload_id": upload_id})
            
        elif action == "chunk":
            upload_id = post.get("upload_id")
            chunk_index = int(post.get("chunk_index"))
            chunk_file = post.get("file")
            
            if not upload_id or upload_id not in _UPLOAD_CACHE:
                return web.Response(status=400, text="Invalid upload session")
                
            info = _UPLOAD_CACHE[upload_id]
            chunk_path = os.path.join(info["dir"], f"chunk_{chunk_index}")
            
            with open(chunk_path, "wb") as f:
                shutil.copyfileobj(chunk_file.file, f)
                
            info["count"] += 1
            return web.json_response({"status": "ok"})
            
        elif action == "finish":
            upload_id = post.get("upload_id")
            if not upload_id or upload_id not in _UPLOAD_CACHE:
                 return web.Response(status=400, text="Invalid upload session")
            
            info = _UPLOAD_CACHE[upload_id]
            filename = info["filename"]
            
            # Reassemble file
            input_dir = folder_paths.get_input_directory()
            target_dir = os.path.join(input_dir, "haigc_psd")
            os.makedirs(target_dir, exist_ok=True)

            safe_name = filename.replace("/", "_").replace("\\", "_")
            out_name = safe_name
            stem, ext = os.path.splitext(safe_name)
            counter = 1
            out_path = os.path.join(target_dir, out_name)
            while os.path.exists(out_path):
                out_name = f"{stem}_{counter}{ext}"
                out_path = os.path.join(target_dir, out_name)
                counter += 1
                
            with open(out_path, "wb") as outfile:
                for i in range(info["total"]):
                    chunk_path = os.path.join(info["dir"], f"chunk_{i}")
                    if not os.path.exists(chunk_path):
                        raise Exception(f"Missing chunk {i}")
                    with open(chunk_path, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)
            
            # Cleanup
            shutil.rmtree(info["dir"], ignore_errors=True)
            del _UPLOAD_CACHE[upload_id]
            
            return web.json_response({"path": os.path.abspath(out_path), "filename": out_name, "subfolder": "haigc_psd"})

        # --- Fallback to Standard Single-File Upload (Legacy) ---
        upload = post.get("file", None)
        if upload is None or not getattr(upload, "filename", None):
            return web.Response(status=400, text="missing file")

        filename = os.path.basename(upload.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext != ".psd":
            return web.Response(status=400, text="only .psd allowed")

        input_dir = folder_paths.get_input_directory()
        target_dir = os.path.join(input_dir, "haigc_psd")
        os.makedirs(target_dir, exist_ok=True)

        safe_name = filename.replace("/", "_").replace("\\", "_")
        out_name = safe_name
        stem, ext = os.path.splitext(safe_name)
        counter = 1
        out_path = os.path.join(target_dir, out_name)
        while os.path.exists(out_path):
            out_name = f"{stem}_{counter}{ext}"
            out_path = os.path.join(target_dir, out_name)
            counter += 1

        with open(out_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)

        return web.json_response({"path": os.path.abspath(out_path), "filename": out_name, "subfolder": "haigc_psd"})
    except Exception as e:
        err_msg = str(e)
        print(f"[HAIGC-PSD] Upload error: {err_msg}")
        if "Request Entity Too Large" in err_msg or "413" in err_msg:
             return web.Response(status=413, text="File too large. Please restart ComfyUI to apply the new 100GB limit.")
        return web.Response(status=500, text=err_msg)

try:
    paths = [
        "/haigc_psd/upload_psd",
        "/haigc/upload_psd",
        "/haigc/upload",
    ]
    for p in paths:
        PromptServer.instance.app.router.add_route("*", p, haigc_psd_upload_psd)
        print(f"[HAIGC-PSD] Registered route: {p}")

    # Remove file size limit for uploads
    if hasattr(PromptServer.instance.app, "_client_max_size"):
        PromptServer.instance.app._client_max_size = 100 * 1024 * 1024 * 1024 # 100 GB
        print("[HAIGC-PSD] Upload size limit set to 100GB")
except Exception as e:
    print(f"[HAIGC-PSD] Error registering routes: {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
