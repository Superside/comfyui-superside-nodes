"""A "Browse..." button for folder inputs: opens the operating system's folder dialog on the machine running ComfyUI.

ComfyUI has no folder widget, so the product nodes take their folder as text. This adds one small route,
POST /superside/pick_folder, that shows the native dialog and returns the chosen path; web/js/folder_picker.js
puts a button on the nodes that calls it. It only makes sense when the browser and ComfyUI are on the same
computer (the dialog opens on the server); on a remote server the route answers with an error and the text
field keeps working.
"""

import asyncio
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folder_picker.ps1")


def product_root(path):
    """The product folder a picked file belongs to: the nearest folder, up to two levels up, that holds product.json."""
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    probe = folder
    for _ in range(3):
        if os.path.isfile(os.path.join(probe, "product.json")):
            return probe
        probe = os.path.dirname(probe)
    return folder


def pick_folder(pick_file=False):
    """Blocking: show the Windows Explorer folder dialog, return the path or ''.

    The dialog is the shell's own (IFileOpenDialog), driven by folder_picker.ps1. The FolderBrowserDialog that
    Windows PowerShell offers opens a small tree window instead, which users did not recognise as a file browser.
    """
    if sys.platform != "win32":
        raise RuntimeError("the folder dialog is only available when ComfyUI runs on Windows; type the path instead")
    args = ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-File", _SCRIPT] + (["-PickFile"] if pick_file else [])
    out = subprocess.run(args, capture_output=True, text=True, timeout=900)
    if out.returncode != 0 and not out.stdout.strip():
        raise RuntimeError((out.stderr or "the folder dialog could not be opened").strip()[:300])
    picked = out.stdout.strip()
    return product_root(picked) if picked and pick_file else picked


def register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return False                                   # imported outside ComfyUI (tests): nothing to register

    @PromptServer.instance.routes.post("/superside/pick_folder")
    async def _pick_folder(request):
        try:
            try:
                body = await request.json()
            except Exception:
                body = {}
            pick_file = body.get("mode") == "file"
            path = await asyncio.get_event_loop().run_in_executor(None, pick_folder, pick_file)
            return web.json_response({"path": path})
        except Exception as e:
            logger.warning("folder dialog failed: %s", e)
            return web.json_response({"path": "", "error": str(e)}, status=200)

    return True
