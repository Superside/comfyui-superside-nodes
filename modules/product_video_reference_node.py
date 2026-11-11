"""Product Video Reference: short silent clips of the product in use, for video models that accept a video reference.

Optional step 3 of the product pipeline. It finds review / demo videos of the product (links you give, or a
YouTube search), samples a frame every few seconds, lets a cheap vision LLM say what happens in each frame,
merges consecutive frames with the same action into segments, cuts the best ones into clips and builds
~20 s edits. Everything goes into <product_dir>/video_refs/ with an index of tags.

Measured on the kitchen-appliance POC (2026-11): 712 frames from 4 oven videos cost about $0.35 to tag and gave 28
clips in 7 action groups (hand on a control, door, part in/out, food, working, still, person). Known limits:
near-duplicate clips of a talking presenter, and an occasional wrong action label - the index marks clips
with on-screen text so they can be skipped. The footage belongs to its authors: internal reference only.

Needs `yt-dlp` and `ffmpeg` on PATH.
"""

import glob
import json
import logging
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import product_sheet_core as core
from .base_node import API_KEY_INPUT_SPEC, APIClientMixin, SupersideFalNode

logger = logging.getLogger(__name__)

ACTIONS = ["hand operates a control", "door or lid opens or closes", "part is pulled out or put in",
           "food or liquid goes in or comes out", "product working (light on, cooking, brewing)",
           "product shown still", "person next to product (scale)", "product not visible"]
ORDER_A = [ACTIONS[5], ACTIONS[0], ACTIONS[1], ACTIONS[2], ACTIONS[3], ACTIONS[4], ACTIONS[6]]
ORDER_B = ACTIONS[0:5]
VF = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=30,setsar=1"

FRAMES = """PRODUCT: {name}
You get {n} frames from one video, in time order, taken at these seconds: {times}
For EACH frame answer one object:
- "action": exactly one of {actions}
- "what": at most 10 words: what the hand / person / product is doing
- "people": "none", "hand" or "person"
- "shot": "close", "medium" or "wide"
- "overlay": true if text, logos, arrows or a channel watermark are drawn over the picture
- "sharp": true if the product is in focus and well lit
Answer with one JSON object {{"frames": [...]}} with exactly {n} entries and nothing else."""


def _run(args, cwd=None):
    subprocess.run(args, check=True, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class SupersideProductVideoReferenceNode(SupersideFalNode, APIClientMixin):
    """Find videos of the product, cut clips by action and build short reference edits."""

    CATEGORY = "Superside"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("edit_overview_path", "video_refs_dir", "storyboard", "report")
    FUNCTION = "find"
    DISPLAY_NAME = "Product Video Reference"
    DESCRIPTION = (
        "Optional step 3: finds videos that show the product in use (your links, or a YouTube search), cuts "
        "them into short silent clips grouped by action (hand on a control, door opening, part pulled out, "
        "working, person for scale) and builds ~20 s edits. Saved in product_dir/video_refs with an index of "
        "tags. Third-party footage: use as internal reference for video models, do not publish. "
        "Needs yt-dlp and ffmpeg installed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "product_dir": ("STRING", {"default": "", "multiline": False, "tooltip": "The folder from Product Research."}),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "video_urls": ("STRING", {"default": "", "multiline": True, "tooltip": "Video links, one per line (YouTube, Vimeo...)."}),
                "search_videos": ("INT", {"default": 3, "min": 0, "max": 8, "tooltip": "How many videos to look for on YouTube. 0 = only video_urls and videos already in the folder."}),
                "max_video_minutes": ("INT", {"default": 12, "min": 1, "max": 60, "tooltip": "Longer videos are skipped."}),
                "seconds_between_frames": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5}),
                "edit_seconds": ("INT", {"default": 20, "min": 6, "max": 60}),
                "tag_model": ("STRING", {"default": "google/gemini-2.5-flash"}),
            },
        }

    # ------------------------------------------------------------------ download

    def _download(self, folder, product, urls, n_search, max_minutes, report):
        vdir = os.path.join(folder, "videos")
        os.makedirs(vdir, exist_ok=True)
        targets = list(urls)
        if n_search:
            targets.append(f'ytsearch{n_search}:{product.get("brand", "")} {product.get("model", "")} {product.get("name", "")} review demo')
        for t in targets:
            try:
                _run(["yt-dlp", "-q", "--no-warnings", "--match-filter", f"duration < {max_minutes * 60}",
                      "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]", "--merge-output-format", "mp4",
                      "--restrict-filenames", "-o", os.path.join(vdir, "%(id)s_%(channel).30s.%(ext)s"), t])
            except Exception as e:
                report.append(f"could not download {t[:80]}: {str(e)[:80]}")
        return sorted(glob.glob(os.path.join(vdir, "*.mp4")))

    # ------------------------------------------------------------------ tag frames

    def _tag_video(self, call, name, model, video, out, step):
        fdir = os.path.join(out, "_frames", core.slug(os.path.basename(video))[:40])
        os.makedirs(fdir, exist_ok=True)
        _run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"fps=1/{step},scale=640:-2", "-q:v", "4", os.path.join(fdir, "%05d.jpg")])
        frames = [(round((i + 0.5) * step, 1), p) for i, p in enumerate(sorted(glob.glob(os.path.join(fdir, "*.jpg"))))]
        batches = [frames[i:i + 16] for i in range(0, len(frames), 16)]

        def tag(batch):
            prompt = FRAMES.format(name=name, n=len(batch), times=", ".join(str(t) for t, _ in batch), actions=json.dumps(ACTIONS))
            for _ in range(2):
                try:
                    got = core.parse_json(core.llm(call, model, prompt, [core.data_url(Image.open(p), None, 75) for _, p in batch], max_tokens=6000))["frames"]
                    if len(got) == len(batch):
                        for g, (t, _) in zip(got, batch):
                            g["t"] = t
                        return got
                except Exception:
                    pass
            return []

        with ThreadPoolExecutor(4) as ex:
            return [f for part in ex.map(tag, batches) for f in part]

    # ------------------------------------------------------------------ segments, clips, edits

    @staticmethod
    def _segments(cache, step):
        segs = []
        for v in cache.values():
            cur = None
            for f in v["frames"]:
                ok = f.get("action") in ACTIONS[:-1] and f.get("sharp", True)
                if cur and ok and f["action"] == cur["action"] and f["t"] - cur["end"] <= step * 1.5:
                    cur["end"] = f["t"]
                    cur["frames"].append(f)
                else:
                    if cur:
                        segs.append(cur)
                    cur = {"file": v["file"], "action": f["action"], "start": f["t"], "end": f["t"], "frames": [f]} if ok else None
            if cur:
                segs.append(cur)
        for s in segs:
            s["len"] = s["end"] - s["start"] + step
            s["overlay"] = sum(bool(f.get("overlay")) for f in s["frames"]) / len(s["frames"])
            s["score"] = min(s["len"], 8.0) * (1 - 0.6 * s["overlay"]) * (0.5 if s["action"] == ACTIONS[5] else 1)
        return [s for s in segs if s["len"] >= 3.0]

    def _cut(self, folder, out, segs, step):
        for old in glob.glob(os.path.join(out, "ref_*.*")):
            os.remove(old)
        index, n = [], 0
        for action in ACTIONS[:-1]:
            picked = []
            for s in sorted([s for s in segs if s["action"] == action], key=lambda s: -s["score"]):
                whats = [f.get("what", "") for f in s["frames"]]
                what = max(set(whats), key=whats.count)
                if any(p["what"] == what and p["file"] == s["file"] for p in picked):      # same shot again: skip
                    continue
                s["what"] = what
                picked.append(s)
                if len(picked) == 4:
                    break
            for s in picked:
                n += 1
                start, dur = max(0, s["start"] - step / 2), min(s["len"], 8.0)
                clip = f'ref_{n:02d}_{core.slug(action)[:28].lower()}.mp4'
                src = os.path.join(folder, "videos", s["file"])
                _run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", src, "-t", str(dur), "-an", "-c:v", "libx264", "-crf", "18",
                      "-pix_fmt", "yuv420p", os.path.join(out, clip)])
                _run(["ffmpeg", "-v", "error", "-y", "-ss", str(start + dur / 2), "-i", src, "-frames:v", "1", "-q:v", "3", os.path.join(out, clip[:-4] + ".jpg")])
                people = [f.get("people", "none") for f in s["frames"]]
                index.append({"clip": clip, "action": action, "what": s["what"], "seconds": round(dur, 1), "people": max(set(people), key=people.count),
                              "shot": s["frames"][len(s["frames"]) // 2].get("shot"), "overlay": s["overlay"] > 0.3, "source": s["file"],
                              "source_start": round(start, 1)})
        json.dump(index, open(os.path.join(out, "index.json"), "w", encoding="utf-8"), indent=1)
        return index

    def _edit(self, out, picks, name, total):
        if not picks:
            return None
        each = max(2.0, min(4.0, total / len(picks)))
        parts = []
        for i, c in enumerate(picks):
            dur = min(each, c["seconds"])
            part = f"_part_{name}_{i}.mp4"
            _run(["ffmpeg", "-v", "error", "-y", "-ss", str(max(0, (c["seconds"] - dur) / 2)), "-i", c["clip"], "-t", str(dur), "-vf", VF, "-an",
                  "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", part], cwd=out)
            parts.append(part)
        open(os.path.join(out, f"_list_{name}.txt"), "w").write("".join(f"file '{p}'\n" for p in parts))
        _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"_list_{name}.txt", "-c", "copy", f"edit_{name}.mp4"], cwd=out)
        for p in parts + [f"_list_{name}.txt"]:
            os.remove(os.path.join(out, p))
        return {"file": f"edit_{name}.mp4", "seconds": round(sum(min(each, c["seconds"]) for c in picks), 1),
                "shots": [{"clip": c["clip"], "action": c["action"], "what": c["what"]} for c in picks]}

    # ------------------------------------------------------------------ node entry

    def find(self, product_dir, api_key, video_urls="", search_videos=3, max_video_minutes=12, seconds_between_frames=2.0,
             edit_seconds=20, tag_model="google/gemini-2.5-flash"):
        folder = product_dir.strip().strip('"')
        if not os.path.isfile(os.path.join(folder, "product.json")):
            raise ValueError(f"'{folder}' is not a product folder (product.json is missing). Run Product Research first.")
        for tool in ("ffmpeg", "yt-dlp"):
            if not shutil.which(tool):
                raise RuntimeError(f"{tool} was not found on PATH. Install it to use Product Video Reference.")
        client = self.get_client(api_key)
        call = lambda endpoint, arguments: self.call_api(client, endpoint, arguments)   # noqa: E731
        product = json.load(open(os.path.join(folder, "product.json"), encoding="utf-8"))
        name = f'{product.get("brand", "")} {product.get("model", "")} {product.get("name", "")}'.strip()
        out = os.path.join(folder, "video_refs")
        os.makedirs(out, exist_ok=True)
        report = []
        urls = [u.strip() for u in video_urls.splitlines() if u.strip().startswith("http")]
        videos = self._download(folder, product, urls, int(search_videos), int(max_video_minutes), report)
        if not videos:
            raise ValueError("No video could be found or downloaded. Give links in video_urls.")
        step = float(seconds_between_frames)
        cache_path = os.path.join(out, "frame_tags.json")
        cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
        for video in videos:
            key = os.path.basename(video)
            if key not in cache and key[:11] not in cache:
                cache[key] = {"file": key, "frames": self._tag_video(call, name, tag_model, video, out, step)}
                json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=0)
            report.append(f"{key[:60]}: {len((cache.get(key) or cache.get(key[:11]))['frames'])} frames read")
        index = self._cut(folder, out, self._segments(cache, step), step)
        by = {}
        for c in index:
            by.setdefault(c["action"], []).append(c)
        control = lambda c: not re.search(r"dial|knob|button|turn|press|lid|switch", c["what"].lower())   # noqa: E731
        for action, clips in by.items():
            clips.sort(key=lambda c: (c["overlay"], control(c) if action == ACTIONS[0] else False, -c["seconds"]))
        a = [by[x][0] for x in ORDER_A if by.get(x)]
        used = {c["clip"] for c in a}
        b = [c for c in (next((c for c in by.get(x, []) if c["clip"] not in used), None) for x in ORDER_B) if c]
        edits = [e for e in (self._edit(out, a, "A_overview", edit_seconds), self._edit(out, b, "B_actions", edit_seconds)) if e]
        json.dump(edits, open(os.path.join(out, "edits.json"), "w", encoding="utf-8"), indent=1)
        report.append(f"{len(index)} clips in {len(by)} action groups; edits: " + ", ".join(f'{e["file"]} ({e["seconds"]} s)' for e in edits))

        strip = Image.new("RGB", (640 * max(1, len(a)), 360 + 60), (24, 24, 24))
        d = ImageDraw.Draw(strip)
        for i, c in enumerate(a):
            strip.paste(Image.open(os.path.join(out, c["clip"][:-4] + ".jpg")).convert("RGB").resize((640, 360)), (i * 640, 0))
            d.text((i * 640 + 10, 372), f'{i + 1}. {c["what"][:44]}', fill=(235, 235, 235), font=core.font(24))
        tensor = torch.from_numpy(np.array(strip).astype(np.float32) / 255.0).unsqueeze(0)
        text = "\n".join(report)
        open(os.path.join(out, "report.txt"), "w", encoding="utf-8").write(text + "\n")
        return (os.path.join(out, edits[0]["file"]) if edits else "", out, tensor, text)
