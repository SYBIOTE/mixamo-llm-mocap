"""Review videos for an SMPL-X clip — rendered headlessly, no Blender UI or MCP.

  python pipeline/render_review.py --spec action_specs/<motion>.json --clip clips/<clip> \
         [--legacy clips/<legacy_clip>] [--scene ybot_rest.blend] [--views front,three_quarter]

Writes into the clip folder, per view (front, three_quarter):
  review_<view>.mp4   a 2x2 grid, time-aligned and labelled:
                        SOURCE plate          | OVERLAY
                        BEFORE (legacy clip)  | AFTER (this clip)
                      OVERLAY is this clip rendered through the PLATE'S OWN
                      CAMERA (recovered by retarget_smplx.py) and composited
                      over the video: where the character and the performer
                      disagree, you see it. Without --legacy the bottom row
                      shows this clip from two angles.
Frames are rendered with bl_motion.py in background Blender and removed
afterwards (--keep-frames to keep them). The legacy clip, if given, is
re-keyed from its curves.json onto the same rig and rendered with the
same camera and look, so the two differ by their motion only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import av
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BLENDER_CANDIDATES = [
    os.environ.get("BLENDER", ""),
    r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "blender",
]


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def blender_exe(arg=None) -> str:
    for c in ([arg] if arg else []) + BLENDER_CANDIDATES:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    raise SystemExit("Blender not found — pass --blender or set BLENDER")


def render(blender, scene, src_flag, src_path, view, out_dir, extra=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [blender, "-b", str(scene), "--python", str(REPO / "pipeline" / "bl_motion.py"), "--",
           "render", src_flag, str(src_path), "--view", view, "--out", str(out_dir), *extra]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not any(out_dir.glob("f*.png")):
        sys.stderr.write(res.stdout[-3000:] + res.stderr[-3000:])
        raise SystemExit(f"render failed: {view} {src_path}")
    return sorted(out_dir.glob("f*.png"))


def read_plate(video: Path):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
    return frames, fps


def label(img, text, sub=None):
    """Title (+ subtitle) on a translucent dark band, legible on any frame."""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    sw = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] if sub else 0
    x1, y1 = 10 + max(tw, sw) + 20, (68 if sub else 46)
    band = img[8:y1, 8:x1].astype(np.float32)
    img[8:y1, 8:x1] = (band * 0.35 + np.array([28, 24, 22], np.float32) * 0.65).astype(np.uint8)
    cv2.putText(img, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 215, 215), 1, cv2.LINE_AA)
    return img


def fit_h(img, h):
    return cv2.resize(img, (int(round(img.shape[1] * h / img.shape[0])) // 2 * 2, h),
                      interpolation=cv2.INTER_AREA)


def encode(path: Path, frames_iter, fps: int, crf: int = 26):
    first = next(frames_iter)
    h, w = first.shape[:2]
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "slow", "movflags": "+faststart"}

    def put(img):
        fr = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")
        for pkt in stream.encode(fr):
            container.mux(pkt)

    put(first)
    for img in frames_iter:
        put(img)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--clip", required=True, help="clip dir holding motion.npz")
    ap.add_argument("--legacy", help="legacy clip dir (curves.json) to show next to it")
    ap.add_argument("--scene", default="ybot_rest.blend", help="the rig's rest scene (.blend)")
    ap.add_argument("--video", help="source plate (default: first .mp4 next to the spec's landmarks)")
    ap.add_argument("--views", default="front,three_quarter")
    ap.add_argument("--blender")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--reuse-frames", action="store_true", help="skip rendering where frames exist")
    args = ap.parse_args()

    spec = json.loads(rpath(args.spec).read_text(encoding="utf-8"))
    clip = rpath(args.clip)
    motion = clip / "motion.npz"
    if not motion.exists():
        raise SystemExit(f"{motion} missing — run retarget_smplx.py first")
    video = rpath(args.video) if args.video else sorted(rpath(spec["landmarks"]).parent.glob("*.mp4"))[0]
    blender = blender_exe(args.blender)
    scene = rpath(args.scene)
    plate, plate_fps = read_plate(video)
    fps = int(round(float(np.load(motion)["fps"])))
    work = clip / "_frames"
    views = [v for v in args.views.split(",") if v]

    def frames_of(flag, path, view, name):
        d = work / name
        if args.reuse_frames and any(d.glob("f*.png")):
            return sorted(d.glob("f*.png"))
        shutil.rmtree(d, ignore_errors=True)
        return render(blender, scene, flag, path, view, d)

    cam = frames_of("--motion", motion, "camera", "new_camera")
    new = {v: frames_of("--motion", motion, v, f"new_{v}") for v in views}
    old = ({v: frames_of("--curves", rpath(args.legacy) / "curves.json", v, f"old_{v}") for v in views}
           if args.legacy else None)
    W, H = 640, 360                                   # one cell of the 2x2 grid

    def plate_at(i):
        return plate[min(len(plate) - 1, int(round(i / fps * plate_fps)))]

    def cell(img):
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    def overlay(i):
        src = plate_at(i).astype(np.float32)
        rgba = cv2.imread(str(cam[min(i, len(cam) - 1)]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if rgba.shape[:2] != src.shape[:2]:
            rgba = cv2.resize(rgba, (src.shape[1], src.shape[0]))
        a = rgba[..., 3:4] / 255.0 * 0.62
        return (src * (1 - a) + rgba[..., :3] * a).astype(np.uint8)

    def grid(view):
        for i in range(len(cam)):
            tl = label(cell(plate_at(i)), "SOURCE VIDEO")
            tr = label(cell(overlay(i)), "OVERLAY", "new retarget seen by the plate's camera")
            if old is not None:
                bl = label(cell(cv2.imread(str(old[view][min(i, len(old[view]) - 1)]))),
                           "BEFORE", "landmark lift + FK aim")
                br = label(cell(cv2.imread(str(new[view][min(i, len(new[view]) - 1)]))),
                           "AFTER", "SMPL-X rotation retarget")
            else:
                other = [v for v in views if v != view] or [view]
                bl = label(cell(cv2.imread(str(new[view][min(i, len(new[view]) - 1)]))),
                           "AFTER", view.replace("_", "-"))
                br = label(cell(cv2.imread(str(new[other[0]][min(i, len(new[other[0]]) - 1)]))),
                           "AFTER", other[0].replace("_", "-"))
            g = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
            g[H - 1:H + 1, :] = 24
            g[:, W - 1:W + 1] = 24
            yield g

    for view in views:
        out = clip / f"review_{view}.mp4"
        encode(out, grid(view), fps, crf=27)
        print("wrote", out)
        if old is None:
            break                                     # one grid already shows both views

    if not args.keep_frames:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
