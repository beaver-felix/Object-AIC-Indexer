# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
#     "pyarrow",
#     "numpy",
#     "opencv-python-headless",
#     "ultralytics",
#     "scenedetect",
#     "orjson",
# ]
# ///
"""AIC_indexer.py: Multi-GPU Video Indexer for VISIONE obj-idx Parquet."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Generator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import orjson
import polars as pl
import pyarrow.parquet as pq

VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".ts",
    ".mts",
    ".m2ts",
}

COLOR_RANGES = [
    ("black", (0, 0, 0), (180, 255, 45)),
    ("white", (0, 0, 200), (180, 30, 255)),
    ("grey", (0, 0, 45), (180, 40, 200)),
    ("red", (0, 70, 50), (10, 255, 255)),
    ("red", (170, 70, 50), (180, 255, 255)),
    ("orange", (11, 70, 50), (25, 255, 255)),
    ("yellow", (26, 70, 50), (35, 255, 255)),
    ("green", (36, 70, 50), (85, 255, 255)),
    ("blue", (86, 70, 50), (125, 255, 255)),
    ("purple", (126, 70, 50), (145, 255, 255)),
    ("pink", (146, 70, 50), (169, 255, 255)),
]

SCHEMA = {
    "video_id": pl.Utf8,
    "time": pl.Float64,
    "frame_idx": pl.Int64,
    "start_time": pl.Float64,
    "end_time": pl.Float64,
    "start_frame": pl.Int64,
    "end_frame": pl.Int64,
    "fps": pl.Float32,
    "width": pl.Int32,
    "height": pl.Int32,
    "objects": pl.List(pl.Utf8),
    "scores": pl.List(pl.Float32),
    "boxes": pl.List(pl.List(pl.Float32)),
    "labels": pl.List(pl.Utf8),
    "txt": pl.Utf8,
    "objects_str": pl.Utf8,
    "colors": pl.List(pl.Utf8),
    "is_monochrome": pl.Boolean,
    "video_path": pl.Utf8,
}

FLUSH_EVERY = 500  # flush records from worker every N to prevent OOM
DEFAULT_TIMEOUT = 300  # inactivity/stall watchdog in seconds without progress
DEFAULT_LARGE_FILE_GB = (
    0.0  # 0 = disabled: preserve strict stride for frame retrieval accuracy
)

# Codecs that OpenCV can't HW-decode reliably on headless platforms — route to ffmpeg pipe
FFMPEG_ONLY_CODECS = {"av1", "vp9", "vp8", "av1_cuvid", "libdav1d"}


def format_eta(seconds: float) -> str:
    """Format duration in seconds to a human-readable ETA string."""
    if seconds < 0:
        return "0s"
    sec = round(seconds)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        m, s = divmod(sec, 60)
        return f"{m}m {s:02d}s"
    h = sec // 3600
    m, s = divmod(sec % 3600, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def compute_overall_eta(
    comp_bytes: int,
    tot_bytes: int,
    n_vids: int,
    total_vids: int,
    elapsed_s: float,
) -> str:
    """Compute overall job remaining time estimate across all workers."""
    if elapsed_s < 1.0:
        return "--"
    rem_bytes = max(0, tot_bytes - comp_bytes)
    rem_vids = max(0, total_vids - n_vids)
    if rem_bytes == 0 and rem_vids == 0:
        return "0s"
    if comp_bytes > 0 and rem_bytes > 0:
        byte_rate = comp_bytes / elapsed_s
        return format_eta(rem_bytes / byte_rate)
    if n_vids > 0 and rem_vids > 0:
        vid_rate = n_vids / elapsed_s
        return format_eta(rem_vids / vid_rate)
    return "--"


def probe_video_ffprobe(video_path: Path) -> tuple[int, int, float, int, str]:
    """Probe video metadata via ffprobe JSON. Returns (width, height, fps, nb_frames, codec_name)."""
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_frames,duration:format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        data = orjson.loads(r.stdout)
        streams = data.get("streams", [])
        if not streams:
            return 0, 0, 25.0, 0, "unknown"
        s = streams[0]
        codec = str(s.get("codec_name", "unknown")).lower()
        w = int(s.get("width", 0))
        h = int(s.get("height", 0))
        fps_str = str(s.get("r_frame_rate", "25/1"))
        if "/" in fps_str:
            fps_parts = fps_str.split("/")
            fps = (
                float(fps_parts[0]) / float(fps_parts[1])
                if len(fps_parts) == 2 and float(fps_parts[1]) > 0
                else float(fps_parts[0])
            )
        else:
            fps = float(fps_str)
        nb_str = str(s.get("nb_frames", "0"))
        nb = int(nb_str) if nb_str.isdigit() else 0
        if nb <= 0:
            dur_val = s.get("duration") or data.get("format", {}).get("duration", "0")
            dur = float(dur_val) if dur_val else 0.0
            if dur > 0 and fps > 0:
                nb = round(dur * fps)
        return w, h, fps, nb, codec
    except (OSError, subprocess.SubprocessError, ValueError, orjson.JSONDecodeError):
        return 0, 0, 25.0, 0, "unknown"


def needs_ffmpeg_decode(video_path: Path) -> tuple[bool, str, int, int, float, int]:
    """Check if video requires ffmpeg decode path. Returns (use_ffmpeg, codec, w, h, fps, nb_frames)."""
    w, h, fps, nb, codec = probe_video_ffprobe(video_path)
    is_problematic = any(c in codec for c in FFMPEG_ONLY_CODECS)
    return is_problematic, codec, w, h, fps, nb


def stream_strided_frames_ffmpeg(
    video_path: Path,
    stride: int = 10,
    batch_size: int = 16,
    w: int = 0,
    h: int = 0,
    fps: float = 25.0,
    stop_event: threading.Event | None = None,
    decoder_state: dict[str, Any] | None = None,
) -> Generator[
    tuple[list[dict[str, Any]], list[np.ndarray], float, int, int], None, None
]:
    """Extract strided frames via ffmpeg raw pipe — handles AV1, VP9, etc."""
    if w <= 0 or h <= 0:
        w, h, fps, _, _ = probe_video_ffprobe(video_path)
    if w <= 0 or h <= 0:
        return
    stride = max(1, int(stride))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "0",
        "-i",
        str(video_path),
        "-vf",
        f"select=not(mod(n\\,{stride}))",
        "-vsync",
        "vfr",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4
    )
    if decoder_state is not None:
        decoder_state["proc"] = proc
        decoder_state["pid"] = proc.pid
        decoder_state["running"] = True
        decoder_state["finished"] = False

    frame_bytes = w * h * 3
    idx = 0
    batch_meta: list[dict[str, Any]] = []
    batch_frames: list[np.ndarray] = []

    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            raw = b""
            while len(raw) < frame_bytes:
                if stop_event and stop_event.is_set():
                    break
                chunk = proc.stdout.read(frame_bytes - len(raw))  # type: ignore[union-attr]
                if not chunk:
                    break
                raw += chunk
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
            src_idx = idx * stride
            t = round(src_idx / fps, 3)
            meta = {
                "start_frame": max(0, src_idx - stride // 2),
                "end_frame": src_idx + stride // 2,
                "start_time": round(max(0.0, (src_idx - stride // 2) / fps), 3),
                "end_time": round((src_idx + stride // 2) / fps, 3),
                "middle_frame": src_idx,
                "middle_time": t,
            }
            batch_meta.append(meta)
            batch_frames.append(frame)
            idx += 1
            if len(batch_frames) >= batch_size:
                yield batch_meta, batch_frames, fps, w, h
                batch_meta = []
                batch_frames = []
        if batch_frames:
            yield batch_meta, batch_frames, fps, w, h
    finally:
        if decoder_state is not None:
            decoder_state["running"] = False
            decoder_state["finished"] = True
        try:
            if proc.stdout:
                proc.stdout.close()
            proc.terminate()
            proc.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            try:
                proc.kill()
                proc.wait()
            except (OSError, subprocess.SubprocessError):
                pass


def encode_positional_boxes(
    boxes: list[list[float]], labels: list[str], n: int = 7, tol: float = 0.1
) -> str:
    """Encode bounding boxes into VISIONE 7x7 spatial tokens (e.g. '2ccar')."""
    tokens = []
    dt = tol / n
    for (y0, x0, y1, x1), label in zip(boxes, labels):
        lbl = label.lower().replace(" ", "_")
        c0, r0 = (
            math.floor((max(0.0, x0) + dt) * n),
            math.floor((max(0.0, y0) + dt) * n),
        )
        c1, r1 = (
            math.floor((min(1.0, x1) - dt) * n),
            math.floor((min(1.0, y1) - dt) * n),
        )
        tokens.extend(
            f"{r}{chr(97 + c)}{lbl}"
            for r in range(max(0, r0), min(n, r1 + 1))
            for c in range(max(0, c0), min(n, c1 + 1))
        )
    return " ".join(sorted(tokens))


def encode_object_counts(labels: list[str], scores: list[float]) -> str:
    """Encode object counts into VISIONE surrogate text format (e.g. '4wcperson1|6')."""
    counts: Counter[str] = Counter()
    tokens = []
    for label, score in zip(labels, scores):
        lbl = label.lower().replace(" ", "_")
        counts[lbl] += 1
        tokens.append(f"4wc{lbl}{counts[lbl]}|{max(1, int(5.0 * score + 2.0))}")
    return " ".join(sorted(tokens))


def analyze_frame_colors(frame_bgr: np.ndarray) -> tuple[bool, list[str]]:
    """Analyze frame color palette and detect monochrome/grayscale."""
    small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    if float(np.mean(hsv[:, :, 1])) < 22.0:
        return True, ["monochrome"]
    colors = []
    for name, low, high in COLOR_RANGES:
        mask = cv2.inRange(
            hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)
        )
        if cv2.countNonZero(mask) >= 327 and name not in colors:  # 64*64*0.08 = 327.68
            colors.append(name)
    return False, colors


def stream_scene_keyframes(
    video_path: Path,
    max_scene_len_sec: float = 10.0,
    thresh: float = 3.0,
    batch_size: int = 16,
    stop_event: threading.Event | None = None,
) -> Generator[
    tuple[list[dict[str, Any]], list[np.ndarray], float, int, int], None, None
]:
    """Single-pass online scene detector and keyframe streamer.

    Detects shot transitions on-the-fly and yields keyframes as the video is read,
    avoiding decoding the entire video twice.
    """
    from scenedetect import FrameTimecode
    from scenedetect.detectors import AdaptiveDetector

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = AdaptiveDetector(adaptive_threshold=thresh)
    shot_start_frame = 0
    shot_start_time = 0.0
    shot_frames: list[np.ndarray] = []

    def make_shot_meta(sf: int, ef: int, st: float, et: float) -> dict[str, Any]:
        mf = (sf + ef) // 2
        mt = round(mf / fps, 3)
        return {
            "start_frame": sf,
            "end_frame": ef,
            "start_time": round(st, 3),
            "end_time": round(et, 3),
            "middle_frame": mf,
            "middle_time": mt,
        }

    batch_meta: list[dict[str, Any]] = []
    batch_frames: list[np.ndarray] = []
    idx = 0

    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            tc = FrameTimecode(timecode=idx, fps=fps)
            cuts = detector.process_frame(tc, frame)
            cur_time = round(idx / fps, 3)

            if cuts:
                for cut_tc in cuts:
                    cut_frame = cut_tc.frame_num
                    if cut_frame > shot_start_frame:
                        end_f = cut_frame - 1
                        end_t = round(end_f / fps, 3)
                        meta = make_shot_meta(
                            shot_start_frame, end_f, shot_start_time, end_t
                        )
                        m_idx = meta["middle_frame"] - shot_start_frame
                        if 0 <= m_idx < len(shot_frames):
                            batch_meta.append(meta)
                            batch_frames.append(shot_frames[m_idx])
                            if len(batch_frames) >= batch_size:
                                yield batch_meta, batch_frames, fps, w, h
                                batch_meta = []
                                batch_frames = []
                        shot_frames = shot_frames[cut_frame - shot_start_frame :]
                        shot_start_frame = cut_frame
                        shot_start_time = round(shot_start_frame / fps, 3)

            shot_frames.append(frame)
            dur = cur_time - shot_start_time
            if max_scene_len_sec > 0 and dur >= max_scene_len_sec:
                end_f = idx
                end_t = cur_time
                meta = make_shot_meta(
                    shot_start_frame, end_f, shot_start_time, end_t
                )
                m_idx = meta["middle_frame"] - shot_start_frame
                if 0 <= m_idx < len(shot_frames):
                    batch_meta.append(meta)
                    batch_frames.append(shot_frames[m_idx])
                    if len(batch_frames) >= batch_size:
                        yield batch_meta, batch_frames, fps, w, h
                        batch_meta = []
                        batch_frames = []
                shot_frames = []
                shot_start_frame = idx + 1
                shot_start_time = round(shot_start_frame / fps, 3)

            idx += 1

        final_cuts = detector.post_process(
            FrameTimecode(timecode=max(0, idx - 1), fps=fps)
        )
        if final_cuts:
            for cut_tc in final_cuts:
                cut_frame = cut_tc.frame_num
                if cut_frame > shot_start_frame:
                    end_f = cut_frame - 1
                    end_t = round(end_f / fps, 3)
                    meta = make_shot_meta(
                        shot_start_frame, end_f, shot_start_time, end_t
                    )
                    m_idx = meta["middle_frame"] - shot_start_frame
                    if 0 <= m_idx < len(shot_frames):
                        batch_meta.append(meta)
                        batch_frames.append(shot_frames[m_idx])
                        if len(batch_frames) >= batch_size:
                            yield batch_meta, batch_frames, fps, w, h
                            batch_meta = []
                            batch_frames = []
                    shot_frames = shot_frames[cut_frame - shot_start_frame :]
                    shot_start_frame = cut_frame
                    shot_start_time = round(shot_start_frame / fps, 3)

        if shot_start_frame < idx:
            end_f = idx - 1
            end_t = round(end_f / fps, 3)
            meta = make_shot_meta(shot_start_frame, end_f, shot_start_time, end_t)
            m_idx = meta["middle_frame"] - shot_start_frame
            if 0 <= m_idx < len(shot_frames):
                batch_meta.append(meta)
                batch_frames.append(shot_frames[m_idx])

        if batch_frames:
            yield batch_meta, batch_frames, fps, w, h
    finally:
        cap.release()


def stream_strided_frames(
    video_path: Path,
    stride: int = 10,
    batch_size: int = 16,
) -> Generator[
    tuple[list[dict[str, Any]], list[np.ndarray], float, int, int], None, None
]:
    """Yield batches of strided frames via OpenCV. For unsupported codecs, use stream_strided_frames_ffmpeg directly."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(stride))
    idx = 0
    batch_meta: list[dict[str, Any]] = []
    batch_frames: list[np.ndarray] = []

    try:
        while True:
            if idx % stride == 0:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = float(ms / 1000.0) if ms > 0 else float(idx / fps)
                meta = {
                    "start_frame": max(0, idx - stride // 2),
                    "end_frame": idx + stride // 2,
                    "start_time": round(max(0.0, (idx - stride // 2) / fps), 3),
                    "end_time": round((idx + stride // 2) / fps, 3),
                    "middle_frame": idx,
                    "middle_time": round(t, 3),
                }
                batch_meta.append(meta)
                batch_frames.append(frame)
                if len(batch_frames) >= batch_size:
                    yield batch_meta, batch_frames, fps, w, h
                    batch_meta = []
                    batch_frames = []
            elif not cap.grab():
                break
            idx += 1
        if batch_frames:
            yield batch_meta, batch_frames, fps, w, h
    finally:
        cap.release()


def _process_single_video(
    video_file: Path,
    model: Any,
    gpu_id: str,
    conf: float,
    batch_size: int,
    use_scenes: bool,
    stride: int,
    max_scene_len: float,
    max_det: int,
    large_threshold_gb: float,
    res_q: mp.Queue,
    prog_q: mp.Queue,
    worker_id: int,
    progress_state: dict[str, Any] | None = None,
) -> tuple[int, int, float]:
    """Process one video — called from worker, running inside stall watchdog thread."""
    fsize = video_file.stat().st_size
    file_gb = fsize / (1024**3)
    file_mb = fsize / (1024**2)
    vid = video_file.stem
    t0 = time.perf_counter()

    use_ffmpeg, codec, w, h, fps, nb_frames = needs_ffmpeg_decode(video_file)

    effective_stride = stride
    force_strided = False
    if large_threshold_gb > 0.0 and file_gb > large_threshold_gb:
        scale = max(1, int(file_gb / large_threshold_gb))
        effective_stride = stride * scale
        force_strided = True

    tot_expected_kfs = math.ceil(nb_frames / effective_stride) if nb_frames > 0 else 0
    engine = (
        "ffmpeg pipe"
        if use_ffmpeg
        else (
            "PySceneDetect"
            if (use_scenes and not force_strided)
            else f"cv2 stride={effective_stride}"
        )
    )
    print(
        f">  [{gpu_id}] [{video_file.name}] Start | {file_mb:.1f} MB | Codec: {codec} | Res: {w}x{h}@{fps:.1f}fps | Engine: {engine}",
        flush=True,
    )

    stop_evt = progress_state["stop_event"] if progress_state else None

    decoder_state: dict[str, Any] = {
        "proc": None,
        "pid": None,
        "running": False,
        "finished": False,
    }

    if use_ffmpeg:
        batch_gen = stream_strided_frames_ffmpeg(
            video_file,
            stride=effective_stride,
            batch_size=batch_size,
            w=w,
            h=h,
            fps=fps,
            stop_event=stop_evt,
            decoder_state=decoder_state,
        )
    elif use_scenes and not force_strided:
        try:
            batch_gen = stream_scene_keyframes(
                video_file,
                max_scene_len_sec=max_scene_len,
                batch_size=batch_size,
                stop_event=stop_evt,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f">  [{gpu_id}] [{video_file.name}] Scene detect fallback ({e}), using strided...",
                flush=True,
            )
            batch_gen = stream_strided_frames(
                video_file, stride=effective_stride, batch_size=batch_size
            )
    else:
        batch_gen = stream_strided_frames(
            video_file, stride=effective_stride, batch_size=batch_size
        )

    # Asynchronous prefetch queue: decouples ffmpeg/cv2 decoding from YOLO inference
    prefetch_q: queue.Queue[Any] = queue.Queue(maxsize=4)
    decoder_finished = threading.Event()
    reader_stop = threading.Event()
    decoder_err: list[Any] = [None]

    def _reader_thread() -> None:
        try:
            for item in batch_gen:
                if (stop_evt and stop_evt.is_set()) or reader_stop.is_set():
                    break
                while not ((stop_evt and stop_evt.is_set()) or reader_stop.is_set()):
                    try:
                        prefetch_q.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:  # noqa: BLE001
            decoder_err[0] = exc
        finally:
            decoder_finished.set()

    reader = threading.Thread(target=_reader_thread, daemon=True)
    reader.start()

    records: list[dict[str, Any]] = []
    n_det = 0
    tot_kfs = 0
    last_log_time = t0

    try:
        while True:
            if stop_evt and stop_evt.is_set():
                print(
                    f">  [{gpu_id}] [{video_file.name}] Stop signal received, aborting processing.",
                    flush=True,
                )
                break

            try:
                item = prefetch_q.get(timeout=0.2)
            except queue.Empty:
                if decoder_finished.is_set() and prefetch_q.empty():
                    if decoder_err[0]:
                        raise decoder_err[0]
                    break
                continue

            batch_meta, batch_frames, b_fps, b_w, b_h = item
            tot_kfs += len(batch_frames)
            is_decoding = not decoder_finished.is_set()
            if progress_state:
                progress_state["last_active"] = time.perf_counter()
                progress_state["kf_count"] = tot_kfs
                progress_state["ffmpeg_running"] = is_decoding

            results = model.predict(
                batch_frames,
                conf=conf,
                device=gpu_id,
                verbose=False,
                max_det=max_det,
                agnostic_nms=True,
            )
            for meta, img, det in zip(batch_meta, batch_frames, results):
                mono, colors = analyze_frame_colors(img)
                b_lbls, b_scs, b_boxes = [], [], []
                if det.boxes is not None and len(det.boxes) > 0:
                    ih, iw = img.shape[:2]
                    for c_val, cl_id, box in zip(
                        det.boxes.conf.cpu().numpy(),
                        det.boxes.cls.cpu().numpy().astype(int),
                        det.boxes.xyxy.cpu().numpy(),
                    ):
                        b_lbls.append(str(det.names.get(cl_id, f"obj_{cl_id}")))
                        b_scs.append(float(c_val))
                        b_boxes.append(
                            [
                                float(box[1] / ih),
                                float(box[0] / iw),
                                float(box[3] / ih),
                                float(box[2] / iw),
                            ]
                        )
                n_det += len(b_lbls)
                records.append(
                    {
                        "video_id": vid,
                        "time": float(meta["middle_time"]),
                        "frame_idx": int(meta["middle_frame"]),
                        "start_time": float(meta["start_time"]),
                        "end_time": float(meta["end_time"]),
                        "start_frame": int(meta["start_frame"]),
                        "end_frame": int(meta["end_frame"]),
                        "fps": float(round(b_fps, 2)),
                        "width": int(b_w),
                        "height": int(b_h),
                        "objects": sorted(set(b_lbls)),
                        "scores": b_scs,
                        "boxes": b_boxes,
                        "labels": b_lbls,
                        "txt": encode_positional_boxes(b_boxes, b_lbls),
                        "objects_str": encode_object_counts(b_lbls, b_scs),
                        "colors": colors,
                        "is_monochrome": mono,
                        "video_path": str(video_file),
                    }
                )
                if len(records) >= FLUSH_EVERY:
                    res_q.put(records)
                    records = []

            now = time.perf_counter()
            if now - last_log_time >= 5.0:
                elapsed = now - t0
                cur_fps = tot_kfs / max(1e-4, elapsed)
                if use_ffmpeg:
                    dec_tag = "ffmpeg: decoding" if is_decoding else "ffmpeg: finished"
                else:
                    dec_tag = "cv2: reading" if is_decoding else "cv2: finished"

                if tot_expected_kfs > 0:
                    pct = min(100.0, (tot_kfs / tot_expected_kfs) * 100.0)
                    rem_kfs = max(0, tot_expected_kfs - tot_kfs)
                    vid_eta_s = rem_kfs / max(0.1, cur_fps)
                    vid_eta_str = format_eta(vid_eta_s)
                    kf_msg = f"{tot_kfs}/{tot_expected_kfs} kf ({pct:>5.1f}%)"
                    eta_msg = f" | ETA: {vid_eta_str}"
                else:
                    vid_eta_str = "--"
                    kf_msg = f"{tot_kfs} kf"
                    eta_msg = ""

                print(
                    f">  [{gpu_id}] [{video_file.name}] {kf_msg} | {dec_tag} | {cur_fps:.1f} fps | {n_det} objs | elapsed: {int(elapsed)}s{eta_msg}",
                    flush=True,
                )
                prog_q.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "video_name": video_file.name,
                        "status": "progress",
                        "num_keyframes": tot_kfs,
                        "tot_expected_kfs": tot_expected_kfs,
                        "fps": round(cur_fps, 1),
                        "num_detections": n_det,
                        "vid_eta": vid_eta_str,
                        "decoder_status": dec_tag,
                    }
                )
                last_log_time = now
    finally:
        reader_stop.set()
        while not prefetch_q.empty():
            try:
                prefetch_q.get_nowait()
            except queue.Empty:
                break
        reader.join(timeout=2.0)

    if records:
        res_q.put(records)

    el = max(1e-4, time.perf_counter() - t0)
    return tot_kfs, n_det, el


def gpu_worker_process(
    worker_id: int,
    gpu_id: str,
    task_q: mp.Queue,
    res_q: mp.Queue,
    prog_q: mp.Queue,
    model_name: str,
    conf: float,
    batch_size: int,
    use_scenes: bool,
    stride: int,
    max_scene_len: float,
    max_det: int = 300,
    timeout: float = DEFAULT_TIMEOUT,
    large_threshold_gb: float = DEFAULT_LARGE_FILE_GB,
) -> None:
    """Worker process: pulls videos from queue, runs YOLO, pushes records with stall watchdog."""
    try:
        import logging

        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        logging.getLogger("scenedetect").setLevel(logging.WARNING)
        from ultralytics import YOLO

        model = YOLO(model_name)
        while True:
            try:
                video_file: Path = task_q.get_nowait()
            except queue.Empty:
                break

            fsize = video_file.stat().st_size
            prog_q.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "video_name": video_file.name,
                    "status": "started",
                    "file_size": fsize,
                }
            )

            progress_state: dict[str, Any] = {
                "last_active": time.perf_counter(),
                "kf_count": 0,
                "stop_event": threading.Event(),
            }
            result_holder: list[Any] = [None]

            def _target(
                vf: Path = video_file,
                ps: dict[str, Any] = progress_state,
                rh: list[Any] = result_holder,
            ) -> None:
                try:
                    rh[0] = _process_single_video(
                        vf,
                        model,
                        gpu_id,
                        conf,
                        batch_size,
                        use_scenes,
                        stride,
                        max_scene_len,
                        max_det,
                        large_threshold_gb,
                        res_q,
                        prog_q,
                        worker_id,
                        progress_state=ps,
                    )
                except Exception as exc:  # noqa: BLE001
                    rh[0] = exc

            t = threading.Thread(target=_target, daemon=True)
            t.start()

            stall_limit = timeout if timeout > 0 else 300.0
            timed_out = False
            while t.is_alive():
                t.join(timeout=2.0)
                if not t.is_alive():
                    break
                idle_s = time.perf_counter() - progress_state["last_active"]
                if stall_limit > 0 and idle_s > stall_limit:
                    timed_out = True
                    progress_state["stop_event"].set()
                    print(
                        f">  [{gpu_id}] [{video_file.name}] STALL TIMEOUT: No progress for {int(idle_s)}s (limit: {int(stall_limit)}s)!",
                        flush=True,
                    )
                    t.join(timeout=5.0)
                    break

            if timed_out:
                prog_q.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "video_name": video_file.name,
                        "file_size": fsize,
                        "num_keyframes": progress_state["kf_count"],
                        "num_detections": 0,
                        "fps": 0.0,
                        "status": "timeout",
                        "error_msg": f"Stalled for >{int(stall_limit)}s without new frames",
                    }
                )
            elif isinstance(result_holder[0], Exception):
                print(
                    f">  [{gpu_id}] [{video_file.name}] ERROR: {result_holder[0]}",
                    flush=True,
                )
                prog_q.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "video_name": video_file.name,
                        "file_size": fsize,
                        "num_keyframes": 0,
                        "num_detections": 0,
                        "fps": 0.0,
                        "status": "error",
                        "error_msg": str(result_holder[0]),
                    }
                )
            elif result_holder[0] is not None:
                tot_kfs, n_det, el = result_holder[0]
                prog_q.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "video_name": video_file.name,
                        "file_size": fsize,
                        "num_keyframes": tot_kfs,
                        "num_detections": n_det,
                        "fps": round(tot_kfs / el, 1),
                        "status": "ok",
                        "error_msg": None,
                    }
                )
            else:
                prog_q.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "video_name": video_file.name,
                        "file_size": fsize,
                        "num_keyframes": 0,
                        "num_detections": 0,
                        "fps": 0.0,
                        "status": "error",
                        "error_msg": "Unknown failure (no result)",
                    }
                )
    except Exception as fatal:  # noqa: BLE001
        prog_q.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "video_name": "<fatal>",
                "file_size": 0,
                "num_keyframes": 0,
                "num_detections": 0,
                "fps": 0.0,
                "status": "error",
                "error_msg": f"FATAL: {fatal}",
            }
        )
    finally:
        res_q.put(("SENTINEL", worker_id))


def collect_video_files(
    inputs: list[str], recursive_dirs: list[str] | None
) -> list[Path]:
    """Scan and deduplicate video files from explicit paths and recursive folders."""
    files = []
    targets = [(Path(p).resolve(), False) for p in inputs]
    if recursive_dirs:
        targets.extend((Path(p).resolve(), True) for p in recursive_dirs)
    for p, rec in targets:
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
        elif p.is_dir():
            pattern = "**/*" if (rec or recursive_dirs is not None) else "*"
            files.extend(
                f.resolve()
                for f in p.glob(pattern)
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS
            )
    return list(dict.fromkeys(files))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AIC_indexer.py: Multi-GPU Video Indexer for VISIONE obj-idx Parquet."
    )
    p.add_argument("inputs", nargs="*", default=[], help="Video files or directories.")
    p.add_argument(
        "-r", "--recursive", nargs="*", default=None, help="Scan folders recursively."
    )
    p.add_argument(
        "-o",
        "--output",
        default="obj-idx.parquet",
        help="Output parquet path (default: obj-idx.parquet).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sample 1 frame every N frames (default: 10).",
    )
    p.add_argument(
        "--scene-detect",
        action="store_true",
        default=False,
        help="Use PySceneDetect shot keyframing.",
    )
    p.add_argument(
        "--max-scene-len",
        type=float,
        default=10.0,
        help="Max shot duration in seconds (default: 10.0).",
    )
    p.add_argument(
        "--model",
        default="yolo26x.pt",
        help="YOLO model path or name (default: yolo26x.pt).",
    )
    p.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)."
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Max detections per frame for NMS (default: 300).",
    )
    p.add_argument(
        "--batch-size", type=int, default=32, help="Inference batch size (default: 32)."
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs (0 = auto-detect all available).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Inactivity stall timeout in seconds without progress before declaring video hung (default: {DEFAULT_TIMEOUT}s). Active videos never time out.",
    )
    p.add_argument(
        "--large-file-threshold",
        type=float,
        default=DEFAULT_LARGE_FILE_GB,
        help=f"GB threshold for adaptive stride / skip scene detect (default: {DEFAULT_LARGE_FILE_GB}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    v_files = collect_video_files(args.inputs, args.recursive)
    if not v_files:
        print("No valid video files found to index.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output).resolve()
    tot_bytes = sum(f.stat().st_size for f in v_files)
    tot_mb = tot_bytes / (1024.0 * 1024.0)

    import torch

    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_workers = args.num_gpus if args.num_gpus > 0 else max(1, n_cuda)
    devices = (
        [f"cuda:{i}" for i in range(n_workers)] if n_cuda > 0 else ["cpu"] * n_workers
    )

    print("=== AIC Indexer Configuration ===")
    print(f"Input Videos     : {len(v_files)} video(s) ({tot_mb:.2f} MB)")
    print(f"Output Path      : {out_path}")
    print(
        f"Pipeline Mode    : {'PySceneDetect' if args.scene_detect else f'1 frame / {args.stride} frames'}"
    )
    print(f"Hardware Workers : {n_workers} worker(s) ({', '.join(devices)})")
    print(
        f"Model & Batch    : {args.model} (conf={args.conf}, batch={args.batch_size})"
    )
    print(
        f"Per-Video Timeout: {args.timeout}s | Large File: >{args.large_file_threshold} GB"
    )
    print("Codec Routing    : AV1/VP9 → ffmpeg pipe, others → OpenCV")
    print("=================================\n", flush=True)

    ctx = mp.get_context("spawn")
    task_q, res_q, prog_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    for f in v_files:
        task_q.put(f)

    procs = [
        ctx.Process(
            target=gpu_worker_process,
            args=(
                i,
                devices[i],
                task_q,
                res_q,
                prog_q,
                args.model,
                args.conf,
                args.batch_size,
                args.scene_detect,
                args.stride,
                args.max_scene_len,
                args.max_det,
                args.timeout,
                args.large_file_threshold,
            ),
        )
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrow_schema = pl.DataFrame(schema=SCHEMA).to_arrow().schema
    parquet_writer: pq.ParquetWriter | None = pq.ParquetWriter(
        str(out_path), schema=arrow_schema, compression="zstd"
    )

    comp_bytes = n_vids = tot_kfs = tot_dets = 0
    errs: list[tuple[str, str]] = []
    active_procs = len(procs)
    t_start = time.perf_counter()

    active_w = set(range(n_workers))
    dead_w: set[int] = set()
    cur_vid: dict[int, str] = {}
    cur_progress: dict[int, dict[str, Any]] = {}
    pending: dict[int, list[dict[str, Any]]] = {w: [] for w in range(n_workers)}
    last_flush_time = time.perf_counter()

    def append_records_to_parquet(recs: list[dict[str, Any]]) -> None:
        nonlocal parquet_writer
        if not recs or parquet_writer is None:
            return
        try:
            chunk_df = pl.DataFrame(recs, schema=SCHEMA)
            parquet_writer.write_table(chunk_df.to_arrow())
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: Incremental parquet write failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def flush_block() -> None:
        nonlocal last_flush_time
        has_pending = any(pending[w] for w in range(n_workers))
        if not has_pending and not active_w:
            return

        el = max(1e-4, time.perf_counter() - t_start)
        eta_str = compute_overall_eta(comp_bytes, tot_bytes, n_vids, len(v_files), el)
        pct_all = (comp_bytes / max(1, tot_bytes)) * 100.0

        lines = [
            f"[{int(el)}s | {comp_bytes / (1024.0 * 1024.0):.2f} / {tot_mb:.2f} MB | {pct_all:>5.1f}% | ETA: {eta_str}]"
        ]
        for w in range(n_workers):
            if pending[w]:
                for item in pending[w]:
                    tag, name = f"[{item['gpu_id']}]", item["video_name"]
                    mb = item["file_size"] / (1024.0 * 1024.0)
                    pct = (item["global_idx"] / len(v_files)) * 100.0
                    idx_str = (
                        f"{item['global_idx']:>{len(str(len(v_files)))}}/{len(v_files)}"
                    )
                    if item["status"] == "ok":
                        lines.append(
                            f">  {tag} [{idx_str} | {pct:>5.1f}%] {name} | {item['num_keyframes']} kf | {item['fps']} fps | {item['num_detections']} objs | {mb:.1f} MB | ETA: {eta_str}"
                        )
                    elif item["status"] == "timeout":
                        msg = str(item["error_msg"] or "Timed out")
                        errs.append((name, msg))
                        lines.append(
                            f">  {tag} [{idx_str} | TIMEOUT] {name} | {msg} | ETA: {eta_str}"
                        )
                    else:
                        msg = str(item["error_msg"] or "Unknown error")
                        errs.append((name, msg))
                        lines.append(
                            f">  {tag} [{idx_str} | FAIL] {name} | {msg} | ETA: {eta_str}"
                        )
                pending[w].clear()
            elif w in active_w and w in cur_vid:
                info_p = cur_progress.get(w)
                if info_p and info_p.get("video_name") == cur_vid[w]:
                    tot_exp = info_p.get("tot_expected_kfs", 0)
                    kfs = info_p.get("num_keyframes", 0)
                    fps_val = info_p.get("fps", 0.0)
                    vid_eta = info_p.get("vid_eta", "--")
                    dec_st = info_p.get("decoder_status", "")
                    dec_msg = f" | {dec_st}" if dec_st else ""
                    if tot_exp > 0:
                        pct = min(100.0, (kfs / tot_exp) * 100.0)
                        lines.append(
                            f">  [{devices[w]}] processing... {cur_vid[w]} ({kfs}/{tot_exp} kf{dec_msg} | {pct:>5.1f}% | {fps_val} fps | ETA: {vid_eta})"
                        )
                    else:
                        lines.append(
                            f">  [{devices[w]}] processing... {cur_vid[w]} ({kfs} kf{dec_msg} | {fps_val} fps)"
                        )
                else:
                    lines.append(f">  [{devices[w]}] processing... {cur_vid[w]}")

        lines.append(f"[ETA: {eta_str}]\n")
        try:
            print("\n".join(lines), flush=True)
        except (BrokenPipeError, OSError):
            pass
        last_flush_time = time.perf_counter()

    while active_procs > 0 or not res_q.empty() or not prog_q.empty():
        try:
            info = prog_q.get(timeout=0.1)
            wid = info["worker_id"]
            status = info.get("status")

            if status == "started":
                cur_vid[wid] = info["video_name"]
                cur_progress.pop(wid, None)
                continue
            if status == "progress":
                cur_progress[wid] = info
                continue

            comp_bytes += info["file_size"]
            n_vids += 1
            tot_kfs += info["num_keyframes"]
            tot_dets += info["num_detections"]
            info["global_idx"] = n_vids
            pending[wid].append(info)
            cur_vid.pop(wid, None)
            cur_progress.pop(wid, None)

            if (active_w and all(pending[w] for w in active_w)) or (
                time.perf_counter() - last_flush_time >= 10.0
            ):
                flush_block()
        except queue.Empty:
            if active_w and (time.perf_counter() - last_flush_time >= 10.0):
                flush_block()

        try:
            recs = res_q.get(timeout=0.1)
            if isinstance(recs, tuple) and recs[0] == "SENTINEL":
                wid = recs[1]
                active_procs -= 1
                dead_w.add(wid)
                active_w.discard(wid)
                cur_progress.pop(wid, None)
            else:
                append_records_to_parquet(recs)
        except queue.Empty:
            pass

        if active_procs > 0 and res_q.empty() and prog_q.empty():
            for i, p in enumerate(procs):
                if i not in dead_w and not p.is_alive():
                    dead_w.add(i)
                    active_procs -= 1
                    active_w.discard(i)
                    cur_progress.pop(i, None)
                    print(
                        f"WARNING: Worker {i} ({devices[i]}) died unexpectedly.",
                        file=sys.stderr,
                        flush=True,
                    )

                    if i in cur_vid:
                        lost_vid_name = cur_vid.pop(i)
                        lost_path = next(
                            (f for f in v_files if f.name == lost_vid_name), None
                        )
                        if lost_path is not None:
                            task_q.put(lost_path)
                            print(
                                f"  Re-queued {lost_vid_name} for retry.",
                                file=sys.stderr,
                                flush=True,
                            )

                    if not task_q.empty():
                        new_id = len(procs)
                        gpu = devices[i]
                        new_p = ctx.Process(
                            target=gpu_worker_process,
                            args=(
                                new_id,
                                gpu,
                                task_q,
                                res_q,
                                prog_q,
                                args.model,
                                args.conf,
                                args.batch_size,
                                args.scene_detect,
                                args.stride,
                                args.max_scene_len,
                                args.max_det,
                                args.timeout,
                                args.large_file_threshold,
                            ),
                        )
                        procs.append(new_p)
                        devices.append(gpu)
                        active_w.add(new_id)
                        pending[new_id] = []
                        new_p.start()
                        active_procs += 1
                        print(
                            f"  Spawned replacement worker {new_id} on {gpu}.",
                            file=sys.stderr,
                            flush=True,
                        )

    flush_block()
    for p in procs:
        p.join()

    if parquet_writer is not None:
        try:
            parquet_writer.close()
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Closing ParquetWriter failed: {exc}", file=sys.stderr)
        parquet_writer = None

    t_tot = max(1e-3, time.perf_counter() - t_start)
    out_mb = out_path.stat().st_size / (1024.0 * 1024.0) if out_path.exists() else 0.0
    print("=== Indexing Complete ===")
    print(f"Total Videos Processed  : {n_vids} / {len(v_files)}")
    print(f"Successful Videos       : {n_vids - len(errs)}")
    if errs:
        print(f"Failed Videos           : {len(errs)}")
    print(f"Total Keyframes Indexed : {tot_kfs:,}")
    print(f"Total Objects Detected  : {tot_dets:,}")
    print(f"Total Processing Time   : {t_tot:.2f} s ({t_tot / 60:.1f} min)")
    print(f"Overall Throughput      : {round(tot_kfs / t_tot, 1)} keyframes/sec")
    print(f"Parquet Output Path     : {out_path}")
    print(f"Parquet File Size       : {out_mb:.2f} MB ({out_mb * 1024:.1f} KB)")
    print("=========================", flush=True)


if __name__ == "__main__":
    main()
