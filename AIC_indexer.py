# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
#     "duckdb",
#     "pyarrow",
#     "orjson",
#     "opencv-python-headless",
#     "ultralytics",
#     "scenedetect",
# ]
# ///
"""
AIC_indexer.py: High-throughput Video Object & Scene Indexer matching VISIONE Advanced Mode.
Optimized for multi-GPU setups (e.g. Kaggle 2x T4) with PySceneDetect keyframing and YOLO detection.
Outputs to obj-idx.parquet with shot boundaries, timestamps, 7x7 spatial grid tokens, and surrogate strings.
"""

from __future__ import annotations

import argparse
import collections
from datetime import datetime
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
from typing import Any

import cv2
import numpy as np
import polars as pl


VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".flv", ".wmv", ".m4v", ".ts", ".mts", ".m2ts"
}

COLOR_RANGES = [
    ("black",  (0, 0, 0),       (180, 255, 45)),
    ("white",  (0, 0, 200),     (180, 30, 255)),
    ("grey",   (0, 0, 45),      (180, 40, 200)),
    ("red",    (0, 70, 50),     (10, 255, 255)),
    ("red",    (170, 70, 50),   (180, 255, 255)),
    ("orange", (11, 70, 50),    (25, 255, 255)),
    ("yellow", (26, 70, 50),    (35, 255, 255)),
    ("green",  (36, 70, 50),    (85, 255, 255)),
    ("blue",   (86, 70, 50),    (125, 255, 255)),
    ("purple", (126, 70, 50),   (145, 255, 255)),
    ("pink",   (146, 70, 50),   (169, 255, 255)),
]


def encode_positional_boxes(
    boxes_yxyx: list[list[float]],
    labels: list[str],
    nrows: int = 7,
    ncols: int = 7,
    rtol: float = 0.1,
) -> str:
    """Encode bounding boxes into VISIONE 7x7 spatial grid tokens (e.g. '2ccar')."""
    tokens: list[str] = []
    xtol = rtol / ncols
    ytol = rtol / nrows

    for (y0, x0, y1, x1), label in zip(boxes_yxyx, labels):
        clean_label = label.lower().replace(" ", "_")
        start_col = math.floor((max(0.0, x0) + xtol) * ncols)
        start_row = math.floor((max(0.0, y0) + ytol) * nrows)
        end_col = math.floor((min(1.0, x1) - xtol) * ncols)
        end_row = math.floor((min(1.0, y1) - ytol) * nrows)

        for r in range(max(0, start_row), min(nrows, end_row + 1)):
            for c in range(max(0, start_col), min(ncols, end_col + 1)):
                col_char = chr(ord("a") + c)
                tokens.append(f"{r}{col_char}{clean_label}")

    tokens.sort()
    return " ".join(tokens)


def encode_object_counts(
    labels: list[str],
    scores: list[float],
) -> str:
    """Encode object counts into VISIONE surrogate text format (e.g. '4wcperson1|6')."""
    counts: collections.Counter[str] = collections.Counter()
    tokens: list[str] = []

    for label, score in zip(labels, scores):
        clean_label = label.lower().replace(" ", "_")
        counts[clean_label] += 1
        cnt = counts[clean_label]
        freq = max(1, int(10.0 * score / 2.0 + 2.0))
        tokens.append(f"4wc{clean_label}{cnt}|{freq}")

    tokens.sort()
    return " ".join(tokens)


def analyze_frame_colors(frame_bgr: np.ndarray) -> tuple[bool, list[str]]:
    """Analyze frame color palette and detect monochrome/grayscale."""
    small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    sat = hsv[:, :, 1]
    is_monochrome = float(np.mean(sat)) < 22.0

    if is_monochrome:
        return True, ["monochrome"]

    detected_colors: list[str] = []
    total_pixels = small.shape[0] * small.shape[1]
    for color_name, lower, upper in COLOR_RANGES:
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        pixel_count = int(cv2.countNonZero(mask))
        if pixel_count / total_pixels >= 0.08:
            if color_name not in detected_colors:
                detected_colors.append(color_name)

    return False, detected_colors


def detect_scenes_and_keyframes(
    video_path: Path,
    max_scene_len_sec: float = 10.0,
    adaptive_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    """
    VISIONE-matching scene detection using PySceneDetect.
    Splits video into shots, partitions long shots by max_scene_len_sec,
    and calculates middle keyframe for each shot.
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import AdaptiveDetector

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=adaptive_threshold))
    scene_manager.detect_scenes(video)
    raw_scenes = scene_manager.get_scene_list()

    fps = float(video.frame_rate) if video.frame_rate > 0 else 25.0
    total_frames = int(video.duration.get_frames())

    # Fallback if no scenes detected: treat whole video as one shot
    if not raw_scenes:
        raw_scenes = [(video.base_timecode, video.duration)]

    shots: list[dict[str, Any]] = []

    for start_tc, end_tc in raw_scenes:
        s_frame = start_tc.get_frames()
        e_frame = max(s_frame, end_tc.get_frames() - 1)
        s_sec = start_tc.get_seconds()
        e_sec = end_tc.get_seconds()
        duration_sec = e_sec - s_sec

        # Partition long scenes into sub-scenes like VISIONE post_process_scenes.py
        if max_scene_len_sec > 0 and duration_sec > max_scene_len_sec:
            num_sub = math.ceil(duration_sec / max_scene_len_sec)
            sub_len_frames = (e_frame - s_frame + 1) / num_sub
            for i in range(num_sub):
                sub_s_frame = int(round(s_frame + i * sub_len_frames))
                sub_e_frame = int(round(s_frame + (i + 1) * sub_len_frames - 1))
                sub_e_frame = min(sub_e_frame, e_frame)
                mid_frame = (sub_s_frame + sub_e_frame) // 2

                shots.append({
                    "start_frame": sub_s_frame,
                    "end_frame": sub_e_frame,
                    "start_time": round(sub_s_frame / fps, 3),
                    "end_time": round(sub_e_frame / fps, 3),
                    "middle_frame": mid_frame,
                    "middle_time": round(mid_frame / fps, 3),
                })
        else:
            mid_frame = (s_frame + e_frame) // 2
            shots.append({
                "start_frame": s_frame,
                "end_frame": e_frame,
                "start_time": round(s_sec, 3),
                "end_time": round(e_sec, 3),
                "middle_frame": mid_frame,
                "middle_time": round((s_sec + e_sec) / 2.0, 3),
            })

    return shots


def extract_shot_keyframes(
    video_path: Path,
    shots: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], np.ndarray]], float, int, int]:
    """Read representative keyframes from video according to shot bounds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0, 0, 0

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sort shots by middle frame to read sequentially
    sorted_shots = sorted(shots, key=lambda s: s["middle_frame"])
    results: list[tuple[dict[str, Any], np.ndarray]] = []

    target_map = {s["middle_frame"]: s for s in sorted_shots}
    current_frame = 0

    # Sequential scan with fast seek when gaps are large
    for shot in sorted_shots:
        target_f = shot["middle_frame"]
        if target_f - current_frame > 30:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            current_frame = target_f

        while current_frame < target_f:
            ret = cap.grab()
            if not ret:
                break
            current_frame += 1

        ret, frame = cap.read()
        if ret and frame is not None:
            results.append((shot, frame))
            current_frame += 1
        else:
            break

    cap.release()
    return results, fps, width, height


def extract_strided_frames(
    video_path: Path,
    stride: int = 10,
) -> tuple[list[tuple[dict[str, Any], np.ndarray]], float, int, int]:
    """
    Extract 1 frame every N frames (default: 10 frames).
    Uses fast cap.grab() skipping to avoid decoding unneeded frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0, 0, 0

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(stride))

    results: list[tuple[dict[str, Any], np.ndarray]] = []
    f_idx = 0

    while True:
        if f_idx % stride == 0:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            t_sec = float(msec / 1000.0) if msec > 0 else float(f_idx / fps)
            shot_meta = {
                "start_frame": max(0, f_idx - stride // 2),
                "end_frame": f_idx + stride // 2,
                "start_time": round(max(0.0, (f_idx - stride // 2) / fps), 3),
                "end_time": round((f_idx + stride // 2) / fps, 3),
                "middle_frame": f_idx,
                "middle_time": round(t_sec, 3),
            }
            results.append((shot_meta, frame))
        else:
            # Fast grab without decoding pixel buffer
            ret = cap.grab()
            if not ret:
                break

        f_idx += 1

    cap.release()
    return results, fps, width, height


def gpu_worker_process(
    worker_id: int,
    gpu_id: str,
    video_tasks: list[Path],
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
    model_name: str,
    conf_thresh: float,
    batch_size: int,
    use_scenes: bool,
    stride: int,
    max_scene_len: float,
) -> None:
    """Worker process dedicated to 1 GPU (e.g. cuda:0 or cuda:1 on Kaggle 2x T4)."""
    import logging
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("scenedetect").setLevel(logging.WARNING)

    import torch
    from ultralytics import YOLO

    device = gpu_id
    model = YOLO(model_name)

    for video_file in video_tasks:
        t_vid_start = time.perf_counter()
        file_size = video_file.stat().st_size
        video_id = video_file.stem
        records: list[dict[str, Any]] = []
        total_detections = 0

        try:
            # Extract frames: default 1 every N frames (default 10) or shot detection
            if use_scenes:
                try:
                    shots = detect_scenes_and_keyframes(video_file, max_scene_len_sec=max_scene_len)
                    keyframes_data, fps, width, height = extract_shot_keyframes(video_file, shots)
                except Exception:
                    keyframes_data, fps, width, height = extract_strided_frames(video_file, stride=stride)
            else:
                keyframes_data, fps, width, height = extract_strided_frames(video_file, stride=stride)

            if keyframes_data:
                for b_start in range(0, len(keyframes_data), batch_size):
                    batch_slice = keyframes_data[b_start : b_start + batch_size]
                    b_imgs = [item[1] for item in batch_slice]

                    # Run YOLO with Tensor Core FP16 acceleration
                    results = model.predict(
                        b_imgs,
                        conf=conf_thresh,
                        device=device,
                        verbose=False,
                    )

                    for (shot_meta, f_img), det in zip(batch_slice, results):
                        is_mono, dominant_colors = analyze_frame_colors(f_img)

                        b_labels: list[str] = []
                        b_scores: list[float] = []
                        b_boxes: list[list[float]] = []

                        if det.boxes is not None and len(det.boxes) > 0:
                            confs = det.boxes.conf.cpu().numpy()
                            clss = det.boxes.cls.cpu().numpy().astype(int)
                            xyxy = det.boxes.xyxy.cpu().numpy()

                            h_img, w_img = f_img.shape[:2]
                            for c_val, cl_id, box in zip(confs, clss, xyxy):
                                label = str(det.names.get(cl_id, f"obj_{cl_id}"))
                                y0 = float(box[1] / h_img)
                                x0 = float(box[0] / w_img)
                                y1 = float(box[3] / h_img)
                                x1 = float(box[2] / w_img)

                                b_labels.append(label)
                                b_scores.append(float(c_val))
                                b_boxes.append([y0, x0, y1, x1])

                        txt_str = encode_positional_boxes(b_boxes, b_labels)
                        obj_str = encode_object_counts(b_labels, b_scores)
                        unique_objects = sorted(list(set(b_labels)))
                        total_detections += len(b_labels)

                        records.append({
                            "video_id": video_id,
                            "time": float(shot_meta["middle_time"]),
                            "frame_idx": int(shot_meta["middle_frame"]),
                            "start_time": float(shot_meta["start_time"]),
                            "end_time": float(shot_meta["end_time"]),
                            "start_frame": int(shot_meta["start_frame"]),
                            "end_frame": int(shot_meta["end_frame"]),
                            "fps": float(round(fps, 2)),
                            "width": int(width),
                            "height": int(height),
                            "objects": unique_objects,
                            "scores": b_scores,
                            "boxes": b_boxes,
                            "labels": b_labels,
                            "txt": txt_str,
                            "objects_str": obj_str,
                            "colors": dominant_colors,
                            "is_monochrome": is_mono,
                            "video_path": str(video_file),
                        })

            elapsed_sec = max(1e-4, time.perf_counter() - t_vid_start)
            result_queue.put(records)
            progress_queue.put({
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "video_name": video_file.name,
                "file_size": file_size,
                "num_keyframes": len(keyframes_data),
                "num_detections": total_detections,
                "elapsed_sec": elapsed_sec,
                "fps": round(len(keyframes_data) / elapsed_sec, 1),
                "status": "ok",
                "error_msg": None,
            })
        except Exception as err:
            elapsed_sec = max(1e-4, time.perf_counter() - t_vid_start)
            result_queue.put([])
            progress_queue.put({
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "video_name": video_file.name,
                "file_size": file_size,
                "num_keyframes": 0,
                "num_detections": 0,
                "elapsed_sec": elapsed_sec,
                "fps": 0.0,
                "status": "error",
                "error_msg": str(err),
            })

    result_queue.put(None)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIC_indexer.py: Index videos to obj-idx.parquet matching VISIONE advanced mode."
    )
    parser.add_argument("inputs", nargs="*", default=[], help="Video files or directories.")
    parser.add_argument("-r", "--recursive", nargs="*", default=None, help="Scan folders recursively.")
    parser.add_argument("-o", "--output", default="obj-idx.parquet", help="Output parquet path (default: obj-idx.parquet).")
    parser.add_argument("--stride", type=int, default=10, help="Sample 1 frame every N frames (default: 10).")
    parser.add_argument("--scene-detect", action="store_true", default=False, help="Use PySceneDetect shot keyframing instead of frame stride.")
    parser.add_argument("--max-scene-len", type=float, default=10.0, help="Max shot duration in seconds before partitioning (default: 10.0).")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model path or name (default: yolo26n.pt).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size (default: 32).")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use (0 = auto-detect all available GPUs e.g. 2 on Kaggle).")
    return parser.parse_args()


def collect_video_files(inputs: list[str], recursive_arg: list[str] | None) -> list[Path]:
    files: list[Path] = []
    is_recursive_mode = recursive_arg is not None

    def scan_dir(dir_path: Path, recursive: bool) -> list[Path]:
        pattern = "**/*" if recursive else "*"
        return [
            p.resolve()
            for p in dir_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]

    for raw in inputs:
        p = Path(raw).resolve()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(p)
        elif p.is_dir():
            files.extend(scan_dir(p, recursive=is_recursive_mode))

    if recursive_arg is not None:
        for raw in recursive_arg:
            p = Path(raw).resolve()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(p)
            elif p.is_dir():
                files.extend(scan_dir(p, recursive=True))

    return list(dict.fromkeys(files))


def main() -> None:
    args = parse_arguments()

    video_files = collect_video_files(args.inputs, args.recursive)
    if not video_files:
        print("No valid video files found to index.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).resolve()
    total_bytes = sum(f.stat().st_size for f in video_files)
    total_mb = total_bytes / (1024.0 * 1024.0)

    # Multi-GPU detection (Kaggle 2x T4 optimization)
    import torch
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_workers = args.num_gpus if args.num_gpus > 0 else max(1, available_gpus)
    devices = [f"cuda:{i}" for i in range(num_workers)] if available_gpus > 0 else ["cpu"] * num_workers

    mode_desc = "PySceneDetect (Shot Keyframing)" if args.scene_detect else f"1 frame every {args.stride} frames"

    # Startup Configuration
    print("=== AIC Indexer Configuration ===")
    print(f"Input Videos     : {len(video_files)} video(s) ({total_mb:.2f} MB)")
    print(f"Output Path      : {output_path}")
    print(f"Pipeline Mode    : {mode_desc}")
    print(f"Hardware Workers : {num_workers} worker(s) ({', '.join(devices)})")
    print(f"Model & Batch    : {args.model} (conf={args.conf}, batch={args.batch_size})")
    print("=================================\n", flush=True)

    # Partition video tasks round-robin across GPU workers
    worker_tasks: list[list[Path]] = [[] for _ in range(num_workers)]
    for idx, v_file in enumerate(video_files):
        worker_tasks[idx % num_workers].append(v_file)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    progress_queue = ctx.Queue()

    processes = []
    for w_id in range(num_workers):
        if not worker_tasks[w_id]:
            continue
        p = ctx.Process(
            target=gpu_worker_process,
            args=(
                w_id,
                devices[w_id],
                worker_tasks[w_id],
                result_queue,
                progress_queue,
                args.model,
                args.conf,
                args.batch_size,
                args.scene_detect,
                args.stride,
                args.max_scene_len,
            ),
        )
        p.start()
        processes.append(p)

    all_records: list[dict[str, Any]] = []
    completed_bytes = 0
    finished_videos = 0
    total_keyframes_count = 0
    total_detections_count = 0
    errors: list[tuple[str, str]] = []
    total_vids = len(video_files)
    active_workers = len(processes)
    start_time_all = time.perf_counter()

    worker_total_tasks = [len(worker_tasks[w]) for w in range(num_workers)]
    worker_finished_count = [0 for _ in range(num_workers)]
    active_worker_ids = {w for w in range(num_workers) if worker_total_tasks[w] > 0}

    print("Starting indexing progression...\n", flush=True)
    worker_pending: dict[int, list[dict[str, Any]]] = {w: [] for w in range(num_workers)}
    first_pending_time: float | None = None

    def flush_block() -> None:
        nonlocal first_pending_time
        has_items = any(len(worker_pending[w]) > 0 for w in range(num_workers))
        if not has_items:
            return

        elapsed_total = max(1e-4, time.perf_counter() - start_time_all)
        elapsed_s = int(elapsed_total)
        comp_mb = completed_bytes / (1024.0 * 1024.0)

        # Header: [104s | 229.29 / 7184.48 MB]
        print(f"[{elapsed_s}s | {comp_mb:.2f} / {total_mb:.2f} MB]")

        # Per-worker completed lines
        for w in range(num_workers):
            for item in worker_pending[w]:
                gpu_tag = f"[{item['gpu_id']}]"
                v_name = item["video_name"]
                pct = (item["video_global_idx"] / total_vids) * 100.0
                mb_size = item["file_size"] / (1024.0 * 1024.0)

                if item["status"] == "ok":
                    print(
                        f">  {gpu_tag} [{item['video_global_idx']:>{len(str(total_vids))}}/{total_vids} | {pct:>5.1f}%] "
                        f"{v_name} | {item['num_keyframes']} kf | {item['fps']} fps | {item['num_detections']} objs | {mb_size:.1f} MB"
                    )
                else:
                    err_msg = str(item["error_msg"] or "Unknown error")
                    errors.append((item["video_name"], err_msg))
                    print(
                        f">  {gpu_tag} [{item['video_global_idx']}/{total_vids} | FAIL] {v_name} | {err_msg}"
                    )
            worker_pending[w].clear()

        # Footer: [ETA: ..] averaged from worker ETAs
        worker_etas: list[float] = []
        for w in range(num_workers):
            rem_w = worker_total_tasks[w] - worker_finished_count[w]
            if rem_w > 0:
                if worker_finished_count[w] > 0:
                    avg_time_w = elapsed_total / worker_finished_count[w]
                    worker_etas.append(rem_w * avg_time_w)
                elif finished_videos > 0:
                    avg_time_all = elapsed_total / finished_videos
                    worker_etas.append(rem_w * avg_time_all)

        if worker_etas:
            eta_sec = int(sum(worker_etas) / len(worker_etas))
            eta_str = f"{eta_sec // 60}m {eta_sec % 60:02d}s" if eta_sec >= 60 else f"{eta_sec}s"
        else:
            eta_str = "0s"

        print(f"[ETA: {eta_str}]\n", flush=True)
        first_pending_time = None

    while active_workers > 0 or not result_queue.empty() or not progress_queue.empty():
        try:
            info = progress_queue.get(timeout=0.1)
            completed_bytes += info["file_size"]
            finished_videos += 1
            total_keyframes_count += info["num_keyframes"]
            total_detections_count += info["num_detections"]

            w_id = info["worker_id"]
            worker_finished_count[w_id] += 1
            info["video_global_idx"] = finished_videos
            worker_pending[w_id].append(info)

            if worker_finished_count[w_id] >= worker_total_tasks[w_id]:
                active_worker_ids.discard(w_id)

            if first_pending_time is None:
                first_pending_time = time.perf_counter()

            all_active_reported = bool(active_worker_ids) and all(
                len(worker_pending[w]) > 0 for w in active_worker_ids
            )
            timed_out = (first_pending_time is not None) and (time.perf_counter() - first_pending_time > 10.0)

            if all_active_reported or timed_out:
                flush_block()
        except queue.Empty:
            if first_pending_time is not None and (time.perf_counter() - first_pending_time > 10.0):
                flush_block()

        try:
            records = result_queue.get(timeout=0.1)
            if records is None:
                active_workers -= 1
            else:
                all_records.extend(records)
        except queue.Empty:
            pass

    flush_block()

    for p in processes:
        p.join()

    print(f"Saving index of {len(all_records)} keyframes to '{output_path}'...", flush=True)
    if all_records:
        df = pl.DataFrame(all_records)
    else:
        df = pl.DataFrame(
            schema={
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
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")

    # Finish Summary
    total_elapsed = max(1e-3, time.perf_counter() - start_time_all)
    overall_fps = round(total_keyframes_count / total_elapsed, 1)
    out_size_mb = output_path.stat().st_size / (1024.0 * 1024.0) if output_path.exists() else 0.0

    print("=== Indexing Complete ===")
    print(f"Total Videos Processed  : {finished_videos} / {total_vids}")
    print(f"Successful Videos       : {finished_videos - len(errors)}")
    if errors:
        print(f"Failed Videos           : {len(errors)}")
    print(f"Total Keyframes Indexed : {total_keyframes_count:,}")
    print(f"Total Objects Detected  : {total_detections_count:,}")
    print(f"Total Processing Time   : {total_elapsed:.2f} s ({total_elapsed / 60:.1f} min)")
    print(f"Overall Throughput      : {overall_fps} keyframes/sec")
    print(f"Parquet Output Path     : {output_path}")
    print(f"Parquet File Size       : {out_size_mb:.2f} MB ({out_size_mb * 1024:.1f} KB)")
    print("=========================", flush=True)


if __name__ == "__main__":
    main()
