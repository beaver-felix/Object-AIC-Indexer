# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
#     "pyarrow",
#     "numpy",
#     "opencv-python-headless",
#     "ultralytics",
#     "scenedetect",
# ]
# ///
"""AIC_indexer.py: Multi-GPU Video Indexer for VISIONE obj-idx Parquet."""

from __future__ import annotations

import argparse
from collections import Counter
import math
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
from typing import Any, Generator

import cv2
import numpy as np
import polars as pl

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".ts", ".mts", ".m2ts"}

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

SCHEMA = {
    "video_id": pl.Utf8, "time": pl.Float64, "frame_idx": pl.Int64,
    "start_time": pl.Float64, "end_time": pl.Float64, "start_frame": pl.Int64, "end_frame": pl.Int64,
    "fps": pl.Float32, "width": pl.Int32, "height": pl.Int32,
    "objects": pl.List(pl.Utf8), "scores": pl.List(pl.Float32), "boxes": pl.List(pl.List(pl.Float32)),
    "labels": pl.List(pl.Utf8), "txt": pl.Utf8, "objects_str": pl.Utf8,
    "colors": pl.List(pl.Utf8), "is_monochrome": pl.Boolean, "video_path": pl.Utf8,
}


def encode_positional_boxes(boxes: list[list[float]], labels: list[str], n: int = 7, tol: float = 0.1) -> str:
    """Encode bounding boxes into VISIONE 7x7 spatial tokens (e.g. '2ccar')."""
    tokens = []
    dt = tol / n
    for (y0, x0, y1, x1), label in zip(boxes, labels):
        lbl = label.lower().replace(" ", "_")
        c0, r0 = math.floor((max(0.0, x0) + dt) * n), math.floor((max(0.0, y0) + dt) * n)
        c1, r1 = math.floor((min(1.0, x1) - dt) * n), math.floor((min(1.0, y1) - dt) * n)
        tokens.extend(f"{r}{chr(97 + c)}{lbl}" for r in range(max(0, r0), min(n, r1 + 1)) for c in range(max(0, c0), min(n, c1 + 1)))
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
        mask = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
        if cv2.countNonZero(mask) >= 327 and name not in colors:  # 64*64*0.08 = 327.68
            colors.append(name)
    return False, colors


def detect_scenes_and_keyframes(video_path: Path, max_scene_len_sec: float = 10.0, thresh: float = 3.0) -> list[dict[str, Any]]:
    """Detect scenes with PySceneDetect, partition long shots, return keyframe metadata."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import AdaptiveDetector

    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(AdaptiveDetector(adaptive_threshold=thresh))
    sm.detect_scenes(video)
    raw = sm.get_scene_list() or [(video.base_timecode, video.duration)]

    fps = float(video.frame_rate) if video.frame_rate > 0 else 25.0
    shots = []
    for s_tc, e_tc in raw:
        sf, ef = s_tc.get_frames(), max(s_tc.get_frames(), e_tc.get_frames() - 1)
        dur = e_tc.get_seconds() - s_tc.get_seconds()
        if max_scene_len_sec > 0 and dur > max_scene_len_sec:
            k = math.ceil(dur / max_scene_len_sec)
            step = (ef - sf + 1) / k
            for i in range(k):
                s, e = int(round(sf + i * step)), min(ef, int(round(sf + (i + 1) * step - 1)))
                m = (s + e) // 2
                shots.append({"start_frame": s, "end_frame": e, "start_time": round(s / fps, 3), "end_time": round(e / fps, 3), "middle_frame": m, "middle_time": round(m / fps, 3)})
        else:
            m = (sf + ef) // 2
            shots.append({"start_frame": sf, "end_frame": ef, "start_time": round(s_tc.get_seconds(), 3), "end_time": round(e_tc.get_seconds(), 3), "middle_frame": m, "middle_time": round((s_tc.get_seconds() + e_tc.get_seconds()) / 2.0, 3)})
    return shots


def stream_shot_keyframes(
    video_path: Path, shots: list[dict[str, Any]], batch_size: int = 16
) -> Generator[tuple[list[dict[str, Any]], list[np.ndarray], float, int, int], None, None]:
    """Yield batches of keyframes for detected shots to keep memory footprint bounded."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cur = 0
    batch_meta: list[dict[str, Any]] = []
    batch_frames: list[np.ndarray] = []

    try:
        for shot in sorted(shots, key=lambda s: s["middle_frame"]):
            target = shot["middle_frame"]
            if target - cur > 30:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                cur = target
            while cur < target and cap.grab():
                cur += 1
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            cur += 1
            batch_meta.append(shot)
            batch_frames.append(frame)
            if len(batch_frames) >= batch_size:
                yield batch_meta, batch_frames, fps, w, h
                batch_meta = []
                batch_frames = []
        if batch_frames:
            yield batch_meta, batch_frames, fps, w, h
    finally:
        cap.release()


def stream_strided_frames(
    video_path: Path, stride: int = 10, batch_size: int = 16
) -> Generator[tuple[list[dict[str, Any]], list[np.ndarray], float, int, int], None, None]:
    """Yield batches of strided frames to keep memory footprint bounded."""
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


def extract_shot_keyframes(video_path: Path, shots: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], np.ndarray]], float, int, int]:
    """Read keyframes from video corresponding to detected shots."""
    results: list[tuple[dict[str, Any], np.ndarray]] = []
    fps, w, h = 25.0, 0, 0
    for b_meta, b_frames, fps, w, h in stream_shot_keyframes(video_path, shots, batch_size=64):
        for m, f in zip(b_meta, b_frames):
            results.append((m, f))
    return results, fps, w, h


def extract_strided_frames(video_path: Path, stride: int = 10) -> tuple[list[tuple[dict[str, Any], np.ndarray]], float, int, int]:
    """Extract frames at fixed stride using fast cap.grab() frame skipping."""
    results: list[tuple[dict[str, Any], np.ndarray]] = []
    fps, w, h = 25.0, 0, 0
    for b_meta, b_frames, fps, w, h in stream_strided_frames(video_path, stride=stride, batch_size=64):
        for m, f in zip(b_meta, b_frames):
            results.append((m, f))
    return results, fps, w, h


def gpu_worker_process(
    worker_id: int, gpu_id: str, task_q: mp.Queue, res_q: mp.Queue, prog_q: mp.Queue,
    model_name: str, conf: float, batch_size: int, use_scenes: bool, stride: int, max_scene_len: float,
    max_det: int = 300,
) -> None:
    """Worker process: pulls videos from queue, runs YOLO, pushes records."""
    try:
        import logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        logging.getLogger("scenedetect").setLevel(logging.WARNING)
        from ultralytics import YOLO

        model = YOLO(model_name)
        while True:
            try:
                video_file = task_q.get_nowait()
            except queue.Empty:
                break

            t0, fsize, vid = time.perf_counter(), video_file.stat().st_size, video_file.stem
            prog_q.put({"worker_id": worker_id, "gpu_id": gpu_id, "video_name": video_file.name, "status": "started"})
            try:
                if use_scenes:
                    try:
                        shots = detect_scenes_and_keyframes(video_file, max_scene_len_sec=max_scene_len)
                        batch_gen = stream_shot_keyframes(video_file, shots, batch_size=batch_size)
                    except Exception:
                        batch_gen = stream_strided_frames(video_file, stride=stride, batch_size=batch_size)
                else:
                    batch_gen = stream_strided_frames(video_file, stride=stride, batch_size=batch_size)

                records: list[dict[str, Any]] = []
                n_det = 0
                tot_kfs = 0
                for batch_meta, batch_frames, fps, w, h in batch_gen:
                    tot_kfs += len(batch_frames)
                    results = model.predict(batch_frames, conf=conf, device=gpu_id, verbose=False, max_det=max_det, agnostic_nms=True)
                    for meta, img, det in zip(batch_meta, batch_frames, results):
                        mono, colors = analyze_frame_colors(img)
                        b_lbls, b_scs, b_boxes = [], [], []
                        if det.boxes is not None and len(det.boxes) > 0:
                            ih, iw = img.shape[:2]
                            for c_val, cl_id, box in zip(det.boxes.conf.cpu().numpy(), det.boxes.cls.cpu().numpy().astype(int), det.boxes.xyxy.cpu().numpy()):
                                b_lbls.append(str(det.names.get(cl_id, f"obj_{cl_id}")))
                                b_scs.append(float(c_val))
                                b_boxes.append([float(box[1] / ih), float(box[0] / iw), float(box[3] / ih), float(box[2] / iw)])
                        n_det += len(b_lbls)
                        records.append({
                            "video_id": vid, "time": float(meta["middle_time"]), "frame_idx": int(meta["middle_frame"]),
                            "start_time": float(meta["start_time"]), "end_time": float(meta["end_time"]),
                            "start_frame": int(meta["start_frame"]), "end_frame": int(meta["end_frame"]),
                            "fps": float(round(fps, 2)), "width": int(w), "height": int(h),
                            "objects": sorted(list(set(b_lbls))), "scores": b_scs, "boxes": b_boxes, "labels": b_lbls,
                            "txt": encode_positional_boxes(b_boxes, b_lbls), "objects_str": encode_object_counts(b_lbls, b_scs),
                            "colors": colors, "is_monochrome": mono, "video_path": str(video_file),
                        })

                el = max(1e-4, time.perf_counter() - t0)
                res_q.put(records)
                prog_q.put({"worker_id": worker_id, "gpu_id": gpu_id, "video_name": video_file.name, "file_size": fsize, "num_keyframes": tot_kfs, "num_detections": n_det, "fps": round(tot_kfs / el, 1), "status": "ok", "error_msg": None})
            except Exception as err:
                el = max(1e-4, time.perf_counter() - t0)
                res_q.put([])
                prog_q.put({"worker_id": worker_id, "gpu_id": gpu_id, "video_name": video_file.name, "file_size": fsize, "num_keyframes": 0, "num_detections": 0, "fps": 0.0, "status": "error", "error_msg": str(err)})
    except Exception as fatal:
        prog_q.put({"worker_id": worker_id, "gpu_id": gpu_id, "video_name": "<fatal>", "file_size": 0, "num_keyframes": 0, "num_detections": 0, "fps": 0.0, "status": "error", "error_msg": f"FATAL: {fatal}"})
    finally:
        res_q.put(("SENTINEL", worker_id))



def collect_video_files(inputs: list[str], recursive_dirs: list[str] | None) -> list[Path]:
    """Scan and deduplicate video files from explicit paths and recursive folders."""
    files = []
    targets = [(Path(p).resolve(), False) for p in inputs]
    if recursive_dirs:
        targets.extend((Path(p).resolve(), True) for p in recursive_dirs)
    for p, rec in targets:
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
        elif p.is_dir():
            files.extend(f.resolve() for f in p.glob("**/*" if (rec or recursive_dirs is not None) else "*") if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
    return list(dict.fromkeys(files))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AIC_indexer.py: Multi-GPU Video Indexer for VISIONE obj-idx Parquet.")
    p.add_argument("inputs", nargs="*", default=[], help="Video files or directories.")
    p.add_argument("-r", "--recursive", nargs="*", default=None, help="Scan folders recursively.")
    p.add_argument("-o", "--output", default="obj-idx.parquet", help="Output parquet path.")
    p.add_argument("--stride", type=int, default=10, help="Sample 1 frame every N frames (default: 10).")
    p.add_argument("--scene-detect", action="store_true", default=False, help="Use PySceneDetect shot keyframing.")
    p.add_argument("--max-scene-len", type=float, default=10.0, help="Max shot duration in seconds (default: 10.0).")
    p.add_argument("--model", default="yolo26x.pt", help="YOLO model path or name (default: yolo26x.pt).")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    p.add_argument("--max-det", type=int, default=300, help="Max detections per frame for NMS (default: 300).")
    p.add_argument("--batch-size", type=int, default=32, help="Inference batch size (default: 32).")
    p.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs (0 = auto-detect all available).")
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
    devices = [f"cuda:{i}" for i in range(n_workers)] if n_cuda > 0 else ["cpu"] * n_workers

    print("=== AIC Indexer Configuration ===")
    print(f"Input Videos     : {len(v_files)} video(s) ({tot_mb:.2f} MB)")
    print(f"Output Path      : {out_path}")
    print(f"Pipeline Mode    : {'PySceneDetect' if args.scene_detect else f'1 frame / {args.stride} frames'}")
    print(f"Hardware Workers : {n_workers} worker(s) ({', '.join(devices)})")
    print(f"Model & Batch    : {args.model} (conf={args.conf}, batch={args.batch_size})")
    print("=================================\n", flush=True)

    ctx = mp.get_context("spawn")
    task_q, res_q, prog_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    for f in v_files:
        task_q.put(f)

    worker_args = lambda i: (i, devices[i], task_q, res_q, prog_q, args.model, args.conf, args.batch_size, args.scene_detect, args.stride, args.max_scene_len, args.max_det)
    procs = [
        ctx.Process(target=gpu_worker_process, args=worker_args(i))
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()

    records: list[dict[str, Any]] = []
    comp_bytes = n_vids = tot_kfs = tot_dets = 0
    errs: list[tuple[str, str]] = []
    active_procs = len(procs)
    t_start = time.perf_counter()

    active_w = set(range(n_workers))
    dead_w: set[int] = set()
    cur_vid: dict[int, str] = {}
    pending: dict[int, list[dict[str, Any]]] = {w: [] for w in range(n_workers)}
    t_first_pending: float | None = None

    def flush_block() -> None:
        nonlocal t_first_pending
        if not any(pending[w] for w in range(n_workers)):
            return

        el = max(1e-4, time.perf_counter() - t_start)
        lines = [f"[{int(el)}s | {comp_bytes / (1024.0 * 1024.0):.2f} / {tot_mb:.2f} MB]"]
        for w in range(n_workers):
            if pending[w]:
                for item in pending[w]:
                    tag, name = f"[{item['gpu_id']}]", item["video_name"]
                    mb = item["file_size"] / (1024.0 * 1024.0)
                    pct = (item["global_idx"] / len(v_files)) * 100.0
                    if item["status"] == "ok":
                        lines.append(f">  {tag} [{item['global_idx']:>{len(str(len(v_files)))}}/{len(v_files)} | {pct:>5.1f}%] {name} | {item['num_keyframes']} kf | {item['fps']} fps | {item['num_detections']} objs | {mb:.1f} MB")
                    else:
                        msg = str(item["error_msg"] or "Unknown error")
                        errs.append((name, msg))
                        lines.append(f">  {tag} [{item['global_idx']}/{len(v_files)} | FAIL] {name} | {msg}")
                pending[w].clear()
            elif w in active_w and w in cur_vid:
                lines.append(f">  [{devices[w]}] processing... {cur_vid[w]}")

        rem = len(v_files) - n_vids
        eta_s = int(rem * (el / n_vids) / max(1, len(active_w))) if rem > 0 and n_vids > 0 else 0
        eta_str = f"{eta_s // 60}m {eta_s % 60:02d}s" if eta_s >= 60 else f"{eta_s}s"
        lines.append(f"[ETA: {eta_str}]\n")
        try:
            print("\n".join(lines))
            sys.stdout.flush()
        except Exception:
            pass
        t_first_pending = None

    while active_procs > 0 or not res_q.empty() or not prog_q.empty():
        try:
            info = prog_q.get(timeout=0.1)
            wid = info["worker_id"]
            if info["status"] == "started":
                cur_vid[wid] = info["video_name"]
                continue

            comp_bytes += info["file_size"]
            n_vids += 1
            tot_kfs += info["num_keyframes"]
            tot_dets += info["num_detections"]
            info["global_idx"] = n_vids
            pending[wid].append(info)
            cur_vid.pop(wid, None)

            if t_first_pending is None:
                t_first_pending = time.perf_counter()
            if (active_w and all(pending[w] for w in active_w)) or (time.perf_counter() - t_first_pending > 10.0):
                flush_block()
        except queue.Empty:
            if t_first_pending is not None and (time.perf_counter() - t_first_pending > 10.0):
                flush_block()

        try:
            recs = res_q.get(timeout=0.1)
            if isinstance(recs, tuple) and recs[0] == "SENTINEL":
                wid = recs[1]
                active_procs -= 1
                dead_w.add(wid)
                active_w.discard(wid)
            else:
                records.extend(recs)
                if records:
                    try:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        pl.DataFrame(records).write_parquet(out_path, compression="zstd")
                    except Exception as e:
                        print(f"WARNING: Incremental save failed: {e}", file=sys.stderr, flush=True)
        except queue.Empty:
            pass

        if active_procs > 0 and res_q.empty() and prog_q.empty():
            for i, p in enumerate(procs):
                if i not in dead_w and not p.is_alive():
                    dead_w.add(i)
                    active_procs -= 1
                    active_w.discard(i)
                    try:
                        print(f"WARNING: Worker {i} ({devices[i]}) died unexpectedly.", file=sys.stderr)
                        sys.stderr.flush()
                    except Exception:
                        pass
                    # Re-queue the video that was being processed when worker died
                    if i in cur_vid:
                        lost_vid_name = cur_vid.pop(i)
                        lost_path = next((f for f in v_files if f.name == lost_vid_name), None)
                        if lost_path is not None:
                            task_q.put(lost_path)
                            print(f"  Re-queued {lost_vid_name} for retry.", file=sys.stderr, flush=True)
                    # Spawn replacement worker on same GPU if tasks remain
                    if not task_q.empty():
                        new_id = len(procs)
                        gpu = devices[i]
                        new_p = ctx.Process(target=gpu_worker_process, args=(new_id, gpu, task_q, res_q, prog_q, args.model, args.conf, args.batch_size, args.scene_detect, args.stride, args.max_scene_len, args.max_det))
                        procs.append(new_p)
                        devices.append(gpu)
                        active_w.add(new_id)
                        pending[new_id] = []
                        new_p.start()
                        active_procs += 1
                        print(f"  Spawned replacement worker {new_id} on {gpu}.", file=sys.stderr, flush=True)

    flush_block()
    for p in procs:
        p.join()

    df = pl.DataFrame(records) if records else pl.DataFrame(schema=SCHEMA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")

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
