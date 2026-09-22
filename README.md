# AIC_indexer

## 1. Installation

Requires Python $\ge 3.10$ and `uv`.

```bash
# Install uv (Linux / macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

No manual `pip install` required. Script uses PEP 723 inline script metadata (`# /// script ... ///`). `uv run` resolves and caches dependencies automatically (`polars`, `duckdb`, `pyarrow`, `orjson`, `opencv-python-headless`, `rich`, `ultralytics`, `scenedetect`).

---

## 2. Usage Examples

```bash
# Mode 1: Recursive scan of one or more folders
uv run AIC_indexer.py -r /path/to/videos_folder1 /path/to/videos_folder2

# Mode 2: Specific video files
uv run AIC_indexer.py /path/to/vid1.mp4 /path/to/vid2.mp4

# Mode 3: Combined files and recursive folders
uv run AIC_indexer.py /path/to/vid1.mp4 /path/to/vid2.mp4 -r /path/to/videos_folder

# Mode 4: Custom output and stride
uv run AIC_indexer.py -r /path/to/videos -o /kaggle/working/obj-idx.parquet --stride 10 --batch-size 32
```

---

## 3. How Arguments Are Parsed

| Argument | Type | Default | Description |
|---|---|---|---|
| `inputs` | Positional (`nargs='*'`) | `[]` | Explicit video file paths or directory paths. |
| `-r`, `--recursive` | Option (`nargs='*'`) | `None` | Enables recursive traversal. Folders passed after `-r` are scanned recursively (`**/*`). Folders passed in `inputs` are also scanned recursively if `-r` is present. |
| `-o`, `--output` | Option | `obj-idx.parquet` | Destination Parquet file path. |
| `--stride` | Option (`int`) | `10` | Frame stride. Only 1 frame every $N$ frames is decoded and passed to detector (`0, 10, 20, ...`). Intermediate frames skipped via fast `cap.grab()`. |
| `--scene-detect` | Flag | `False` | Enables PySceneDetect adaptive shot detection. Extracts middle keyframe per shot. |
| `--max-scene-len` | Option (`float`) | `10.0` | Maximum shot duration (seconds) before splitting into sub-shots (when `--scene-detect` active). |
| `--model` | Option | `yolo26n.pt` | YOLO model name or checkpoint path. |
| `--conf` | Option (`float`) | `0.25` | Object detection confidence threshold. |
| `--batch-size` | Option (`int`) | `32` | Frame inference batch size per GPU. |
| `--num-gpus` | Option (`int`) | `0` | Number of GPUs to use. `0` auto-detects all GPUs (e.g. 2 workers for Kaggle 2x T4). |

### Argument Resolution Rules:
- **`AIC_indexer.py -r folder1 folder2`**: `-r` collects `['folder1', 'folder2']`, scans recursively for video extensions.
- **`AIC_indexer.py vid1.mp4 vid2.mp4`**: `inputs` collects files, processed directly.
- **`AIC_indexer.py vid1.mp4 -r folder1`**: `inputs` collects `vid1.mp4`, `-r` scans `folder1` recursively. Duplicate files deduplicated preserving order.

---

## 4. Running on Kaggle (2x T4 Multi-GPU Setup)

### Step 1: Configure Notebook Environment
1. In Kaggle Notebook right sidebar: **Settings** $\rightarrow$ **Accelerator** $\rightarrow$ select **GPU T4 x2**.
2. Verify **Internet** is toggled **ON** (required for `uv` to download dependencies and YOLO weights on first run).

### Step 2: Install UV in Notebook Cell
Run in first code cell:
```bash
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"
```

### Step 3: Run Script
Run in second code cell:
```bash
!uv run AIC_indexer.py \
    -r /kaggle/input/your-video-dataset \
    -o obj-idx.parquet \
    --num-gpus 2 \
    --batch-size 64 \
    --stride 10
```

### How Decisions are made:
- **2x T4 GPUs (32 GB VRAM total)**:
  - Script detects `torch.cuda.device_count() == 2`.
  - Spawns 2 isolated worker processes using `mp.get_context("spawn")`: Worker 0 on `cuda:0`, Worker 1 on `cuda:1`.
  - Videos partitioned round-robin across workers so both GPUs run at 100% utilization concurrently.
- **4 vCPUs**:
  - Video decoding runs concurrently across workers.
  - Fast selective decoding (`cap.grab()`) skips color decoding for 90% of frames, preventing CPU decoding bottlenecks.
- **30 GB System RAM**:
  - Streaming IPC queue: keyframe records stream directly from workers to the writer, preventing RAM bloat on large collections.
- **Disk I/O**:
  - Writes directly to `/kaggle/working/obj-idx.parquet` with Zstandard compression via Polars. Zero temporary image files dumped to disk.

### Step 4: Verify Output in Notebook
```python
import polars as pl
df = pl.read_parquet("/kaggle/working/obj-idx.parquet")
print(f"Total indexed keyframes: {len(df)}")
print(df.select(["video_id", "time", "frame_idx", "objects", "txt"]).head(10))
```

---

## 5. Parquet Schema (`obj-idx.parquet`)

| Column | Type | Description |
|---|---|---|
| `video_id` | `String` | Video stem filename (e.g. `L01_V001`) |
| `time` | `Float64` | Keyframe timestamp (seconds) |
| `frame_idx` | `Int64` | Keyframe frame index |
| `start_time` | `Float64` | Segment start timestamp (seconds) |
| `end_time` | `Float64` | Segment end timestamp (seconds) |
| `start_frame` | `Int64` | Segment start frame number |
| `end_frame` | `Int64` | Segment end frame number |
| `fps` | `Float32` | Video frame rate |
| `width` | `Int32` | Frame width |
| `height` | `Int32` | Frame height |
| `objects` | `List(String)` | Unique detected labels in keyframe |
| `scores` | `List(Float32)` | Confidence scores per detection |
| `boxes` | `List(List(Float32))` | Normalized `[y0, x0, y1, x1]` coordinates |
| `labels` | `List(String)` | Label for each bounding box |
| `txt` | `String` | VISIONE $7 \times 7$ grid tokens (`2ccar 3dperson`) |
| `objects_str` | `String` | VISIONE surrogate count tokens (`4wccar1\|6 4wcperson1\|5`) |
| `colors` | `List(String)` | Frame color palette tags |
| `is_monochrome` | `Boolean` | True if frame is grayscale / monochrome |
| `video_path` | `String` | Source video file path |

---

## 6. Querying the Index

Query with DuckDB or Polars:

```python
import duckdb

# Find exact video and timestamp where a car is on the left side
con = duckdb.connect()
query = """
SELECT video_id, time, frame_idx, txt, objects_str
FROM 'obj-idx.parquet'
WHERE list_contains(objects, 'car')
  AND txt LIKE '%bcar%'
ORDER BY time ASC
LIMIT 10;
"""
print(con.execute(query).df())
```

---

## 7. Detectable Classes

### A. Object Classes (80 COCO Categories via YOLO26)

| Category | Classes |
|---|---|
| **People** | `person` |
| **Vehicles** | `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat` |
| **Outdoor / Traffic** | `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench` |
| **Animals** | `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe` |
| **Accessories** | `backpack`, `umbrella`, `handbag`, `tie`, `suitcase` |
| **Sports** | `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket` |
| **Kitchenware** | `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl` |
| **Food** | `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake` |
| **Furniture** | `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet` |
| **Electronics & Appliances** | `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`, `clock` |
| **Indoor / Misc** | `book`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush` |

### B. Color & Palette Classes (11 HSV Categories)

| Type | Labels |
|---|---|
| **Chromatic** | `red`, `orange`, `yellow`, `green`, `blue`, `purple`, `pink` |
| **Achromatic** | `black`, `white`, `grey` |
| **Frame Type** | `monochrome` (grayscale / low saturation flag) |

