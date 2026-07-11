#!/usr/bin/env python3
"""
Jetson CSI live camera benchmark
- CSI camera -> GStreamer/Argus/ISP -> OpenCV
- YOLO11 TensorRT person detection
- ByteTrack ID tracking
- State JSONL + per-frame metrics CSV
- 5-second countdown before measurement

This script intentionally prefers Jetson's system OpenCV build
(/usr/lib/python3.10/dist-packages), because that build has GStreamer support.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Keep the virtual environment's NumPy, but prefer Jetson's system OpenCV.
import numpy as np

SYSTEM_DIST_PACKAGES = "/usr/lib/python3.10/dist-packages"
if os.path.isdir(SYSTEM_DIST_PACKAGES):
    sys.path.insert(0, SYSTEM_DIST_PACKAGES)

import cv2  # noqa: E402

if sys.path and sys.path[0] == SYSTEM_DIST_PACKAGES:
    sys.path.pop(0)


def has_gstreamer() -> bool:
    """Return True when the loaded OpenCV build has GStreamer support."""
    for line in cv2.getBuildInformation().splitlines():
        if "GStreamer" in line:
            return "YES" in line
    return False


def csi_pipeline(
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
) -> str:
    """
    Pipeline matched to the previously working Jetson camera_capture.py.
    The sensor may run at a native mode and nvvidconv converts to the requested size.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1"
    )


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.source == "csi":
        if not has_gstreamer():
            raise RuntimeError(
                "현재 OpenCV는 GStreamer를 지원하지 않습니다.\n"
                f"cv2 path: {cv2.__file__}\n"
                "Jetson 시스템 OpenCV(/usr/lib/python3.10/dist-packages)를 사용해야 합니다."
            )

        pipeline = csi_pipeline(
            sensor_id=args.sensor_id,
            width=args.cap_width,
            height=args.cap_height,
            fps=args.cap_fps,
        )
        print("[INFO] CSI GStreamer pipeline:")
        print(pipeline)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    elif args.source == "usb":
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
        cap.set(cv2.CAP_PROP_FPS, args.cap_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    elif args.source == "file":
        if not args.input:
            raise ValueError("--source file 사용 시 --input 경로가 필요합니다.")
        cap = cv2.VideoCapture(args.input)

    else:
        raise ValueError(f"지원하지 않는 source: {args.source}")

    if not cap.isOpened():
        raise RuntimeError(
            "Camera/video open failed.\n"
            f"cv2 path: {cv2.__file__}\n"
            f"GStreamer support: {has_gstreamer()}"
        )

    return cap


def safe_cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def read_frame(
    cap: cv2.VideoCapture,
    max_failures: int = 30,
) -> np.ndarray:
    """Read a valid frame or fail after repeated errors."""
    for attempt in range(1, max_failures + 1):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        print(f"[WARN] 프레임 읽기 실패 {attempt}/{max_failures}")
        time.sleep(0.02)

    raise RuntimeError(f"프레임 읽기 {max_failures}회 연속 실패")


def warmup_model(
    model: Any,
    cap: cv2.VideoCapture,
    args: argparse.Namespace,
) -> None:
    if args.warmup_frames <= 0:
        return

    print(f"[INFO] TensorRT/YOLO warm-up: {args.warmup_frames} frames")
    for _ in range(args.warmup_frames):
        frame = read_frame(cap)
        model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=[0],
            verbose=False,
        )
        safe_cuda_sync()


def run_countdown(
    cap: cv2.VideoCapture,
    seconds: float,
    display: bool,
) -> bool:
    """
    Keep consuming live frames during the countdown.
    Returns False when the user cancels with q or ESC.
    """
    if seconds <= 0:
        return True

    print(f"[INFO] {seconds:.1f}초 후 측정을 시작합니다.")
    deadline = time.perf_counter() + seconds
    last_count = None

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break

        frame = read_frame(cap)
        count = max(1, math.ceil(remaining))

        if count != last_count:
            print(f"[COUNTDOWN] {count}")
            last_count = count

        if display:
            preview = frame.copy()
            cv2.putText(
                preview,
                f"Starting in {count}",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("HUT Live Camera Track", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return False

    print("[INFO] 측정을 시작합니다.")
    return True


def extract_objects(
    result: Any,
    previous_state: dict[int, dict[str, Any]],
    now_sec: float,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return objects

    boxes = result.boxes
    xyxy_values = boxes.xyxy.detach().cpu().numpy()
    conf_values = (
        boxes.conf.detach().cpu().numpy()
        if boxes.conf is not None
        else np.zeros(len(xyxy_values), dtype=float)
    )
    class_values = (
        boxes.cls.detach().cpu().numpy().astype(int)
        if boxes.cls is not None
        else np.zeros(len(xyxy_values), dtype=int)
    )
    track_values = (
        boxes.id.detach().cpu().numpy().astype(int)
        if boxes.id is not None
        else np.full(len(xyxy_values), -1, dtype=int)
    )

    active_ids: set[int] = set()

    for xyxy, confidence, class_id, track_id in zip(
        xyxy_values,
        conf_values,
        class_values,
        track_values,
    ):
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        previous = previous_state.get(int(track_id))
        if previous is not None:
            dt = max(now_sec - float(previous["timestamp"]), 1e-6)
            velocity = [
                (center_x - float(previous["center"][0])) / dt,
                (center_y - float(previous["center"][1])) / dt,
            ]
            previous_velocity = previous["velocity"]
            acceleration = [
                (velocity[0] - float(previous_velocity[0])) / dt,
                (velocity[1] - float(previous_velocity[1])) / dt,
            ]
        else:
            velocity = [0.0, 0.0]
            acceleration = [0.0, 0.0]

        speed = math.hypot(velocity[0], velocity[1])

        previous_state[int(track_id)] = {
            "center": [center_x, center_y],
            "velocity": velocity,
            "timestamp": now_sec,
        }
        active_ids.add(int(track_id))

        objects.append(
            {
                "id": int(track_id),
                "class_id": int(class_id),
                "class_name": "person" if int(class_id) == 0 else str(int(class_id)),
                "confidence": float(confidence),
                "bbox_xyxy": [x1, y1, x2, y2],
                "center": [center_x, center_y],
                "velocity_px_s": velocity,
                "speed_px_s": speed,
                "acceleration_px_s2": acceleration,
                "source": "live_csi_yolo_bytetrack",
            }
        )

    # Remove stale IDs so the dictionary does not grow forever.
    stale_ids = [
        track_id
        for track_id, state in previous_state.items()
        if now_sec - float(state["timestamp"]) > 10.0
    ]
    for track_id in stale_ids:
        previous_state.pop(track_id, None)

    return objects


def draw_overlay(
    frame: np.ndarray,
    objects: list[dict[str, Any]],
    average_fps: float,
    recent_fps: float,
) -> np.ndarray:
    output = frame.copy()

    for obj in objects:
        x1, y1, x2, y2 = [int(round(v)) for v in obj["bbox_xyxy"]]
        center_x, center_y = [int(round(v)) for v in obj["center"]]
        track_id = obj["id"]
        confidence = obj["confidence"]
        confirmed = track_id >= 0
        box_color = (0, 255, 0) if confirmed else (0, 255, 255)
        status = "TRACK" if confirmed else "CAND"

        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(output, (center_x, center_y), 4, (0, 0, 255), -1)
        cv2.putText(
            output,
            f"{status} id:{track_id} conf:{confidence:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            box_color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"FPS avg:{average_fps:.2f} recent:{recent_fps:.2f} persons:{len(objects)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jetson CSI live person detection/tracking benchmark"
    )

    parser.add_argument("--source", choices=["csi", "usb", "file"], default="csi")
    parser.add_argument("--input", default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sensor-id", type=int, default=0)

    parser.add_argument("--cap-width", type=int, default=1280)
    parser.add_argument("--cap-height", type=int, default=720)
    parser.add_argument("--cap-fps", type=int, default=30)

    parser.add_argument(
        "--model",
        default="models/engines/yolo11n_trt_fp16_imgsz640_static.engine",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--tracker", default="bytetrack.yaml")

    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="카메라 입력/FPS만 측정하고 사람 탐지는 하지 않음",
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument(
        "--save-video",
        default="",
        help="bbox/id/conf가 표시된 진단용 overlay MP4 저장 경로",
    )
    parser.add_argument(
        "--save-raw-video",
        default="",
        help="표시 없는 원본 카메라 MP4 저장 경로",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="저장 영상의 재생 FPS",
    )
    parser.add_argument("--start-delay", type=float, default=5.0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--print-every", type=int, default=30)

    parser.add_argument(
        "--save-jsonl",
        default="outputs/state_json/live_csi_test_state.jsonl",
    )
    parser.add_argument(
        "--save-metrics",
        default="outputs/metrics/live/live_csi_test_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("========== Environment ==========")
    print(f"Python          : {sys.executable}")
    print(f"OpenCV path     : {cv2.__file__}")
    print(f"OpenCV version  : {cv2.__version__}")
    print(f"GStreamer       : {has_gstreamer()}")

    model = None
    if not args.capture_only:
        from ultralytics import YOLO

        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"모델 파일을 찾지 못했습니다: {model_path}\n"
                "find models -type f \\( -name '*.engine' -o -name '*.pt' \\) 명령으로 확인하세요."
            )

        print(f"[INFO] 모델 로딩: {model_path}")
        model = YOLO(str(model_path))

    cap = open_capture(args)

    try:
        if model is not None:
            warmup_model(model, cap, args)

        if not run_countdown(cap, args.start_delay, args.display):
            print("[INFO] 사용자가 측정을 취소했습니다.")
            return

        jsonl_path = Path(args.save_jsonl)
        metrics_path = Path(args.save_metrics)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        previous_state: dict[int, dict[str, Any]] = {}

        start_time = time.perf_counter()
        recent_start = start_time
        recent_frames = 0
        frame_index = 0

        overlay_writer = None
        raw_writer = None
        video_size = (args.cap_width, args.cap_height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if args.save_video:
            overlay_path = Path(args.save_video)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_writer = cv2.VideoWriter(
                str(overlay_path),
                fourcc,
                args.video_fps,
                video_size,
            )
            if not overlay_writer.isOpened():
                raise RuntimeError(f"overlay video writer open failed: {overlay_path}")
            print(f"[INFO] Overlay video: {overlay_path}")

        if args.save_raw_video:
            raw_path = Path(args.save_raw_video)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_writer = cv2.VideoWriter(
                str(raw_path),
                fourcc,
                args.video_fps,
                video_size,
            )
            if not raw_writer.isOpened():
                raise RuntimeError(f"raw video writer open failed: {raw_path}")
            print(f"[INFO] Raw video: {raw_path}")

        with jsonl_path.open("w", encoding="utf-8") as jsonl_file, metrics_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow(
                [
                    "frame",
                    "timestamp_sec",
                    "capture_ms",
                    "infer_track_ms",
                    "state_ms",
                    "display_ms",
                    "total_loop_ms",
                    "person_count",
                    "average_fps",
                    "recent_fps",
                ]
            )

            while True:
                loop_start = time.perf_counter()
                elapsed_before_read = loop_start - start_time

                if args.duration > 0 and elapsed_before_read >= args.duration:
                    break

                capture_start = time.perf_counter()
                frame = read_frame(cap)
                capture_end = time.perf_counter()
                capture_ms = (capture_end - capture_start) * 1000.0

                inference_ms = 0.0
                objects: list[dict[str, Any]] = []

                if model is not None:
                    inference_start = time.perf_counter()
                    results = model.track(
                        frame,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        classes=[0],
                        tracker=args.tracker,
                        persist=True,
                        verbose=False,
                    )
                    safe_cuda_sync()
                    inference_end = time.perf_counter()
                    inference_ms = (inference_end - inference_start) * 1000.0

                    state_start = time.perf_counter()
                    timestamp_sec = inference_end - start_time
                    first_result = results[0] if results else None
                    objects = extract_objects(
                        first_result,
                        previous_state,
                        timestamp_sec,
                    )
                else:
                    state_start = time.perf_counter()
                    timestamp_sec = state_start - start_time

                state = {
                    "frame": frame_index,
                    "timestamp": timestamp_sec,
                    "scene_state": "OCCUPIED" if objects else "EMPTY",
                    "object_count": len(objects),
                    "objects": objects,
                }
                jsonl_file.write(json.dumps(state, ensure_ascii=False) + "\n")
                state_end = time.perf_counter()
                state_ms = (state_end - state_start) * 1000.0

                elapsed = max(state_end - start_time, 1e-6)
                average_fps = (frame_index + 1) / elapsed

                recent_frames += 1
                recent_elapsed = state_end - recent_start
                recent_fps = (
                    recent_frames / recent_elapsed
                    if recent_elapsed > 0
                    else 0.0
                )

                display_start = time.perf_counter()

                overlay = None
                if args.display or overlay_writer is not None:
                    overlay = draw_overlay(
                        frame,
                        objects,
                        average_fps,
                        recent_fps,
                    )

                if overlay_writer is not None and overlay is not None:
                    if (overlay.shape[1], overlay.shape[0]) != video_size:
                        overlay = cv2.resize(overlay, video_size)
                    overlay_writer.write(overlay)

                if raw_writer is not None:
                    raw_frame = frame
                    if (raw_frame.shape[1], raw_frame.shape[0]) != video_size:
                        raw_frame = cv2.resize(raw_frame, video_size)
                    raw_writer.write(raw_frame)

                if args.display and overlay is not None:
                    cv2.imshow("HUT Live Camera Track", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                display_end = time.perf_counter()
                display_ms = (display_end - display_start) * 1000.0

                loop_end = time.perf_counter()
                total_loop_ms = (loop_end - loop_start) * 1000.0

                writer.writerow(
                    [
                        frame_index,
                        f"{timestamp_sec:.6f}",
                        f"{capture_ms:.3f}",
                        f"{inference_ms:.3f}",
                        f"{state_ms:.3f}",
                        f"{display_ms:.3f}",
                        f"{total_loop_ms:.3f}",
                        len(objects),
                        f"{average_fps:.3f}",
                        f"{recent_fps:.3f}",
                    ]
                )

                if frame_index % args.print_every == 0:
                    print(
                        f"[LIVE] frame={frame_index} "
                        f"avg_fps={average_fps:.2f} "
                        f"recent_fps={recent_fps:.2f} "
                        f"capture={capture_ms:.2f}ms "
                        f"infer+track={inference_ms:.2f}ms "
                        f"state={state_ms:.2f}ms "
                        f"persons={len(objects)}"
                    )
                    recent_start = loop_end
                    recent_frames = 0

                frame_index += 1

        total_elapsed = time.perf_counter() - start_time
        final_fps = frame_index / max(total_elapsed, 1e-6)

        print("========== Live Camera Benchmark Done ==========")
        print(f"Frames        : {frame_index}")
        print(f"Elapsed sec   : {total_elapsed:.3f}")
        print(f"Average FPS   : {final_fps:.3f}")
        print(f"State JSONL   : {jsonl_path}")
        print(f"Metrics CSV   : {metrics_path}")

    finally:
        if "overlay_writer" in locals() and overlay_writer is not None:
            overlay_writer.release()
        if "raw_writer" in locals() and raw_writer is not None:
            raw_writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
