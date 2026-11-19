"""Webcam + Gameplay split-screen layout for 9:16 vertical shorts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import Settings
from app.schemas import TransitionSettings, TransitionStyle
from app.services.media import MediaInfo, escape_filter_path, filter_relative_path

logger = logging.getLogger(__name__)

# Default presets file location
PRESETS_FILENAME = "webcam_presets.json"


@dataclass(slots=True)
class WebcamBox:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebcamBox:
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
        )


def parse_webcam_region(raw: Any) -> tuple[int, int, int, int] | None:
    """Parse webcam region from tuple, list, dict, or string."""
    if not raw:
        return None
    try:
        if isinstance(raw, (tuple, list)) and len(raw) == 4:
            x, y, w, h = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        elif isinstance(raw, dict) and {"x", "y", "w", "h"}.issubset(raw.keys()):
            x, y, w, h = (int(raw["x"]), int(raw["y"]), int(raw["w"]), int(raw["h"]))
        elif isinstance(raw, str):
            import re

            parts = [p for p in re.split(r"[,:\s]+", raw.strip()) if p]
            if len(parts) == 4:
                x, y, w, h = (
                    int(float(parts[0])),
                    int(float(parts[1])),
                    int(float(parts[2])),
                    int(float(parts[3])),
                )
            else:
                return None
        else:
            return None

        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
        return (x, y, w, h)
    except (ValueError, TypeError):
        return None


def get_presets_file_path(settings: Settings) -> Path:
    return settings.data_dir / PRESETS_FILENAME


def load_webcam_preset(
    channel: str | None,
    source_size: tuple[int, int],
    presets_file: Path,
) -> tuple[int, int, int, int] | None:
    """Retrieve saved webcam box for a channel, scaling coordinates if resolution differs."""
    if not channel or not presets_file.exists():
        return None
    try:
        content = json.loads(presets_file.read_text(encoding="utf-8"))
        norm_channel = channel.strip().lower()
        preset = content.get(norm_channel)
        if not preset:
            return None
        box = WebcamBox.from_dict(preset["box"])
        orig_w = int(preset.get("source_width") or source_size[0])
        orig_h = int(preset.get("source_height") or source_size[1])
        sw, sh = source_size
        if orig_w > 0 and orig_h > 0 and (orig_w != sw or orig_h != sh):
            scale_x = sw / orig_w
            scale_y = sh / orig_h
            return (
                int(box.x * scale_x),
                int(box.y * scale_y),
                int(box.w * scale_x),
                int(box.h * scale_y),
            )
        return box.as_tuple()
    except Exception as exc:
        logger.warning("Failed to load webcam preset for %s: %s", channel, exc)
        return None


def save_webcam_preset(
    channel: str | None,
    region: tuple[int, int, int, int],
    source_size: tuple[int, int],
    presets_file: Path,
) -> None:
    """Save the webcam box for a channel to avoid re-detecting next time."""
    if not channel:
        return
    try:
        presets_file.parent.mkdir(parents=True, exist_ok=True)
        content: dict[str, Any] = {}
        if presets_file.exists():
            try:
                content = json.loads(presets_file.read_text(encoding="utf-8"))
            except Exception:
                content = {}
        norm_channel = channel.strip().lower()
        content[norm_channel] = {
            "channel": channel,
            "box": {"x": region[0], "y": region[1], "w": region[2], "h": region[3]},
            "source_width": source_size[0],
            "source_height": source_size[1],
        }
        presets_file.write_text(json.dumps(content, indent=2), encoding="utf-8")
        logger.info("Saved webcam preset for channel '%s'", channel)
    except Exception as exc:
        logger.warning("Failed to save webcam preset for %s: %s", channel, exc)


def detect_webcam_region(
    source_path: Path,
    *,
    duration: float,
    media_width: int,
    media_height: int,
    sample_count: int = 10,
) -> tuple[int, int, int, int] | None:
    """Auto-detect the facecam overlay box using multi-frame OpenCV face & contour detection.

    Returns (x, y, w, h) in source pixel coordinates, or None if no facecam overlay is detected.
    """
    if not source_path.exists() or media_width <= 0 or media_height <= 0:
        return None

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        logger.warning("Haar cascade not found at %s", cascade_path)
        return None

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        logger.warning("Failed to load OpenCV face cascade")
        return None

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        logger.warning("Could not open video source for webcam detection: %s", source_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(duration * fps)

    step = max(1, total_frames // (sample_count + 1))
    frame_indices = [step * (i + 1) for i in range(sample_count)]

    boxes: list[tuple[int, int, int, int]] = []

    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Facecam faces in gaming streams are typically 3% to 30% of frame width
            min_face = max(24, int(w * 0.035))
            max_face = int(w * 0.35)

            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(min_face, min_face),
                maxSize=(max_face, max_face),
            )
            if len(faces) == 0:
                continue

            for fx, fy, fw, fh in faces:
                # Disqualify faces that take up most of the screen (not an overlay webcam)
                if fw > w * 0.45 or fh > h * 0.45:
                    continue

                # Facecam overlays are usually placed in corners or along borders
                cx, cy = fx + fw / 2, fy + fh / 2
                in_corner_or_edge = cx < w * 0.40 or cx > w * 0.60 or cy < h * 0.40 or cy > h * 0.60
                if not in_corner_or_edge:
                    continue

                # Expand face to full facecam box (webcam window)
                # Attempt to find bounding rectangular border via Canny edges
                pad_x = int(fw * 1.0)
                pad_y_top = int(fh * 0.6)
                pad_y_bottom = int(fh * 1.4)

                roi_x0 = max(0, fx - pad_x)
                roi_y0 = max(0, fy - pad_y_top)
                roi_x1 = min(w, fx + fw + pad_x)
                roi_y1 = min(h, fy + fh + pad_y_bottom)

                roi_w = roi_x1 - roi_x0
                roi_h = roi_y1 - roi_y0
                if roi_w < 50 or roi_h < 50:
                    continue

                roi = gray[roi_y0:roi_y1, roi_x0:roi_x1]
                edges = cv2.Canny(roi, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detected_box: tuple[int, int, int, int] | None = None
                best_area = 0
                for c in contours:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                    if 4 <= len(approx) <= 6:
                        bx, by, bw, bh = cv2.boundingRect(approx)
                        area = bw * bh
                        aspect = bw / max(1, bh)
                        # Typical webcam aspect ratios are 1.1 - 1.85 (4:3 or 16:9)
                        if area > (fw * fh * 1.5) and 1.1 <= aspect <= 1.9 and area > best_area:
                            best_area = area
                            detected_box = (roi_x0 + bx, roi_y0 + by, bw, bh)

                if detected_box is None:
                    # Fallback to standard 16:9 box around the face
                    box_w = min(w, int(fw * 2.8))
                    box_h = min(h, int(box_w * 9 / 16))
                    box_x = max(0, min(w - box_w, int(cx - box_w / 2)))
                    box_y = max(0, min(h - box_h, int(cy - box_h * 0.42)))
                    detected_box = (box_x, box_y, box_w, box_h)

                boxes.append(detected_box)
                break  # Take primary candidate per frame
    finally:
        cap.release()

    if not boxes:
        logger.info("No webcam overlay detected across sampled frames")
        return None

    # Compute stable median box
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    ws = [b[2] for b in boxes]
    hs = [b[3] for b in boxes]

    med_x = int(np.median(xs))
    med_y = int(np.median(ys))
    med_w = int(np.median(ws))
    med_h = int(np.median(hs))

    # Clamp to video boundaries
    med_x = max(0, min(media_width - 10, med_x))
    med_y = max(0, min(media_height - 10, med_y))
    med_w = max(20, min(media_width - med_x, med_w))
    med_h = max(20, min(media_height - med_y, med_h))

    logger.info(
        "Auto-detected stable webcam overlay: x=%d, y=%d, w=%d, h=%d (from %d samples)",
        med_x,
        med_y,
        med_w,
        med_h,
        len(boxes),
    )
    return (med_x, med_y, med_w, med_h)


def compute_smart_center_x(
    source_path: Path,
    *,
    duration: float,
    media_width: int,
    media_height: int,
    crop_width: int,
    sample_count: int = 8,
) -> int:
    """Analyze motion across frames to find the horizontal center of activity for gameplay."""
    max_x = max(0, media_width - crop_width)
    if max_x == 0:
        return 0

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return max_x // 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(duration * fps)
    step = max(1, total_frames // (sample_count + 1))

    motion_x_samples: list[float] = []
    prev_gray: np.ndarray | None = None

    try:
        for i in range(sample_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, step * (i + 1))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            # Downscale for fast motion analysis
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                # Sum difference along columns
                col_motion = np.sum(diff, axis=0, dtype=np.float32)
                total_motion = np.sum(col_motion)
                if total_motion > 100:
                    cols = np.arange(len(col_motion))
                    center_col = float(np.sum(cols * col_motion) / total_motion)
                    # Scale back to media_width
                    scaled_x = center_col * (media_width / 320.0)
                    motion_x_samples.append(scaled_x)
            prev_gray = gray
    finally:
        cap.release()

    if not motion_x_samples:
        return max_x // 2

    avg_motion_x = float(np.median(motion_x_samples))
    clamped_x = int(max(0, min(max_x, avg_motion_x - crop_width / 2)))
    return clamped_x


def build_facecam_filter_graph(
    *,
    media: MediaInfo,
    output_size: tuple[int, int],
    webcam_region: tuple[int, int, int, int],
    webcam_height: int,
    gap: int = 0,
    divider: bool = False,
    remove_webcam_from_gameplay: bool = False,
    smart_center_x: int | None = None,
    caption_path: Path | None = None,
    transitions: TransitionSettings | None = None,
    settings: Settings | None = None,
    fonts_dir: Path | None = None,
    work_dir: Path | None = None,
    duration: float = 30.0,
    index: int = 0,
) -> tuple[str, str]:
    """Construct FFmpeg filter graph for Facecam Top + Gameplay Fill layout.

    Layout:
      - Top: 1080 x webcam_height (facecam aspect-fill cropped and scaled)
      - Gap: Optional gap or divider line
      - Bottom: 1080 x (1920 - webcam_height - gap) (gameplay center-cropped and scaled)
      - Subtitles: Placed in the lower area of the gameplay panel (safe zone)
    """
    if transitions is None:
        transitions = TransitionSettings()
    if settings is None:
        settings = Settings()
    if fonts_dir is None:
        fonts_dir = Path("/tmp")
    if work_dir is None:
        work_dir = Path("/tmp")

    width, height = output_size
    fps = settings.output_fps
    pre_label = f"clip{index}facecam_pre"
    out_label = "vout"

    wx, wy, ww, wh = webcam_region
    target_top_h = max(240, min(height - 300, webcam_height))
    target_gap = max(0, min(50, gap))
    target_bottom_h = height - target_top_h - target_gap

    # 1. Top Panel (Webcam): Scale to 1080 x target_top_h without stretching
    aspect_top = width / target_top_h
    if ww / max(1, wh) > aspect_top:
        cam_crop_w = round(wh * aspect_top)
        cam_crop_h = wh
        cam_crop_x = wx + (ww - cam_crop_w) // 2
        cam_crop_y = wy
    else:
        cam_crop_w = ww
        cam_crop_h = round(ww / aspect_top)
        cam_crop_x = wx
        cam_crop_y = wy + (wh - cam_crop_h) // 2

    cam_crop_x = max(0, min(media.width - cam_crop_w, cam_crop_x))
    cam_crop_y = max(0, min(media.height - cam_crop_h, cam_crop_y))

    # 2. Bottom Panel (Gameplay): Fill 1080 x target_bottom_h
    aspect_bottom = width / target_bottom_h
    source_aspect = media.width / max(1, media.height)

    if source_aspect > aspect_bottom:
        game_crop_h = media.height
        game_crop_w = round(media.height * aspect_bottom)
    else:
        game_crop_w = media.width
        game_crop_h = round(media.width / aspect_bottom)

    game_crop_w = min(media.width, game_crop_w)
    game_crop_h = min(media.height, game_crop_h)

    max_gx = max(0, media.width - game_crop_w)
    max_gy = max(0, media.height - game_crop_h)

    if smart_center_x is not None:
        gx = max(0, min(max_gx, smart_center_x))
    elif remove_webcam_from_gameplay:
        # If webcam is on the left, shift gameplay crop to the right, and vice versa
        webcam_center_x = wx + ww / 2
        gx = max_gx if webcam_center_x < media.width / 2 else 0
    else:
        gx = max_gx // 2

    gy = max_gy // 2

    caption_filter = ""
    if caption_path is not None:
        escaped_ass = escape_filter_path(filter_relative_path(caption_path, work_dir))
        escaped_fonts = escape_filter_path(filter_relative_path(fonts_dir, work_dir))
        caption_filter = f",ass=filename='{escaped_ass}':fontsdir='{escaped_fonts}'"

    # Build the split and stack graph
    lines: list[str] = [
        "[0:v]split=2[in_cam][in_game]",
        (
            f"[in_cam]crop={cam_crop_w}:{cam_crop_h}:{cam_crop_x}:{cam_crop_y},"
            f"scale={width}:{target_top_h}:flags=lanczos,setsar=1[panel_top]"
        ),
        (
            f"[in_game]crop={game_crop_w}:{game_crop_h}:{gx}:{gy},"
            f"scale={width}:{target_bottom_h}:flags=lanczos,setsar=1[panel_bottom]"
        ),
    ]

    if target_gap > 0:
        lines.append(f"color=c=black:s={width}x{target_gap}:d={duration:.3f},fps={fps}[panel_gap]")
        lines.append("[panel_top][panel_gap][panel_bottom]vstack=inputs=3[stacked]")
    else:
        lines.append("[panel_top][panel_bottom]vstack=inputs=2[stacked]")

    if divider:
        divider_y = target_top_h - 1
        lines.append(
            f"[stacked]drawbox=y={divider_y}:w={width}:h=2:color=white@0.6:t=fill,"
            f"fps={fps}{caption_filter},format=yuv420p[{pre_label}]"
        )
    else:
        lines.append(f"[stacked]fps={fps}{caption_filter},format=yuv420p[{pre_label}]")

    chain = ";".join(lines)

    # Attach transitions
    if not transitions.enabled:
        chain += f";[{pre_label}]null[{out_label}]"
        return chain, out_label

    transition_duration = max(0.2, min(transitions.duration, duration / 3.0))
    if transitions.style == TransitionStyle.FADE:
        chain += (
            f";[{pre_label}]fade=t=in:st=0:d={transition_duration:.3f},"
            f"fade=t=out:st={max(0.0, duration - transition_duration):.3f}:"
            f"d={transition_duration:.3f},format=yuv420p[{out_label}]"
        )
    elif transitions.style == TransitionStyle.ZOOM:
        zoom = 0.08
        position = f"clip(on/({transition_duration:.3f}*{fps}),0,1)"
        end_position = (
            f"clip((on-{max(0.0, duration - transition_duration) * fps:.1f})/"
            f"({transition_duration:.3f}*{fps}),0,1)"
        )

        def _smooth(p: str) -> str:
            return f"({p}*{p}*(3-2*{p}))"

        zoom_expr = (
            f"{1 + zoom:.3f}-{zoom:.3f}*{_smooth(position)}" f"+{zoom:.3f}*{_smooth(end_position)}"
        )
        chain += (
            f";[{pre_label}]zoompan=z='{zoom_expr}':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:"
            f"s={width}x{height}:fps={fps},format=yuv420p[{out_label}]"
        )
    else:  # SLIDE

        def _smooth(p: str) -> str:
            return f"({p}*{p}*(3-2*{p}))"

        percent = f"clip(t/{transition_duration:.3f},0,1)"
        end_percent = (
            f"clip((t-{max(0.0, duration - transition_duration):.3f})/"
            f"{transition_duration:.3f},0,1)"
        )
        x_expr = (
            f"if(lt(t,{transition_duration:.3f}),"
            f"{width}*(1-{_smooth(percent)}),"
            f"if(gt(t,{max(0.0, duration - transition_duration):.3f}),"
            f"-{width}*{_smooth(end_percent)},0))"
        )
        chain += (
            f";nullsrc=s={width}x{height}:d={duration:.3f}:r={fps}[slidebg];"
            f"[slidebg][{pre_label}]overlay=x='{x_expr}':y=0:shortest=1,"
            f"format=yuv420p[{out_label}]"
        )

    return chain, out_label
