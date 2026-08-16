from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import Direction, RenderOptions


ProgressCallback = Callable[[float], None]


def _runtime_threads(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


FFMPEG_THREADS = _runtime_threads("FFMPEG_THREADS", 2)
cv2.setNumThreads(_runtime_threads("OPENCV_THREADS", 1))


def _smoothstep(edge0: float, edge1: float, value: np.ndarray | float) -> np.ndarray:
    width = np.maximum(np.asarray(edge1) - np.asarray(edge0), 1e-6)
    scaled = np.clip((value - edge0) / width, 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _normalize_percentile(image: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    scale = float(np.percentile(image, percentile))
    if scale <= 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip(image / scale, 0.0, 1.0).astype(np.float32)


def _hex_rgb(value: str) -> np.ndarray:
    return np.array([int(value[index : index + 2], 16) for index in (1, 3, 5)], dtype=np.float32)


def _fit_image(image: np.ndarray, output_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if output_width and output_width != width:
        target_height = max(2, round(height * output_width / width))
        interpolation = cv2.INTER_AREA if output_width < width else cv2.INTER_CUBIC
        image = cv2.resize(image, (output_width, target_height), interpolation=interpolation)
    height, width = image.shape[:2]
    even_width = width - (width % 2)
    even_height = height - (height % 2)
    return image[:even_height, :even_width]


def _paper_background(height: int, width: int, color: str) -> np.ndarray:
    base = np.empty((height, width, 3), dtype=np.float32)
    base[:] = _hex_rgb(color)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx - width / 2) / max(width / 2, 1)
    ny = (yy - height / 2) / max(height / 2, 1)
    vignette = np.clip((nx * nx + ny * ny - 0.35) * 7.0, 0.0, 1.0)[..., None]
    return np.clip(base * (1.0 - 0.055 * vignette), 0, 255)


def build_detail_mask(rgb: np.ndarray) -> np.ndarray:
    """Create a soft mask for colored contours, hatching and high-frequency detail."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    luminance = cv2.GaussianBlur(gray, (0, 0), 0.75)
    gx = cv2.Scharr(luminance, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(luminance, cv2.CV_32F, 0, 1)
    luminance_edges = _normalize_percentile(cv2.magnitude(gx, gy), 97.5)

    color_edges = np.zeros_like(gray)
    for channel in cv2.split(lab):
        channel = cv2.GaussianBlur(channel, (0, 0), 0.9)
        cgx = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
        cgy = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
        color_edges = np.maximum(color_edges, cv2.magnitude(cgx, cgy))
    color_edges = _normalize_percentile(color_edges, 98.0)

    local_mean = cv2.boxFilter(gray, cv2.CV_32F, (9, 9), normalize=True)
    local_square = cv2.boxFilter(gray * gray, cv2.CV_32F, (9, 9), normalize=True)
    local_variance = _normalize_percentile(
        np.sqrt(np.maximum(local_square - local_mean * local_mean, 0.0)), 97.0
    )

    fine = np.abs(
        cv2.GaussianBlur(gray, (0, 0), 0.7)
        - cv2.GaussianBlur(gray, (0, 0), 2.4)
    )
    fine = _normalize_percentile(fine, 97.0)
    darkness_support = np.clip((0.78 - gray) / 0.55, 0.0, 1.0) * local_variance

    feature = np.maximum(0.60 * luminance_edges, 0.48 * color_edges)
    feature = np.maximum(feature, 0.72 * local_variance)
    feature = np.maximum(feature, 0.78 * fine)
    feature = np.maximum(feature, 0.42 * darkness_support)
    mask = _smoothstep(0.10, 0.46, feature)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.65)
    return np.clip(mask, 0.0, 1.0)


def _low_frequency_noise(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    small_h = max(4, math.ceil(height / 100))
    small_w = max(4, math.ceil(width / 100))
    noise = rng.random((small_h, small_w), dtype=np.float32)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), max(height, width) / 100.0)
    noise -= noise.min()
    return noise / max(float(noise.max()), 1e-6)


def build_detail_arrival(
    height: int,
    width: int,
    direction: Direction,
    rng: np.random.Generator,
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    if direction == Direction.right_to_left:
        base = 1.0 - x
    elif direction == Direction.left_to_right:
        base = x
    elif direction == Direction.top_to_bottom:
        base = y
    elif direction == Direction.bottom_to_top:
        base = 1.0 - y
    elif direction == Direction.center_out:
        base = np.sqrt(((x - 0.5) / 0.71) ** 2 + ((y - 0.5) / 0.71) ** 2)
    else:
        base = 0.56 * (1.0 - x) + 0.22 * y + 0.22 * np.sin(y * math.pi * 3.0) ** 2

    noise = _low_frequency_noise(height, width, rng)
    arrival = base + (noise - 0.5) * 0.24
    arrival -= float(arrival.min())
    arrival /= max(float(arrival.max()), 1e-6)
    return arrival.astype(np.float32)


def _ordered_brush_points(
    height: int,
    width: int,
    radius: float,
    direction: Direction,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    spacing = radius * 1.08
    xs = np.arange(-radius * 0.2, width + radius * 0.2 + spacing, spacing)
    ys = np.arange(-radius * 0.2, height + radius * 0.2 + spacing, spacing)
    points: list[tuple[float, float, float]] = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else xs[::-1]
        for x in row_xs:
            jitter_x = rng.uniform(-0.16, 0.16) * radius
            jitter_y = rng.uniform(-0.16, 0.16) * radius
            scale = rng.uniform(0.88, 1.14)
            points.append((float(x + jitter_x), float(y + jitter_y), float(radius * scale)))

    def priority(point: tuple[float, float, float]) -> float:
        x, y, _ = point
        nx = x / max(width, 1)
        ny = y / max(height, 1)
        if direction == Direction.right_to_left:
            return 1.0 - nx + 0.10 * math.sin(ny * math.pi * 4.0)
        if direction == Direction.left_to_right:
            return nx + 0.10 * math.sin(ny * math.pi * 4.0)
        if direction == Direction.top_to_bottom:
            return ny + 0.10 * math.sin(nx * math.pi * 4.0)
        if direction == Direction.bottom_to_top:
            return 1.0 - ny + 0.10 * math.sin(nx * math.pi * 4.0)
        if direction == Direction.center_out:
            return math.hypot(nx - 0.5, ny - 0.5)
        return 0.58 * (1.0 - nx) + 0.20 * ny + 0.22 * math.sin((nx + ny) * math.pi * 2.0)

    decorated = [(priority(point) + rng.uniform(-0.045, 0.045), point) for point in points]
    decorated.sort(key=lambda item: item[0])
    return [point for _, point in decorated]


def build_fill_arrival(
    height: int,
    width: int,
    radius_ratio: float,
    direction: Direction,
    rng: np.random.Generator,
) -> np.ndarray:
    radius = max(8.0, min(height, width) * radius_ratio)
    points = _ordered_brush_points(height, width, radius, direction, rng)
    arrival = np.full((height, width), np.inf, dtype=np.float32)
    count = max(len(points) - 1, 1)

    for index, (cx, cy, point_radius) in enumerate(points):
        x0 = max(0, int(cx - point_radius))
        x1 = min(width, int(cx + point_radius) + 1)
        y0 = max(0, int(cy - point_radius))
        y1 = min(height, int(cy + point_radius) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / point_radius
        inside = distance <= 1.0
        stamp_time = index / count
        candidate = stamp_time + distance.astype(np.float32) * (0.035 / max(radius_ratio, 0.03))
        region = arrival[y0:y1, x0:x1]
        np.minimum(region, np.where(inside, candidate, np.inf), out=region)

    finite = np.isfinite(arrival)
    if not finite.all():
        arrival[~finite] = 1.0
    arrival -= float(arrival.min())
    arrival /= max(float(arrival.max()), 1e-6)
    return arrival


def render_video(
    input_path: Path,
    output_path: Path,
    options: RenderOptions,
    progress: ProgressCallback,
) -> tuple[int, int, int]:
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        rgb = np.asarray(image)

    rgb = _fit_image(rgb, options.output_width)
    height, width = rgb.shape[:2]
    total_frames = max(2, round(options.duration * options.fps))
    hold_frames = min(total_frames - 1, round(options.final_hold * options.fps))
    animation_frames = max(1, total_frames - hold_frames)

    rng = np.random.default_rng(options.seed)
    original = rgb.astype(np.float32)
    background = _paper_background(height, width, options.background)
    detail_alpha = build_detail_mask(rgb)
    detail_arrival = build_detail_arrival(height, width, options.direction, rng)
    fill_arrival = build_fill_arrival(
        height, width, options.brush_radius, options.direction, rng
    )

    temporary = output_path.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(options.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-threads",
        str(FFMPEG_THREADS),
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("FFmpeg no abrió su entrada estándar")

    detail_end = options.detail_ratio
    fill_start = max(0.0, detail_end - options.fill_overlap)
    try:
        for frame_index in range(total_frames):
            if frame_index >= animation_frames:
                frame = original
            else:
                timeline = frame_index / max(animation_frames - 1, 1)
                detail_progress = float(
                    _smoothstep(0.0, detail_end, timeline)
                )
                detail_reveal = _smoothstep(
                    -0.035,
                    0.035,
                    detail_progress - detail_arrival,
                )
                detail = (detail_alpha * detail_reveal)[..., None]
                frame = background * (1.0 - detail) + original * detail

                if timeline > fill_start:
                    fill_progress = float(
                        _smoothstep(fill_start, 1.0, timeline)
                    )
                    fill_softness = options.brush_feather * 0.22
                    fill = _smoothstep(
                        -fill_softness,
                        fill_softness,
                        fill_progress - fill_arrival,
                    )[..., None]
                    frame = frame * (1.0 - fill) + original * fill

                if timeline >= 0.999:
                    frame = original

            process.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
            progress((frame_index + 1) / total_frames)
    except (BrokenPipeError, OSError) as exc:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"FFmpeg interrumpió el render: {stderr[-1000:]}") from exc
    finally:
        process.stdin.close()

    stderr_bytes = process.stderr.read() if process.stderr else b""
    return_code = process.wait()
    if return_code != 0:
        temporary.unlink(missing_ok=True)
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg terminó con código {return_code}: {stderr[-1000:]}")
    os.replace(temporary, output_path)
    return width, height, total_frames
